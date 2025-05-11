# visualization.py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import config

class InteractiveMap:
    def __init__(self, grid_map, access_points, item_locations,
                 rssi_fingerprints_data, num_rows, num_cols):
        self.grid_map_data = grid_map # Dữ liệu 0, 1
        self.access_points = access_points
        self.item_locations_dict = item_locations # item_name -> list of (r,c)
        self.rssi_fingerprints_data = rssi_fingerprints_data
        self.num_rows = num_rows
        self.num_cols = num_cols

        self.fig, self.ax = plt.subplots(figsize=(
            self.num_cols / (10 / config.GRID_RESOLUTION_M) / 1.1,
            self.num_rows / (10 / config.GRID_RESOLUTION_M) / 1.1
        ))
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        self.cart_actual_pos_grid = None
        self.cart_estimated_pos_float = None
        self.target_item_name = None
        self.target_item_pos_grid = None
        self.current_path_nodes = None # Sẽ lưu trữ các node (hàng, cột)
        self.error_m = None

        # Tạo colormap tùy chỉnh: 0 (lối đi) -> trắng, 1 (kệ) -> xám
        self.custom_cmap = mcolors.ListedColormap([config.COLOR_PATH_ON_MAP, config.COLOR_SHELF_ON_MAP])
        self.bounds = [-0.5, 0.5, 1.5] # Biên cho 2 màu
        self.norm = mcolors.BoundaryNorm(self.bounds, self.custom_cmap.N)


        self.plot_initial_map()

    def _grid_to_metric(self, r_or_list_r, c_or_list_c=None):
        """Chuyển đổi tọa độ ô lưới sang tọa độ mét để vẽ (lấy tâm ô)."""
        if c_or_list_c is None: # Nếu truyền vào list of tuples (r,c)
            if not r_or_list_r: return np.array([]), np.array([])
            coords_m = np.array([(c * config.GRID_RESOLUTION_M + config.GRID_RESOLUTION_M / 2,
                                  r * config.GRID_RESOLUTION_M + config.GRID_RESOLUTION_M / 2)
                                 for r, c in r_or_list_r])
            return coords_m[:, 1], coords_m[:, 0] # trả về (list_y_m, list_x_m)
        else: # Nếu truyền vào r và c riêng lẻ
            x_m = c_or_list_c * config.GRID_RESOLUTION_M + config.GRID_RESOLUTION_M / 2
            y_m = r_or_list_r * config.GRID_RESOLUTION_M + config.GRID_RESOLUTION_M / 2
            return y_m, x_m


    def plot_initial_map(self):
        self.ax.clear()
        self.ax.imshow(self.grid_map_data, cmap=self.custom_cmap, norm=self.norm,
                       origin='lower', interpolation='nearest',
                       extent=[0, self.num_cols * config.GRID_RESOLUTION_M,
                               0, self.num_rows * config.GRID_RESOLUTION_M])

        if self.access_points:
            aps_y_m, aps_x_m = self._grid_to_metric(self.access_points)
            self.ax.scatter(aps_x_m, aps_y_m, marker='o',
                            color=config.COLOR_AP_MARKER, s=100, label='AP', zorder=5)

        labeled_items = set()
        for item_name, loc_list in self.item_locations_dict.items():
            if loc_list and item_name not in labeled_items:
                # Chỉ vẽ điểm tiếp cận đầu tiên cho mỗi món hàng trên bản đồ ban đầu
                item_r, item_c = loc_list[0]
                item_y_m, item_x_m = self._grid_to_metric(item_r, item_c)
                self.ax.scatter(item_x_m, item_y_m, marker='D', color='orange', s=80,
                                label=f'{item_name} (điểm tiếp cận)', zorder=5)
                labeled_items.add(item_name)

        self.ax.set_title("Bản đồ Siêu thị Tương tác (Click lối đi để đặt xe đẩy)")
        self.ax.set_xlabel(f"Chiều rộng (mét)")
        self.ax.set_ylabel(f"Chiều cao (mét)")
        self.ax.invert_yaxis()
        self.ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
        self.ax.grid(True, which='both', color='lightgray', linestyle=':', linewidth=0.5)
        self.fig.tight_layout(rect=[0, 0, 0.83, 1])
        self.fig.canvas.draw_idle()

    def update_plot_elements(self):
        # Xóa các đối tượng động cũ (nếu có)
        for artist in self.ax.collections: # scatter plots
            if artist.get_label() and "AP" not in artist.get_label() and "điểm tiếp cận" not in artist.get_label():
                artist.remove()
        for line in self.ax.lines: # lines (đường đi, đường lỗi)
            line.remove()

        # Vẽ lại các đối tượng động
        if self.cart_actual_pos_grid:
            cart_y_m, cart_x_m = self._grid_to_metric(*self.cart_actual_pos_grid)
            self.ax.scatter(cart_x_m, cart_y_m, marker='s',
                            color=config.COLOR_CART_ACTUAL_MARKER, s=150, label='Xe đẩy (Thực tế)', zorder=10)

        if self.cart_estimated_pos_float:
            est_y_m, est_x_m = self._grid_to_metric(*self.cart_estimated_pos_float)
            self.ax.scatter(est_x_m, est_y_m, marker='P',
                            color=config.COLOR_CART_ESTIMATED_MARKER, s=150,
                            label=f'Xe đẩy (KNN K={config.K_NEIGHBORS})', zorder=10)
            if self.cart_actual_pos_grid and self.error_m is not None:
                actual_y_m, actual_x_m = self._grid_to_metric(*self.cart_actual_pos_grid)
                self.ax.plot([actual_x_m, est_x_m], [actual_y_m, est_y_m],
                             color=config.COLOR_ERROR_LINE, linestyle='--', linewidth=1.5,
                             label=f'Sai số: {self.error_m:.2f}m', zorder=8)

        if self.target_item_pos_grid:
            target_y_m, target_x_m = self._grid_to_metric(*self.target_item_pos_grid)
            self.ax.scatter(target_x_m, target_y_m, marker='*',
                            color=config.COLOR_TARGET_ITEM_MARKER, s=250,
                            label=f'Đến: {self.target_item_name}', zorder=10)

        if self.current_path_nodes:
            path_y_m, path_x_m = self._grid_to_metric(self.current_path_nodes)
            if len(path_x_m) > 0: # Chỉ vẽ nếu có đường đi
                self.ax.plot(path_x_m, path_y_m, color=config.COLOR_PATH_LINE,
                             linewidth=3, label='Đường đi', zorder=7)

        # Cập nhật lại legend
        handles, labels = self.ax.get_legend_handles_labels()
        unique_labels = {}
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels[label] = handle
        self.ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', bbox_to_anchor=(1.35, 1))

        self.fig.canvas.draw_idle()


    def onclick(self, event):
        if event.inaxes != self.ax or event.button != 1: # Chỉ xử lý click chuột trái trong axes
            return

        clicked_x_m, clicked_y_m = event.xdata, event.ydata
        if clicked_x_m is None or clicked_y_m is None: return # Click ngoài vùng dữ liệu

        clicked_c = int(clicked_x_m / config.GRID_RESOLUTION_M)
        clicked_r = int(clicked_y_m / config.GRID_RESOLUTION_M)

        if 0 <= clicked_r < self.num_rows and 0 <= clicked_c < self.num_cols:
            if self.grid_map_data[clicked_r, clicked_c] == config.CELL_TYPE_SHELF:
                print(f"Bạn đã click vào kệ hàng tại ô ({clicked_r}, {clicked_c}). Vui lòng click vào lối đi.")
                return

            self.cart_actual_pos_grid = (clicked_r, clicked_c)
            print(f"\nĐã đặt xe đẩy tại vị trí thực tế (ô lưới): {self.cart_actual_pos_grid} "
                  f"({self.cart_actual_pos_grid[0]*config.GRID_RESOLUTION_M:.1f}m, "
                  f"{self.cart_actual_pos_grid[1]*config.GRID_RESOLUTION_M:.1f}m)")

            self.target_item_name = None
            self.target_item_pos_grid = None
            self.current_path_nodes = None
            self.error_m = None
            self.cart_estimated_pos_float = None # Xóa vị trí ước tính cũ

            self.update_plot_elements() # Cập nhật plot chỉ với vị trí xe đẩy thực tế

            if hasattr(self, 'on_map_click_callback'):
                self.on_map_click_callback(self.cart_actual_pos_grid)
        else:
            print("Click ra ngoài bản đồ.")

def create_and_show_interactive_map(grid_map, access_points, item_locations,
                                    rssi_fingerprints_data, num_rows, num_cols, on_map_click_func):
    interactive_plot = InteractiveMap(grid_map, access_points, item_locations,
                                      rssi_fingerprints_data, num_rows, num_cols)
    interactive_plot.on_map_click_callback = on_map_click_func
    return interactive_plot # Trả về đối tượng để main có thể gọi plt.show()