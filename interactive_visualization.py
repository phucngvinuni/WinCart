# interactive_visualization.py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import config
from supermarket_model import PATHWAY_ID, OBSTACLE_ID, AP_ID, STALL_ID_START, STALL_ID_END, ITEM_ID_START
import matplotlib.colors as mcolors
class InteractiveSupermarketPlotter:
    def __init__(self, supermarket_obj, on_map_click_callback_func=None):
        self.supermarket = supermarket_obj
        self.on_map_click_callback = on_map_click_callback_func

        self.fig, self.ax = plt.subplots(figsize=(
            max(10, self.supermarket.width_m / 3), # Điều chỉnh kích thước plot
            max(8, self.supermarket.height_m / 3)
        ))
        # Ngắt kết nối các sự kiện click mặc định nếu có
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id) # Example
        for event_name in ['button_press_event', 'button_release_event', 'motion_notify_event']:
            # Thử ngắt kết nối handler cũ nếu có
            if hasattr(self.fig.canvas, f"_callbacks") and event_name in self.fig.canvas._callbacks:
                 for cid_val in list(self.fig.canvas._callbacks[event_name]): # iterate over a copy
                        self.fig.canvas.mpl_disconnect(cid_val)


        self.click_event_handler_id = self.fig.canvas.mpl_connect('button_press_event', self._handle_mouse_click)


        self.cart_actual_pos_rc = None       # (row, col)
        self.cart_estimated_pos_rc_float = None # (float_row, float_col)
        self.target_item_name_display = None
        self.target_approachable_pos_rc = None # (row, col)
        self.current_path_rc_nodes = None    # List of (row, col)
        self.localization_error_m = None
        self.current_message_on_plot = "Click vào lối đi để đặt xe đẩy."

        self._dynamic_artists = [] # List để lưu các artist cần xóa và vẽ lại

        self._setup_static_plot_elements()
        self.update_dynamic_plot_elements() # Vẽ trạng thái ban đầu của các phần động

    def _grid_rc_to_plot_xy(self, r_or_list_r, c_or_list_c=None):
        """Chuyển đổi (hàng, cột) lưới sang (x,y) để vẽ (tâm ô)."""
        res = self.supermarket.resolution_m
        if c_or_list_c is None: # list of tuples (r,c)
            if not r_or_list_r: return np.array([]), np.array([])
            plot_coords = np.array([(c * res + res / 2, r * res + res / 2) for r, c in r_or_list_r])
            return plot_coords[:, 0], plot_coords[:, 1] # (list_x, list_y)
        else: # r và c riêng lẻ
            plot_x = c_or_list_c * res + res / 2
            plot_y = r_or_list_r * res + res / 2
            return plot_x, plot_y

    def _plot_xy_to_grid_rc(self, plot_x, plot_y):
        """Chuyển đổi (x,y) từ plot (mét) sang (hàng, cột) lưới."""
        res = self.supermarket.resolution_m
        grid_c = int(plot_x / res)
        grid_r = int(plot_y / res) # Matplotlib origin 'lower' nên y_plot tăng từ dưới lên
        # Điều chỉnh nếu origin của imshow là 'upper'
        # grid_r = int((self.supermarket.height_m - plot_y) / res)
        return grid_r, grid_c


    def _setup_static_plot_elements(self):
        self.ax.clear()
        grid_to_display = self.supermarket.grid_map
        num_r_map, num_c_map = grid_to_display.shape

        # --- Tạo Colormap và Legend động ---
        unique_ids_on_map = np.unique(grid_to_display)
        colors_for_map = []
        self.legend_handles = []

        id_to_color_map = {}
        # Lối đi
        if PATHWAY_ID in unique_ids_on_map:
            id_to_color_map[PATHWAY_ID] = config.COLOR_PATH_ON_MAP
            self.legend_handles.append(Patch(facecolor=config.COLOR_PATH_ON_MAP, edgecolor='gray', label='Lối đi'))
        # Vật cản chung
        if OBSTACLE_ID in unique_ids_on_map:
            id_to_color_map[OBSTACLE_ID] = config.COLOR_OBSTACLE_ON_MAP
            self.legend_handles.append(Patch(facecolor=config.COLOR_OBSTACLE_ON_MAP, edgecolor='black', label='Vật cản'))
        # AP (nền ô)
        if AP_ID in unique_ids_on_map and np.any(grid_to_display == AP_ID):
            id_to_color_map[AP_ID] = config.COLOR_AP_CELL_ON_MAP
            # Legend cho AP marker sẽ được thêm riêng

        # Gian hàng
        stall_ids = sorted([uid for uid in unique_ids_on_map if STALL_ID_START <= uid <= STALL_ID_END])
        if stall_ids:
            stall_cmap_palette = plt.cm.get_cmap('terrain', len(stall_ids) if len(stall_ids)>0 else 1) # Đổi cmap
            for i, s_id in enumerate(stall_ids):
                color = stall_cmap_palette(i / max(1, len(stall_ids)-1) if len(stall_ids)>1 else 0.5) # Phân bố màu
                id_to_color_map[s_id] = color
                s_name = self.supermarket.stall_definitions.get(s_id, {}).get("name", f"Gian hàng {s_id}")
                self.legend_handles.append(Patch(facecolor=color, edgecolor='black', label=s_name))
        # Mặt hàng
        item_ids = sorted([uid for uid in unique_ids_on_map if uid >= ITEM_ID_START])
        if item_ids:
            item_cmap_palette = plt.cm.get_cmap('Pastel1', len(item_ids) if len(item_ids)>0 else 1) # Đổi cmap
            for i, i_id in enumerate(item_ids):
                color = item_cmap_palette(i / max(1, len(item_ids)-1) if len(item_ids)>1 else 0.5)
                id_to_color_map[i_id] = color
                i_name = self.supermarket.item_definitions.get(i_id, {}).get("name", f"Mặt hàng {i_id}")
                self.legend_handles.append(Patch(facecolor=color, edgecolor='black', label=i_name))
        
        # Tạo mảng màu dựa trên id_to_color_map
        # và một grid_map mới chứa index màu thay vì ID
        map_ids_sorted = sorted(list(id_to_color_map.keys()))
        if not map_ids_sorted: # Bản đồ hoàn toàn trống
            final_cmap = mcolors.ListedColormap(['lightgray'])
            final_norm = mcolors.BoundaryNorm([-0.5, 0.5], final_cmap.N)
            self.ax.imshow(grid_to_display, cmap=final_cmap, norm=final_norm, origin='lower', interpolation='nearest',
                           extent=[0, num_c_map * self.supermarket.resolution_m,
                                   0, num_r_map * self.supermarket.resolution_m])
        else:
            color_palette_ordered = [id_to_color_map[id_val] for id_val in map_ids_sorted]
            final_cmap = mcolors.ListedColormap(color_palette_ordered)
            
            boundaries = np.concatenate([[-0.5], np.array(map_ids_sorted) + 0.5])
            # Xử lý trường hợp map_ids_sorted có các số không liên tiếp
            # Chúng ta cần boundaries cho mỗi màu. Nếu map_ids_sorted = [0, 50, 100]
            # boundaries sẽ là [-0.5, 0.5, 49.5, 50.5, 99.5, 100.5]
            # Hoặc cách đơn giản hơn: tạo temp_grid với các giá trị 0, 1, 2...
            value_to_color_index_for_plot = {val: i for i, val in enumerate(map_ids_sorted)}
            temp_display_grid = np.zeros_like(grid_to_display, dtype=int)
            for r_idx in range(num_r_map):
                for c_idx in range(num_c_map):
                    original_val = grid_to_display[r_idx, c_idx]
                    temp_display_grid[r_idx, c_idx] = value_to_color_index_for_plot.get(original_val, -1) # -1 cho các giá trị không có trong map

            self.ax.imshow(temp_display_grid, cmap=final_cmap, origin='lower', interpolation='nearest',
                           extent=[0, num_c_map * self.supermarket.resolution_m,
                                   0, num_r_map * self.supermarket.resolution_m],
                           vmin = -0.5, vmax = len(color_palette_ordered) - 0.5)


        # Vẽ AP markers (tĩnh)
        if self.supermarket.access_points:
            aps_x_m, aps_y_m = self._grid_rc_to_plot_xy(self.supermarket.access_points)
            self.ax.scatter(aps_x_m, aps_y_m, marker='o', facecolor=config.COLOR_AP_MARKER,
                            edgecolor='black', s=120, label='AP', zorder=10)
            # Thêm vào legend nếu chưa có
            if not any(artist.get_label() == 'AP' for artist in self.legend_handles if isinstance(artist, Patch) or isinstance(artist, plt.Line2D)):
                 self.legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='AP',
                                       markerfacecolor=config.COLOR_AP_MARKER, markeredgecolor='black', markersize=10))


        self.ax.set_xlabel("Chiều rộng (mét)")
        self.ax.set_ylabel("Chiều cao (mét)")
        self.ax.invert_yaxis() # Gốc (0,0) ở trên cùng bên trái cho imshow
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True, which='major', color='gainsboro', linestyle='-', linewidth=0.5)

        # Đặt ticks dựa trên kích thước mét
        x_ticks_m = np.arange(0, self.supermarket.width_m + self.supermarket.resolution_m,
                              step=max(self.supermarket.resolution_m, round(self.supermarket.width_m / 10)))
        y_ticks_m = np.arange(0, self.supermarket.height_m + self.supermarket.resolution_m,
                              step=max(self.supermarket.resolution_m, round(self.supermarket.height_m / 10)))
        self.ax.set_xticks(x_ticks_m)
        self.ax.set_yticks(y_ticks_m)
        self.ax.set_xlim([0, self.supermarket.width_m])
        self.ax.set_ylim([self.supermarket.height_m, 0]) # Y đảo ngược


    def update_dynamic_plot_elements(self, message_to_display=None):
        # Xóa các artists động cũ
        for artist in self._dynamic_artists:
            artist.remove()
        self._dynamic_artists = []

        current_title = "Bản đồ Siêu thị Tương tác"
        if message_to_display:
            current_title += f"\n{message_to_display}"
        elif self.current_message_on_plot:
            current_title += f"\n{self.current_message_on_plot}"
        self.ax.set_title(current_title)


        # Vẽ xe đẩy thực tế
        if self.cart_actual_pos_rc:
            cart_x_m, cart_y_m = self._grid_rc_to_plot_xy(*self.cart_actual_pos_rc)
            artist = self.ax.scatter(cart_x_m, cart_y_m, marker_size=180, marker_face_color=config.COLOR_CART_ACTUAL_MARKER,
                                     edge_color='black', s=180, label='Xe đẩy (Thực tế)', zorder=12) #sửa marker
            self._dynamic_artists.append(artist)

        # Vẽ xe đẩy ước tính và đường lỗi
        if self.cart_estimated_pos_rc_float and self.cart_actual_pos_rc:
            est_x_m, est_y_m = self._grid_rc_to_plot_xy(*self.cart_estimated_pos_rc_float)
            artist = self.ax.scatter(est_x_m, est_y_m, marker_size=180, marker_face_color=config.COLOR_CART_ESTIMATED_MARKER, #sửa marker
                                     edge_color='black', s=180, label=f'Xe đẩy (KNN K={config.K_NEIGHBORS})', zorder=12)
            self._dynamic_artists.append(artist)

            if self.localization_error_m is not None:
                act_x_m, act_y_m = self._grid_rc_to_plot_xy(*self.cart_actual_pos_rc)
                line, = self.ax.plot([act_x_m, est_x_m], [act_y_m, est_y_m],
                                      color=config.COLOR_ERROR_LINE, linestyle='--', linewidth=1.5,
                                      label=f'Sai số: {self.localization_error_m:.2f}m', zorder=11)
                self._dynamic_artists.append(line)

        # Vẽ mục tiêu
        if self.target_approachable_pos_rc:
            target_x_m, target_y_m = self._grid_rc_to_plot_xy(*self.target_approachable_pos_rc)
            label_target = f"Đến: {self.target_item_name_display}" if self.target_item_name_display else "Mục tiêu"
            artist = self.ax.scatter(target_x_m, target_y_m, marker='*', facecolor=config.COLOR_TARGET_ITEM_MARKER, #sửa marker
                                     edgecolor='black', s=300, label=label_target, zorder=12)
            self._dynamic_artists.append(artist)

        # Vẽ đường đi
        if self.current_path_rc_nodes:
            path_x_m, path_y_m = self._grid_rc_to_plot_xy(self.current_path_rc_nodes)
            if len(path_x_m) > 0:
                line, = self.ax.plot(path_x_m, path_y_m, color=config.COLOR_PATH_LINE,
                                     linewidth=3.5, label='Đường đi', zorder=9, alpha=0.8)
                self._dynamic_artists.append(line)

        # Cập nhật legend động (chỉ các label mới)
        current_handles, current_labels = self.ax.get_legend_handles_labels()
        # Kết hợp với legend tĩnh đã có
        final_handles = self.legend_handles[:] # Copy
        final_labels = [h.get_label() for h in self.legend_handles]

        for h, l in zip(current_handles, current_labels):
            if l and l not in final_labels: # Chỉ thêm label mới và có tên
                 # Kiểm tra xem handle có thực sự là artist không (tránh lỗi với text)
                if isinstance(h, (plt.Line2D, plt.collections.PathCollection, Patch)):
                    final_handles.append(h)
                    final_labels.append(l)
        
        # Loại bỏ các legend trùng lặp dựa trên label
        unique_legend_items = {}
        for handle, label in zip(final_handles, final_labels):
            if label not in unique_legend_items:
                unique_legend_items[label] = handle
        
        if unique_legend_items:
             self.ax.legend(unique_legend_items.values(), unique_legend_items.keys(),
                           loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small')
        else: # Nếu không có gì trong legend (rất hiếm)
            current_legend = self.ax.get_legend()
            if current_legend:
                current_legend.remove()


        self.fig.canvas.draw_idle()


    def _handle_mouse_click(self, event):
        if event.inaxes != self.ax or event.button != 1: return
        if event.xdata is None or event.ydata is None: return # Click ngoài vùng dữ liệu

        # Chuyển đổi tọa độ click (mét) sang ô lưới (hàng, cột)
        # Lưu ý: origin='lower' của imshow nghĩa là ydata tăng từ dưới lên
        # nhưng self.supermarket.grid_map[row, col] thì row tăng từ trên xuống.
        # Cần điều chỉnh cho đúng.
        # Nếu self.ax.invert_yaxis() đã được gọi, thì ydata sẽ giảm từ trên xuống.

        plot_x_m, plot_y_m = event.xdata, event.ydata
        
        # Chuyển đổi từ tọa độ plot (mét) sang ô lưới (hàng, cột)
        # Do Y axis bị đảo ngược, phép tính cho hàng cần cẩn thận
        clicked_c = int(plot_x_m / self.supermarket.resolution_m)
        # Nếu Y axis bị đảo ngược (ylim là [cao, thấp]), thì ydata của click sẽ là giá trị từ trên xuống
        clicked_r = int(plot_y_m / self.supermarket.resolution_m) # Nếu origin='lower' và không invert_yaxis
        # Nếu Y axis đã invert (ylim là [cao, thấp]) và origin='lower' thì ydata của click là giá trị từ trên xuống
        # clicked_r = int(plot_y_m / self.supermarket.resolution_m) # This should be correct if ylim is inverted

        # Kiểm tra biên một lần nữa
        if not (0 <= clicked_r < self.supermarket.num_rows and 0 <= clicked_c < self.supermarket.num_cols):
            print("Click ngoài bản đồ (sau khi chuyển đổi).")
            return

        if self.supermarket.grid_map[clicked_r, clicked_c] != PATHWAY_ID and \
           self.supermarket.grid_map[clicked_r, clicked_c] != AP_ID : # Cho phép click lên ô AP
            print(f"Bạn đã click vào ô không phải lối đi (ID: {self.supermarket.grid_map[clicked_r, clicked_c]}) "
                  f"tại ({clicked_r}, {clicked_c}). Vui lòng click vào lối đi.")
            return

        #self.cart_actual_pos_rc = (clicked_r, clicked_c) # Sẽ được set bởi callback
        print(f"\nSự kiện Click: Ô ({clicked_r}, {clicked_c}) ~ ({plot_y_m:.1f}m, {plot_x_m:.1f}m)")

        # Xóa trạng thái cũ trước khi gọi callback
        self.cart_estimated_pos_rc_float = None
        self.target_item_name_display = None
        self.target_approachable_pos_rc = None
        self.current_path_rc_nodes = None
        self.localization_error_m = None
        self.current_message_on_plot = "Đang xử lý yêu cầu sau click..."
        # self.update_dynamic_plot_elements() # Để callback xử lý việc vẽ lại

        if self.on_map_click_callback:
            self.on_map_click_callback((clicked_r, clicked_c)) # Truyền vị trí click (ô lưới)

    def update_cart_location(self, actual_rc, estimated_rc_float, error_m, message=None):
        self.cart_actual_pos_rc = actual_rc
        self.cart_estimated_pos_rc_float = estimated_rc_float
        self.localization_error_m = error_m
        self.current_message_on_plot = message if message else "Đã cập nhật vị trí xe đẩy."
        self.update_dynamic_plot_elements()

    def update_path_to_target(self, target_name, target_approachable_rc, path_rc_nodes, message=None):
        self.target_item_name_display = target_name
        self.target_approachable_pos_rc = target_approachable_rc
        self.current_path_rc_nodes = path_rc_nodes
        self.current_message_on_plot = message if message else f"Đang dẫn đường đến {target_name}."
        self.update_dynamic_plot_elements()

    def clear_path_and_target(self, message=None):
        self.target_item_name_display = None
        self.target_approachable_pos_rc = None
        self.current_path_rc_nodes = None
        self.current_message_on_plot = message if message else "Đã xóa mục tiêu và đường đi."
        # Giữ lại vị trí xe đẩy
        self.update_dynamic_plot_elements()

    def show(self):
        plt.show()