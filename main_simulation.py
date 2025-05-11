# main_simulation.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# Quan trọng: Import lớp SupermarketMap và các hằng số ID từ file kia
from supermarket_model import SupermarketMap, PATHWAY_ID, OBSTACLE_ID, AP_ID, STALL_ID_START, STALL_ID_END, ITEM_ID_START

def plot_supermarket(supermarket: SupermarketMap, cart_pos=None, path=None, target_item_pos=None):
    """
    Vẽ bản đồ siêu thị, vị trí xe đẩy, đường đi và món hàng mục tiêu.
    supermarket: một instance của lớp SupermarketMap.
    """
    grid_map_to_plot = supermarket.grid_map
    num_rows, num_cols = grid_map_to_plot.shape
    
    plt.figure(figsize=(max(10, supermarket.width_m / 4), max(8, supermarket.height_m / 4)))

    unique_values = np.unique(grid_map_to_plot)
    
    color_list = []
    legend_patches = []

    # 0: Lối đi
    if PATHWAY_ID in unique_values:
        color_list.append('white')
        legend_patches.append(Patch(facecolor='white', edgecolor='black', label=f'Lối đi ({PATHWAY_ID})'))
    
    # 1: Vật cản chung
    if OBSTACLE_ID in unique_values:
        color_list.append('dimgray')
        legend_patches.append(Patch(facecolor='dimgray', edgecolor='black', label=f'Vật cản ({OBSTACLE_ID})'))

    # 2: AP (màu nền, AP sẽ được vẽ scatter riêng)
    # Chỉ thêm màu cho AP nếu có AP trên bản đồ và nó không bị đè bởi màu khác
    # (Điều này phức tạp, BoundaryNorm sẽ xử lý việc này dựa trên unique_values)
    ap_present_on_map = False
    if AP_ID in unique_values:
        # Kiểm tra xem AP_ID có thực sự được sử dụng trên bản đồ không (không phải chỉ trong unique_values)
        # Điều này quan trọng nếu AP_ID có trong unique_values nhưng không có ô nào thực sự là AP_ID
        # (ví dụ: tất cả AP đều cố gắng đặt lên chỗ bị chiếm)
        if np.any(grid_map_to_plot == AP_ID):
            ap_present_on_map = True
            color_list.append('lightcoral') # Màu nền cho ô AP
            # Legend cho AP sẽ được thêm từ scatter plot sau

    
    # Gian hàng
    stall_ids_on_map = sorted([val for val in unique_values if STALL_ID_START <= val <= STALL_ID_END])
    if stall_ids_on_map:
        stall_cmap = plt.cm.get_cmap('Pastel2', len(stall_ids_on_map))
        for i, stall_id in enumerate(stall_ids_on_map):
            color_list.append(stall_cmap(i))
            stall_name = supermarket.stall_definitions.get(stall_id, {}).get("name", f"Gian hàng {stall_id}")
            legend_patches.append(Patch(facecolor=stall_cmap(i), edgecolor='black', label=f'{stall_name} ({stall_id})'))

    # Mặt hàng
    item_ids_on_map = sorted([val for val in unique_values if val >= ITEM_ID_START])
    if item_ids_on_map:
        item_cmap = plt.cm.get_cmap('tab20', max(1,len(item_ids_on_map))) # Ensure at least 1 color for cmap
        for i, item_id in enumerate(item_ids_on_map):
            color_list.append(item_cmap(i))
            item_name = supermarket.item_definitions.get(item_id, {}).get("name", f"Mặt hàng {item_id}")
            legend_patches.append(Patch(facecolor=item_cmap(i), edgecolor='black', label=f'{item_name} ({item_id})'))
    
    # Tạo BoundaryNorm
    # Sắp xếp tất cả các unique_values làm ranh giới
    
    # --- ĐÂY LÀ CHỖ SỬA ---
    if unique_values.size > 0: # Chỉ tạo boundaries nếu có unique_values
        boundaries = np.concatenate([[-0.5], np.array(sorted(unique_values)) + 0.5])
    else: # Trường hợp bản đồ trống hoàn toàn
        boundaries = np.array([-0.5, 0.5]) # Ranh giới mặc định cho bản đồ trống

    # Loại bỏ các giá trị trùng lặp trong boundaries nếu có
    boundaries = np.unique(boundaries)


    if not color_list:
        cmap_display = ListedColormap(['lightgray']) # Màu mặc định nếu không có gì để vẽ
        # Đảm bảo boundaries hợp lệ cho norm ngay cả khi color_list trống
        if len(boundaries) < 2: boundaries = np.array([-0.5, 0.5])
        norm = BoundaryNorm(boundaries, cmap_display.N if cmap_display.N > 0 else 1)

    elif len(color_list) != (len(boundaries) - 1) :
        print(f"Warning: Mismatch between color list ({len(color_list)}) and boundaries-1 ({len(boundaries)-1}). This can happen if an ID (e.g. AP_ID) is in unique_values but no grid cell has that ID.")
        print(f"  Unique values on map: {unique_values}")
        print(f"  Constructed boundaries: {boundaries}")
        print(f"  Color list has {len(color_list)} colors.")

        # Xây dựng lại boundaries dựa trên các giá trị có màu
        # Điều này phức tạp hơn. Tạm thời, chúng ta có thể thử một cách tiếp cận khác:
        # Tạo một mapping từ unique_values sang index màu
        value_to_color_idx = {}
        current_color_idx = 0
        if PATHWAY_ID in unique_values:
            value_to_color_idx[PATHWAY_ID] = current_color_idx; current_color_idx+=1
        if OBSTACLE_ID in unique_values:
            value_to_color_idx[OBSTACLE_ID] = current_color_idx; current_color_idx+=1
        if ap_present_on_map: # Chỉ thêm nếu AP thực sự có trên map và có màu
             value_to_color_idx[AP_ID] = current_color_idx; current_color_idx+=1
        for stall_id in stall_ids_on_map:
            value_to_color_idx[stall_id] = current_color_idx; current_color_idx+=1
        for item_id in item_ids_on_map:
            value_to_color_idx[item_id] = current_color_idx; current_color_idx+=1
        
        # Bây giờ, color_list phải khớp với số lượng key trong value_to_color_idx
        if len(color_list) == len(value_to_color_idx):
            temp_grid = np.zeros_like(grid_map_to_plot, dtype=float)
            for r in range(num_rows):
                for c in range(num_cols):
                    val = grid_map_to_plot[r,c]
                    if val in value_to_color_idx:
                        temp_grid[r,c] = value_to_color_idx[val]
                    else: # Giá trị không mong muốn
                        temp_grid[r,c] = -1 # Hoặc một giá trị để bỏ qua

            cmap_display = ListedColormap(color_list)
            plt.imshow(temp_grid, cmap=cmap_display, origin='lower', interpolation='nearest', vmin=-0.5, vmax=len(color_list)-0.5)
            norm = None # Không dùng norm khi đã chuyển đổi giá trị
        else:
            print("  Fallback to generic cmap due to persistent mismatch.")
            cmap_display = plt.cm.get_cmap('nipy_spectral', int(np.max(grid_map_to_plot)) + 1 if unique_values.size > 0 else 2)
            norm = None 
            plt.imshow(grid_map_to_plot, cmap=cmap_display, norm=norm, origin='lower', interpolation='nearest')

    else: # Trường hợp lý tưởng: len(color_list) == len(boundaries) - 1
        cmap_display = ListedColormap(color_list)
        norm = BoundaryNorm(boundaries, cmap_display.N)
        plt.imshow(grid_map_to_plot, cmap=cmap_display, norm=norm, origin='lower', interpolation='nearest')


    # Đánh dấu APs
    # Chỉ vẽ APs nếu chúng thực sự tồn tại trong supermarket.access_points (đã được thêm thành công)
    if supermarket.access_points:
        ap_r_plot, ap_c_plot = zip(*supermarket.access_points)
        plt.scatter(ap_c_plot, ap_r_plot, marker='o', color='red', edgecolor='black', s=100, label='AP (Điểm)', zorder=5)
        # Thêm vào legend nếu chưa có
        if not any(isinstance(h, plt.Line2D) and h.get_label() == 'AP (Điểm)' for h in legend_patches):
             legend_patches.append(plt.Line2D([0], [0], marker='o', color='w', label='AP (Điểm)',
                                   markerfacecolor='red', markeredgecolor='black', markersize=10))


    if cart_pos:
        plt.scatter(cart_pos[1], cart_pos[0], marker='s', color='blue', edgecolor='black', s=150, label='Xe đẩy', zorder=5)
        if not any(isinstance(h, plt.Line2D) and h.get_label() == 'Xe đẩy' for h in legend_patches):
            legend_patches.append(plt.Line2D([0], [0], marker='s', color='w', label='Xe đẩy',
                                   markerfacecolor='blue', markeredgecolor='black', markersize=10))
    if target_item_pos:
        item_name_at_target = supermarket.get_item_name_at_location(target_item_pos[0], target_item_pos[1])
        label_target = f'{item_name_at_target} (Mục tiêu)' if item_name_at_target else "Mục tiêu không rõ"
        plt.scatter(target_item_pos[1], target_item_pos[0], marker='*', color='magenta',edgecolor='black', s=250, label=label_target, zorder=5)
        if not any(isinstance(h, plt.Line2D) and h.get_label() == label_target for h in legend_patches):
             legend_patches.append(plt.Line2D([0], [0], marker='*', color='w', label=label_target,
                                   markerfacecolor='magenta',markeredgecolor='black', markersize=15))
    if path:
        if len(path) > 1 : # Ensure path has at least two points to draw a line
            path_rows, path_cols = zip(*path)
            plt.plot(path_cols, path_rows, color='cyan', linewidth=2, label='Đường đi', zorder=4)
            if not any(isinstance(h, plt.Line2D) and h.get_label() == 'Đường đi' for h in legend_patches):
                legend_patches.append(plt.Line2D([0], [0], color='cyan', lw=2, label='Đường đi'))
        elif len(path) == 1: # If path is a single point, scatter it
             plt.scatter(path[0][1], path[0][0], color='cyan', marker='x', s=50, label='Điểm bắt đầu đường đi', zorder=4)


    plt.xticks(np.arange(-0.5, num_cols, step=max(1, num_cols // 10)),
               [f"{i*supermarket.resolution_m:.0f}" for i in np.arange(0, num_cols + supermarket.resolution_m, step=max(1, num_cols // 10))])
    plt.yticks(np.arange(-0.5, num_rows, step=max(1, num_rows // 10)),
               [f"{i*supermarket.resolution_m:.0f}" for i in np.arange(0, num_rows + supermarket.resolution_m, step=max(1, num_rows // 10))])
    plt.xlabel("Chiều rộng (mét)")
    plt.ylabel("Chiều cao (mét)")
    plt.title("Bản đồ Siêu thị Mô phỏng (OOP)")
    plt.grid(True, which='major', color='lightgray', linestyle='-', linewidth=0.5)
    plt.xlim([-0.5, num_cols - 0.5])
    plt.ylim([num_rows - 0.5, -0.5]) # Invert Y axis

    plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize='small')
    plt.tight_layout(rect=[0, 0, 0.82, 1]) # Adjust to prevent legend cutoff
    plt.show()


if __name__ == "__main__":
    # --- KHỞI TẠO VÀ THIẾT LẬP SIÊU THỊ ---
    supermarket = SupermarketMap(width_m=50, height_m=30, resolution_m=0.5)

    # Thêm vật cản chung
    supermarket.add_general_obstacle(r_start=0, c_start=0, obs_height=1, obs_width=supermarket.num_cols) # Tường trên
    supermarket.add_general_obstacle(r_start=supermarket.num_rows-1, c_start=0, obs_height=1, obs_width=supermarket.num_cols) # Tường dưới
    supermarket.add_general_obstacle(r_start=0, c_start=0, obs_height=supermarket.num_rows, obs_width=1) # Tường trái
    

    # Thêm gian hàng
    stall1_r, stall1_c, stall1_h, stall1_w = 6, 1, 20, 4
    id_fresh = supermarket.add_stall_area(stall1_r, stall1_c, stall1_h, stall1_w, "Trái Cây 1")

    stall3_r, stall3_c, stall3_h, stall3_w = 6, 14, 20, 6
    id_fresh_2 = supermarket.add_stall_area(stall3_r, stall3_c, stall3_h, stall3_w, "Trái Cây")

    stall4_r, stall4_c, stall4_h, stall4_w = 6, 28, 20, 6
    id_fresh_3 = supermarket.add_stall_area(stall4_r, stall4_c, stall4_h, stall4_w, "Rau")

    stall2_r, stall2_c, stall2_h, stall2_w = 38, 15, 10, 4
    id_drinks = supermarket.add_stall_area(stall2_r, stall2_c, stall2_h, stall2_w, "Đồ Uống")

    stall5_r, stall5_c, stall5_h, stall5_w = 1, 60, 4, 27
    id_meats = supermarket.add_stall_area(stall5_r, stall5_c, stall5_h, stall5_w, "Thịt")

    stall6_r, stall6_c, stall6_h, stall6_w = 35, 65, 16, 16
    id_cakes = supermarket.add_stall_area(stall6_r, stall6_c, stall6_h, stall6_w, "Bánh")

    # Thêm mặt hàng vào gian hàng
    if id_fresh != -1:
        supermarket.add_item_to_grid(stall1_r, stall1_c, item_height=5, item_width=stall1_w, item_name="Táo")
        supermarket.add_item_to_grid(stall1_r + 5, stall1_c, item_height=5, item_width=stall1_w, item_name="Chuối")
        supermarket.add_item_to_grid(stall1_r + 10, stall1_c, item_height=5, item_width=stall1_w, item_name="Bơ")
        supermarket.add_item_to_grid(stall1_r + 15, stall1_c, item_height=5, item_width=stall1_w, item_name="Dứa")

    if id_fresh_2 != -1:
        supermarket.add_item_to_grid(stall3_r, stall3_c, item_height=5, item_width=stall3_w, item_name="Nho")
        supermarket.add_item_to_grid(stall3_r + 5, stall3_c, item_height=5, item_width=stall3_w, item_name="Măng Cụt")
        supermarket.add_item_to_grid(stall3_r + 10, stall3_c, item_height=5, item_width=stall3_w, item_name="Thơm")
        supermarket.add_item_to_grid(stall3_r + 15, stall3_c, item_height=5, item_width=stall3_w, item_name="Xoài")

    if id_fresh_3 != -1:
        supermarket.add_item_to_grid(stall4_r, stall4_c, item_height=5, item_width=stall4_w, item_name="Rau Muống")
        supermarket.add_item_to_grid(stall4_r + 5, stall4_c, item_height=5, item_width=stall4_w, item_name="Rau Dền")
        supermarket.add_item_to_grid(stall4_r + 10, stall4_c, item_height=5, item_width=stall4_w, item_name="Cải Xanh")
        supermarket.add_item_to_grid(stall4_r + 15, stall4_c, item_height=5, item_width=stall4_w, item_name="Mồng Tơi")

    if id_meats != -1:
        supermarket.add_item_to_grid(stall5_r, stall5_c, item_height=stall5_h, item_width=9, item_name="Thịt Gà")
        supermarket.add_item_to_grid(stall5_r, stall5_c + 9, item_height=stall5_h, item_width=9, item_name="Thịt Heo")
        supermarket.add_item_to_grid(stall5_r, stall5_c + 18, item_height=stall5_h, item_width=9, item_name="Thịt Bò")

    if id_cakes != -1:
        supermarket.add_item_to_grid(stall6_r, stall6_c, item_height=stall6_h//2, item_width=stall6_w//2, item_name="Bánh Quy")
        supermarket.add_item_to_grid(stall6_r + stall6_w//2, stall6_c, item_height=stall6_h//2, item_width=stall6_w//2, item_name="Bánh Bao")
        supermarket.add_item_to_grid(stall6_r + stall6_h//2, stall6_c + stall6_w//2, item_height=stall6_h//2, item_width=stall6_w//2, item_name="Bánh Mì")
        supermarket.add_item_to_grid(stall6_r, stall6_c + stall6_w//2, item_height=stall6_h//2, item_width=stall6_w//2, item_name="Bánh Xếp")
          

    if id_drinks != -1:
        supermarket.add_item_to_grid(stall2_r, stall2_c, item_height=stall2_h, item_width=stall2_w // 2, item_name="Nước Ngọt")
        supermarket.add_item_to_grid(stall2_r, stall2_c + stall2_w // 2, item_height=stall2_h, item_width=stall2_w // 2, item_name="Nước Suối")

    # Thêm APs
    ap_locations = [
        (2, 2), (2, supermarket.num_cols - 3),
        (supermarket.num_rows - 3, 2), (supermarket.num_rows - 3, supermarket.num_cols - 3),
        (supermarket.num_rows // 2, supermarket.num_cols // 2)
    ]
    for r, c in ap_locations:
        supermarket.add_access_point(r, c)
    
    # Vẽ bản đồ ban đầu
    plot_supermarket(supermarket)

    # --- VÍ DỤ TƯƠNG TÁC ---
    cart_start_pos = (supermarket.num_rows - 5, 5)
    target_item_name = "Nước Ngọt"
    
    item_cells = supermarket.get_item_locations_by_name(target_item_name)
    
    if item_cells:
        target_pos_on_grid = item_cells[0] # Lấy vị trí đầu tiên của mặt hàng
        print(f"Mặt hàng '{target_item_name}' được tìm thấy tại: {target_pos_on_grid}")

        # Tạo đường đi giả định (cần thuật toán tìm đường thực tế)
        path_to_target = []
        curr_r, curr_c = cart_start_pos
        target_r, target_c = target_pos_on_grid
        path_to_target.append((curr_r, curr_c))
        
        # Di chuyển theo cột trước, rồi theo hàng (đơn giản, không tránh vật cản)
        # Đảm bảo không tạo đường đi lặp lại vô hạn nếu curr_c/curr_r đã bằng target_c/target_r
        while curr_c != target_c:
            prev_c = curr_c
            curr_c += 1 if target_c > curr_c else -1
            path_to_target.append((curr_r, curr_c))
            if prev_c == curr_c: # Safety break
                print("Warning: Pathfinding for C stuck.")
                break
        while curr_r != target_r:
            prev_r = curr_r
            curr_r += 1 if target_r > curr_r else -1
            path_to_target.append((curr_r, curr_c))
            if prev_r == curr_r: # Safety break
                print("Warning: Pathfinding for R stuck.")
                break
        
        plot_supermarket(supermarket, cart_pos=cart_start_pos, path=path_to_target, target_item_pos=target_pos_on_grid)
    else:
        print(f"Không tìm thấy mặt hàng '{target_item_name}'.")