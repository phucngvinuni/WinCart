# main_interactive_simulation.py
# ... (các import và hàm plot_supermarket giữ nguyên như trước) ...
# main_interactive_simulation.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import speech_recognition as sr # Thêm import này
import re # Thêm thư viện regex để tách từ tốt hơn

# Quan trọng: Import lớp SupermarketMap và các hằng số ID từ file kia
from supermarket_model import SupermarketMap, PATHWAY_ID, OBSTACLE_ID, AP_ID, STALL_ID_START, STALL_ID_END, ITEM_ID_START
# --- HÀM VẼ BẢN ĐỒ (Giữ nguyên từ trước) ---
def plot_supermarket(supermarket: SupermarketMap, cart_pos=None, path=None, target_item_pos=None, message=None):
    # ... (Nội dung hàm plot_supermarket giữ nguyên) ...
    grid_map_to_plot = supermarket.grid_map
    num_rows, num_cols = grid_map_to_plot.shape
    
    plt.figure(figsize=(max(10, supermarket.width_m / 3.5), max(8, supermarket.height_m / 3.5))) 

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

    ap_present_on_map = False
    if AP_ID in unique_values and np.any(grid_map_to_plot == AP_ID):
        ap_present_on_map = True
        color_list.append('lightcoral')
    
    stall_ids_on_map = sorted([val for val in unique_values if STALL_ID_START <= val <= STALL_ID_END])
    if stall_ids_on_map:
        stall_cmap_len = len(stall_ids_on_map) if len(stall_ids_on_map) > 0 else 1
        stall_cmap = plt.cm.get_cmap('Pastel2', stall_cmap_len)
        for i, stall_id in enumerate(stall_ids_on_map):
            color_list.append(stall_cmap(i))
            stall_name = supermarket.stall_definitions.get(stall_id, {}).get("name", f"Gian hàng {stall_id}")
            legend_patches.append(Patch(facecolor=stall_cmap(i), edgecolor='black', label=f'{stall_name} ({stall_id})'))

    item_ids_on_map = sorted([val for val in unique_values if val >= ITEM_ID_START])
    if item_ids_on_map:
        item_cmap_len = len(item_ids_on_map) if len(item_ids_on_map) > 0 else 1
        item_cmap = plt.cm.get_cmap('tab20', item_cmap_len) 
        for i, item_id in enumerate(item_ids_on_map):
            color_list.append(item_cmap(i))
            item_name = supermarket.item_definitions.get(item_id, {}).get("name", f"Mặt hàng {item_id}")
            legend_patches.append(Patch(facecolor=item_cmap(i), edgecolor='black', label=f'{item_name} ({item_id})'))
    
    if unique_values.size > 0:
        boundaries = np.concatenate([[-0.5], np.array(sorted(unique_values)) + 0.5])
    else:
        boundaries = np.array([-0.5, 0.5])
    boundaries = np.unique(boundaries)

    norm = None
    if not color_list:
        cmap_display = ListedColormap(['lightgray'])
        if len(boundaries) < 2: boundaries = np.array([-0.5, 0.5])
        norm = BoundaryNorm(boundaries, cmap_display.N if cmap_display.N > 0 else 1)
        plt.imshow(grid_map_to_plot, cmap=cmap_display, norm=norm, origin='lower', interpolation='nearest')
    elif len(color_list) == (len(boundaries) - 1):
        cmap_display = ListedColormap(color_list)
        norm = BoundaryNorm(boundaries, cmap_display.N)
        plt.imshow(grid_map_to_plot, cmap=cmap_display, norm=norm, origin='lower', interpolation='nearest')
    else: 
        print(f"Plotting Warning: Mismatch color_list ({len(color_list)}) and boundaries-1 ({len(boundaries)-1}).")
        value_to_color_idx = {}
        actual_color_list_for_mapping = []
        idx_count = 0
        
        if PATHWAY_ID in unique_values: 
            value_to_color_idx[PATHWAY_ID] = idx_count; idx_count+=1
            actual_color_list_for_mapping.append('white')
        if OBSTACLE_ID in unique_values: 
            value_to_color_idx[OBSTACLE_ID] = idx_count; idx_count+=1
            actual_color_list_for_mapping.append('dimgray')
        if ap_present_on_map:
            value_to_color_idx[AP_ID] = idx_count; idx_count+=1
            actual_color_list_for_mapping.append('lightcoral')
        
        if stall_ids_on_map:
            stall_cmap_remap_len = len(stall_ids_on_map) if len(stall_ids_on_map) > 0 else 1
            stall_cmap_remap = plt.cm.get_cmap('Pastel2', stall_cmap_remap_len)
            for i, stall_id in enumerate(stall_ids_on_map):
                value_to_color_idx[stall_id] = idx_count; idx_count+=1
                actual_color_list_for_mapping.append(stall_cmap_remap(i))
        if item_ids_on_map:
            item_cmap_remap_len = len(item_ids_on_map) if len(item_ids_on_map) > 0 else 1
            item_cmap_remap = plt.cm.get_cmap('tab20', item_cmap_remap_len)
            for i, item_id in enumerate(item_ids_on_map):
                value_to_color_idx[item_id] = idx_count; idx_count+=1
                actual_color_list_for_mapping.append(item_cmap_remap(i))

        if actual_color_list_for_mapping:
            temp_grid = np.full_like(grid_map_to_plot, -1, dtype=float) # Initialize with -1
            for r_idx in range(num_rows):
                for c_idx in range(num_cols):
                    val = grid_map_to_plot[r_idx,c_idx]
                    if val in value_to_color_idx: # Check if value has a mapping
                        temp_grid[r_idx,c_idx] = value_to_color_idx[val]
            
            cmap_display = ListedColormap(actual_color_list_for_mapping)
            plt.imshow(temp_grid, cmap=cmap_display, origin='lower', interpolation='nearest', 
                       vmin=-0.5, vmax=len(actual_color_list_for_mapping)-0.5)
        else: 
            plt.imshow(grid_map_to_plot, cmap='nipy_spectral', origin='lower', interpolation='nearest')


    if supermarket.access_points:
        ap_r_plot, ap_c_plot = zip(*supermarket.access_points)
        plt.scatter(ap_c_plot, ap_r_plot, marker='o', color='red', edgecolor='black', s=100, label='AP (Điểm)', zorder=5)
        if not any(isinstance(h, plt.Line2D) and h.get_label() == 'AP (Điểm)' for h in legend_patches):
             legend_patches.append(plt.Line2D([0], [0], marker='o', color='w', label='AP (Điểm)', markerfacecolor='red', markeredgecolor='black', markersize=10))

    if cart_pos:
        plt.scatter(cart_pos[1], cart_pos[0], marker='s', color='blue', edgecolor='black', s=150, label='Xe đẩy', zorder=5)
        if not any(isinstance(h, plt.Line2D) and h.get_label() == 'Xe đẩy' for h in legend_patches):
            legend_patches.append(plt.Line2D([0], [0], marker='s', color='w', label='Xe đẩy', markerfacecolor='blue', markeredgecolor='black', markersize=10))
    
    if target_item_pos:
        item_name_at_target = supermarket.get_item_name_at_location(target_item_pos[0], target_item_pos[1])
        label_target = f'{item_name_at_target} (Mục tiêu)' if item_name_at_target else "Mục tiêu không rõ"
        plt.scatter(target_item_pos[1], target_item_pos[0], marker='*', color='magenta',edgecolor='black', s=250, label=label_target, zorder=5)
        if not any(isinstance(h, plt.Line2D) and h.get_label() == label_target for h in legend_patches):
             legend_patches.append(plt.Line2D([0], [0], marker='*', color='w', label=label_target, markerfacecolor='magenta',markeredgecolor='black', markersize=15))
    
    if path:
        if len(path) > 1 :
            path_rows, path_cols = zip(*path)
            plt.plot(path_cols, path_rows, color='cyan', linewidth=2, label='Đường đi', zorder=4)
            if not any(isinstance(h, plt.Line2D) and h.get_label() == 'Đường đi' for h in legend_patches):
                legend_patches.append(plt.Line2D([0], [0], color='cyan', lw=2, label='Đường đi'))
        elif len(path) == 1:
             plt.scatter(path[0][1], path[0][0], color='cyan', marker='x', s=50, label='Điểm bắt đầu đường đi', zorder=4)

    plt.xticks(np.arange(-0.5, num_cols, step=max(1, num_cols // 10)), [f"{i*supermarket.resolution_m:.0f}" for i in np.arange(0, num_cols + supermarket.resolution_m, step=max(1, num_cols // 10))])
    plt.yticks(np.arange(-0.5, num_rows, step=max(1, num_rows // 10)), [f"{i*supermarket.resolution_m:.0f}" for i in np.arange(0, num_rows + supermarket.resolution_m, step=max(1, num_rows // 10))])
    plt.xlabel("Chiều rộng (mét)")
    plt.ylabel("Chiều cao (mét)")
    
    title_text = "Bản đồ Siêu thị Tương tác bằng Giọng nói"
    if message:
        title_text += f"\n{message}"
    plt.title(title_text)

    plt.grid(True, which='major', color='lightgray', linestyle='-', linewidth=0.5)
    plt.xlim([-0.5, num_cols - 0.5])
    plt.ylim([num_rows - 0.5, -0.5])

    plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize='small')
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.show()


# --- HÀM NHẬN DIỆN GIỌNG NÓI (Giữ nguyên) ---
def recognize_speech_from_mic(recognizer, microphone):
    # ... (Nội dung hàm giữ nguyên) ...
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")
    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    with microphone as source:
        print("Điều chỉnh theo tiếng ồn xung quanh...")
        recognizer.adjust_for_ambient_noise(source, duration=1) 
        print("Ngưỡng năng lượng: {}".format(recognizer.energy_threshold))
        print("Hãy nói yêu cầu của bạn (ví dụ: 'dẫn tôi đến chỗ bán thịt bò', 'tìm gian hàng rau củ'...).")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10) # Tăng phrase_time_limit
        except sr.WaitTimeoutError:
            print("Không có âm thanh nào được phát hiện trong khoảng thời gian chờ.")
            return {"success": False, "error": "timeout", "transcription": None}

    print("Đã ghi nhận! Đang xử lý...")
    response = {"success": True, "error": None, "transcription": None}
    try:
        response["transcription"] = recognizer.recognize_google(audio, language="vi-VN")
    except sr.RequestError:
        response["success"] = False; response["error"] = "API unavailable"
    except sr.UnknownValueError:
        response["error"] = "Unable to recognize speech"
    return response

# --- HÀM TRÍCH XUẤT TỪ KHÓA MẶT HÀNG/GIAN HÀNG ---
def extract_keywords(text, supermarket_data: SupermarketMap):
    """
    Trích xuất từ khóa mặt hàng hoặc gian hàng từ văn bản.
    Trả về (loại_từ_khóa, tên_chuẩn_hóa, vị_trí_đầu_tiên_nếu_tìm_thấy)
    loại_từ_khóa: "item", "stall", hoặc None
    """
    if not text:
        return None, None, None

    text_lower = text.lower()
    
    # Ưu tiên tìm tên mặt hàng đầy đủ trước (khớp nhiều từ)
    # Sắp xếp tên mặt hàng theo độ dài giảm dần để khớp cụm dài trước
    sorted_item_names = sorted(supermarket_data.item_definitions.values(), key=lambda x: len(x['name']), reverse=True)
    for item_props in sorted_item_names:
        item_name_defined = item_props['name']
        # Tạo các biến thể của tên mặt hàng (ví dụ: bỏ dấu, chữ thường) để khớp
        item_name_lower = item_name_defined.lower()
        # (Tùy chọn nâng cao: bỏ dấu tiếng Việt cho item_name_lower để khớp tốt hơn)
        
        if item_name_lower in text_lower:
            locations = supermarket_data.get_item_locations_by_name(item_name_defined)
            if locations:
                print(f"Tìm thấy từ khóa mặt hàng (khớp nhiều từ): '{item_name_defined}'")
                return "item", item_name_defined, locations[0]

    # Tiếp theo, tìm tên gian hàng (khớp nhiều từ)
    sorted_stall_names = sorted(supermarket_data.stall_definitions.values(), key=lambda x: len(x['name']), reverse=True)
    for stall_id, stall_props in supermarket_data.stall_definitions.items(): # Lặp qua dict gốc để có stall_id
        stall_name_defined = stall_props['name']
        stall_name_lower = stall_name_defined.lower()
        
        if stall_name_lower in text_lower:
            # Tìm một mặt hàng bất kỳ trong gian hàng đó hoặc vị trí của gian hàng
            # Cách 1: Tìm vị trí đầu tiên của gian hàng trên bản đồ
            stall_coords = stall_props.get("coords") # (r, c, h, w)
            if stall_coords:
                # Lấy ô đầu tiên của gian hàng làm mục tiêu
                target_pos = (stall_coords[0], stall_coords[1]) 
                print(f"Tìm thấy từ khóa gian hàng (khớp nhiều từ): '{stall_name_defined}' tại {target_pos}")
                return "stall", stall_name_defined, target_pos
            # Cách 2: Tìm mặt hàng đầu tiên trong gian hàng đó (nếu muốn dẫn đến sản phẩm cụ thể)
            # for item_id_val, item_def_val in supermarket_data.item_definitions.items():
            #     item_locs = supermarket_data.get_item_locations_by_name(item_def_val["name"])
            #     if item_locs:
            #         r_item, c_item = item_locs[0]
            #         # Kiểm tra xem mặt hàng có thuộc gian hàng này không
            #         if stall_coords[0] <= r_item < stall_coords[0] + stall_coords[2] and \
            #            stall_coords[1] <= c_item < stall_coords[1] + stall_coords[3]:
            #             print(f"Tìm thấy từ khóa gian hàng '{stall_name_defined}', dẫn đến mặt hàng '{item_def_val['name']}'")
            #             return "item", item_def_val['name'], item_locs[0]


    # Nếu không khớp cụm dài, thử khớp từng từ (đơn giản hơn)
    words = re.findall(r'\b\w+\b', text_lower) # Tách thành các từ
    for word in words:
        # Kiểm tra với tên mặt hàng (đã chuẩn hóa thành chữ thường)
        for item_name_defined_props in supermarket_data.item_definitions.values():
            item_name_defined = item_name_defined_props['name']
            if word == item_name_defined.lower(): # So sánh từng từ
                locations = supermarket_data.get_item_locations_by_name(item_name_defined)
                if locations:
                    print(f"Tìm thấy từ khóa mặt hàng (khớp 1 từ): '{item_name_defined}'")
                    return "item", item_name_defined, locations[0]
        
        # Kiểm tra với tên gian hàng
        for stall_name_defined_props in supermarket_data.stall_definitions.values():
            stall_name_defined = stall_name_defined_props['name']
            if word == stall_name_defined.lower():
                stall_coords = stall_name_defined_props.get("coords")
                if stall_coords:
                    target_pos = (stall_coords[0], stall_coords[1])
                    print(f"Tìm thấy từ khóa gian hàng (khớp 1 từ): '{stall_name_defined}' tại {target_pos}")
                    return "stall", stall_name_defined, target_pos
    
    print(f"Không tìm thấy từ khóa nào phù hợp trong: '{text}'")
    return None, None, None


# --- HÀM TÌM ĐƯỜNG ĐI ĐƠN GIẢN (Giữ nguyên) ---
def find_simple_path(start_pos, target_pos, supermarket_map_obj: SupermarketMap):
    # ... (Nội dung hàm giữ nguyên) ...
    path = []
    curr_r, curr_c = start_pos
    target_r, target_c = target_pos
    path.append((curr_r, curr_c))

    while curr_c != target_c:
        prev_c = curr_c
        curr_c += 1 if target_c > curr_c else -1
        path.append((curr_r, curr_c))
        if prev_c == curr_c: break 
    
    while curr_r != target_r:
        prev_r = curr_r
        curr_r += 1 if target_r > curr_r else -1
        path.append((curr_r, curr_c))
        if prev_r == curr_r: break
    return path


if __name__ == "__main__":
    # --- KHỞI TẠO SIÊU THỊ (Giữ nguyên hoặc điều chỉnh nếu cần) ---
    supermarket = SupermarketMap(width_m=50, height_m=30, resolution_m=0.5)
    # ... (Toàn bộ phần add_general_obstacle, add_stall_area, add_item_to_grid, add_access_point giữ nguyên) ...
    # (Lấy từ code bạn đã cung cấp ở lần trước)
    supermarket.add_general_obstacle(r_start=0, c_start=0, obs_height=1, obs_width=supermarket.num_cols)
    supermarket.add_general_obstacle(r_start=supermarket.num_rows-1, c_start=0, obs_height=1, obs_width=supermarket.num_cols)
    supermarket.add_general_obstacle(r_start=0, c_start=0, obs_height=supermarket.num_rows, obs_width=1)
    supermarket.add_general_obstacle(r_start=0, c_start=supermarket.num_cols-1, obs_height=supermarket.num_rows, obs_width=1)

    stall1_r, stall1_c, stall1_h, stall1_w = 6, 2, 20, 4 
    id_fresh = supermarket.add_stall_area(stall1_r, stall1_c, stall1_h, stall1_w, "Trái Cây 1")
    if id_fresh != -1:
        supermarket.add_item_to_grid(stall1_r, stall1_c, 5, stall1_w, "Táo")
        supermarket.add_item_to_grid(stall1_r + 5, stall1_c, 5, stall1_w, "Chuối")
        supermarket.add_item_to_grid(stall1_r + 10, stall1_c, 5, stall1_w, "Bơ")
        supermarket.add_item_to_grid(stall1_r + 15, stall1_c, 5, stall1_w, "Dứa")

    stall3_r, stall3_c, stall3_h, stall3_w = 6, 14, 20, 6
    id_fresh_2 = supermarket.add_stall_area(stall3_r, stall3_c, stall3_h, stall3_w, "Trái Cây Tổng Hợp") # Đổi tên để tránh trùng
    if id_fresh_2 != -1:
        supermarket.add_item_to_grid(stall3_r, stall3_c, 5, stall3_w, "Nho")
        supermarket.add_item_to_grid(stall3_r + 5, stall3_c, 5, stall3_w, "Măng Cụt")
        supermarket.add_item_to_grid(stall3_r + 10, stall3_c, 5, stall3_w, "Thơm")
        supermarket.add_item_to_grid(stall3_r + 15, stall3_c, 5, stall3_w, "Xoài")

    stall4_r, stall4_c, stall4_h, stall4_w = 6, 28, 20, 6
    id_fresh_3 = supermarket.add_stall_area(stall4_r, stall4_c, stall4_h, stall4_w, "Rau Củ") # Đổi tên
    if id_fresh_3 != -1:
        supermarket.add_item_to_grid(stall4_r, stall4_c, 5, stall4_w, "Rau Muống")
        supermarket.add_item_to_grid(stall4_r + 5, stall4_c, 5, stall4_w, "Rau Dền")
        supermarket.add_item_to_grid(stall4_r + 10, stall4_c, 5, stall4_w, "Cải Xanh")
        supermarket.add_item_to_grid(stall4_r + 15, stall4_c, 5, stall4_w, "Mồng Tơi")


    stall2_r, stall2_c, stall2_h, stall2_w = 38, 15, 10, 4
    id_drinks = supermarket.add_stall_area(stall2_r, stall2_c, stall2_h, stall2_w, "Đồ Uống")
    if id_drinks != -1:
        supermarket.add_item_to_grid(stall2_r, stall2_c, stall2_h, stall2_w // 2, "Nước Ngọt")
        supermarket.add_item_to_grid(stall2_r, stall2_c + stall2_w // 2, stall2_h, stall2_w // 2, "Nước Suối")

    stall5_r, stall5_c, stall5_h, stall5_w = 2, 60, 4, 27 
    id_meats = supermarket.add_stall_area(stall5_r, stall5_c, stall5_h, stall5_w, "Thịt Tươi") # Đổi tên
    if id_meats != -1:
        supermarket.add_item_to_grid(stall5_r, stall5_c, stall5_h, 9, "Thịt Gà")
        supermarket.add_item_to_grid(stall5_r, stall5_c + 9, stall5_h, 9, "Thịt Heo")
        supermarket.add_item_to_grid(stall5_r, stall5_c + 18, stall5_h, 9, "Thịt Bò")


    stall6_r, stall6_c, stall6_h, stall6_w = 35, 65, 16, 16
    id_cakes = supermarket.add_stall_area(stall6_r, stall6_c, stall6_h, stall6_w, "Bánh Ngọt") # Đổi tên
    if id_cakes != -1:
        supermarket.add_item_to_grid(stall6_r, stall6_c, stall6_h//2, stall6_w//2, "Bánh Quy")
        supermarket.add_item_to_grid(stall6_r + stall6_h//2, stall6_c, stall6_h//2, stall6_w//2, "Bánh Bao") # Sửa vị trí
        supermarket.add_item_to_grid(stall6_r, stall6_c + stall6_w//2, stall6_h//2, stall6_w//2, "Bánh Mì") # Sửa vị trí
        supermarket.add_item_to_grid(stall6_r + stall6_h//2, stall6_c + stall6_w//2, stall6_h//2, stall6_w//2, "Bánh Xếp") # Sửa vị trí
    
    ap_locations = [
        (3, 3), (3, supermarket.num_cols - 4), 
        (supermarket.num_rows - 4, 3), (supermarket.num_rows - 4, supermarket.num_cols - 4),
        (supermarket.num_rows // 2, supermarket.num_cols // 2)
    ]
    for r, c in ap_locations:
        supermarket.add_access_point(r, c)
    # --- Kết thúc phần khởi tạo siêu thị ---

    cart_current_pos = (supermarket.num_rows - 5, 5) 
    plot_supermarket(supermarket, cart_pos=cart_current_pos, message="Siêu thị sẵn sàng. Nhấn Ctrl+C ở terminal để thoát.")

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    try: import pyaudio
    except ImportError: print("Lỗi: PyAudio chưa được cài đặt..."); exit()

    while True:
        print("\n------------------------------------------")
        speech_response = recognize_speech_from_mic(recognizer, microphone)
        
        plot_message = None
        target_pos_on_grid = None # Sẽ là (r,c)
        target_name_display = None # Tên để hiển thị
        path_to_target = None

        if speech_response["transcription"]:
            print(f"Bạn đã nói: \"{speech_response['transcription']}\"")
            spoken_text = speech_response["transcription"]
            
            keyword_type, found_name, found_pos = extract_keywords(spoken_text, supermarket)

            if found_name and found_pos:
                target_pos_on_grid = found_pos
                target_name_display = found_name
                print(f"Đang tìm đường đến '{target_name_display}' ({keyword_type}) tại {target_pos_on_grid}...")
                path_to_target = find_simple_path(cart_current_pos, target_pos_on_grid, supermarket)
                plot_message = f"Đang dẫn đường đến: {target_name_display}"
            else:
                print(f"Không thể xác định mặt hàng/gian hàng từ: \"{spoken_text}\"")
                plot_message = f"Không hiểu rõ: {spoken_text}"
        
        elif speech_response["error"]:
            print(f"Lỗi nhận diện: {speech_response['error']}")
            plot_message = f"Lỗi: {speech_response['error']}"
        
        plt.close('all') 
        plot_supermarket(supermarket, 
                         cart_pos=cart_current_pos, 
                         path=path_to_target, 
                         target_item_pos=target_pos_on_grid, # Bây giờ target_item_pos có thể là gian hàng
                         message=plot_message)