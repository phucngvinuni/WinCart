import numpy as np
import matplotlib.pyplot as plt
import math

# --- Các tham số đã định nghĩa ở phần bản đồ ---
# Kích thước siêu thị (mét)
supermarket_width = 50
supermarket_height = 30
# Độ phân giải lưới (mét/ô)
grid_resolution = 0.5 # Giữ nguyên hoặc điều chỉnh nếu muốn

num_cols = int(supermarket_width / grid_resolution)
num_rows = int(supermarket_height / grid_resolution)

# 0: Lối đi
# 1: Kệ hàng/Vật cản
grid_map = np.zeros((num_rows, num_cols), dtype=int)

# --- Vẽ kệ hàng (VÍ DỤ - bạn có thể thay đổi hoặc thêm kệ) ---
# Kệ 1
shelf1_row_start, shelf1_col_start = num_rows // 4, num_cols // 4
shelf1_rows, shelf1_cols_count = num_rows // 2, 2 # Kệ dọc
grid_map[shelf1_row_start : shelf1_row_start + shelf1_rows,
         shelf1_col_start : shelf1_col_start + shelf1_cols_count] = 1

# Kệ 2
shelf2_row_start, shelf2_col_start = num_rows // 4, (num_cols // 4) * 3 - 2 # Trừ đi bề rộng kệ
shelf2_rows, shelf2_cols_count = num_rows // 2, 2 # Kệ dọc
grid_map[shelf2_row_start : shelf2_row_start + shelf2_rows,
         shelf2_col_start : shelf2_col_start + shelf2_cols_count] = 1

# --- Đặt 4 AP ở 4 góc ---
# Chúng ta sẽ đặt chúng cách mép một chút để không nằm hoàn toàn ở biên
margin = 2 # Số ô cách mép

access_points = [
    (margin, margin),                               # Góc trên bên trái
    (margin, num_cols - 1 - margin),                # Góc trên bên phải
    (num_rows - 1 - margin, margin),                # Góc dưới bên trái
    (num_rows - 1 - margin, num_cols - 1 - margin)  # Góc dưới bên phải
]

# --- Tham số mô phỏng RSSI (GIỮ NGUYÊN HOẶC ĐIỀU CHỈNH) ---
P_TX_MAX_RSSI = -30
PATH_LOSS_EXPONENT_N = 2.8
SHELF_ATTENUATION_DB = 4.0
NOISE_STD_DEV_DB = 3.0
MIN_RSSI_THRESHOLD = -95

# --- Các hàm tiện ích (euclidean_distance, get_line_cells, count_shelf_intersections, calculate_rssi) ---
# GIỮ NGUYÊN CÁC HÀM NÀY TỪ ĐOẠN MÃ TRƯỚC

def euclidean_distance(p1_grid, p2_grid, resolution):
    dist_pixels = math.sqrt((p1_grid[0] - p2_grid[0])**2 + (p1_grid[1] - p2_grid[1])**2)
    return dist_pixels * resolution

def get_line_cells(x1, y1, x2, y2):
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    while True:
        points.append((y1, x1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return points

def count_shelf_intersections(ap_pos_grid, cell_pos_grid, current_grid_map):
    line_cells = get_line_cells(ap_pos_grid[1], ap_pos_grid[0], cell_pos_grid[1], cell_pos_grid[0])
    shelf_crossings = 0
    for r, c in line_cells[1:-1]:
        if 0 <= r < current_grid_map.shape[0] and 0 <= c < current_grid_map.shape[1]:
            if current_grid_map[r, c] == 1:
                shelf_crossings += 1
    return shelf_crossings

def calculate_rssi(ap_pos_grid, cell_pos_grid, current_grid_map, resolution):
    distance_m = euclidean_distance(ap_pos_grid, cell_pos_grid, resolution)
    if distance_m < resolution / 2: # Xử lý trường hợp ở rất gần hoặc trùng AP
        # Trả về RSSI tối đa với một chút nhiễu nhỏ hơn
        return P_TX_MAX_RSSI + np.random.normal(0, NOISE_STD_DEV_DB / 3)

    path_loss_db = 10 * PATH_LOSS_EXPONENT_N * math.log10(distance_m)
    num_shelves = count_shelf_intersections(ap_pos_grid, cell_pos_grid, current_grid_map)
    total_shelf_attenuation_db = num_shelves * SHELF_ATTENUATION_DB
    noise_db = np.random.normal(0, NOISE_STD_DEV_DB)
    rssi = P_TX_MAX_RSSI - path_loss_db - total_shelf_attenuation_db + noise_db
    return max(rssi, MIN_RSSI_THRESHOLD)


# --- Tạo bản đồ RSSI (Fingerprint Map) ---
rssi_fingerprints = {}
for r_idx in range(num_rows):
    for c_idx in range(num_cols):
        if grid_map[r_idx, c_idx] != 1:
            current_cell_rssi_values = []
            for ap_id, ap_pos in enumerate(access_points):
                rssi_val = calculate_rssi(ap_pos, (r_idx, c_idx), grid_map, grid_resolution)
                current_cell_rssi_values.append(rssi_val)
            rssi_fingerprints[(r_idx, c_idx)] = current_cell_rssi_values

# --- Trực quan hóa RSSI từ một AP cụ thể ---
# GIỮ NGUYÊN HÀM plot_rssi_heatmap_for_ap TỪ ĐOẠN MÃ TRƯỚC

def plot_rssi_heatmap_for_ap(ap_index, fingerprints_data, base_grid_map, aps_list, res):
    rssi_map_for_ap = np.full_like(base_grid_map, np.nan, dtype=float)
    ap_to_plot_pos = aps_list[ap_index]

    for (r,c), rssi_list in fingerprints_data.items():
        if base_grid_map[r,c] != 1:
            rssi_map_for_ap[r,c] = rssi_list[ap_index]

    plt.figure(figsize=(base_grid_map.shape[1] / (10 / res) / 1.5 , base_grid_map.shape[0] / (10 / res) / 1.5 )) # Điều chỉnh kích thước plot
    plt.imshow(base_grid_map, cmap='Greys', origin='lower', alpha=0.3, interpolation='nearest')
    plt.imshow(rssi_map_for_ap, cmap='jet', origin='lower', alpha=0.7, interpolation='bilinear', vmin=MIN_RSSI_THRESHOLD, vmax=P_TX_MAX_RSSI)

    for i, (r_ap, c_ap) in enumerate(aps_list):
        marker = 'X' if i == ap_index else 'o'
        color = 'black' if i == ap_index else 'red'
        size = 150 if i == ap_index else 80
        is_first_other_ap = True
        if i == ap_index :
             plt.scatter(c_ap, r_ap, marker=marker, color=color, s=size, label=f'AP {i} (Đang hiển thị)')
        elif is_first_other_ap and i != ap_index :
             plt.scatter(c_ap, r_ap, marker=marker, color=color, s=size, label=f'Các AP khác')
             is_first_other_ap = False
        else:
             plt.scatter(c_ap, r_ap, marker=marker, color=color, s=size)


    plt.colorbar(label='RSSI (dBm)')
    plt.title(f"Mô phỏng RSSI Heatmap cho AP {ap_index} tại ({ap_to_plot_pos[0]*grid_resolution:.1f}m, {ap_to_plot_pos[1]*grid_resolution:.1f}m)")
    plt.xlabel(f"Chiều rộng (ô lưới - 1 ô = {grid_resolution}m)")
    plt.ylabel(f"Chiều cao (ô lưới - 1 ô = {grid_resolution}m)")
    # Điều chỉnh tick cho phù hợp hơn
    x_ticks = np.arange(0, num_cols, step=max(1, int(num_cols / 10)))
    y_ticks = np.arange(0, num_rows, step=max(1, int(num_rows / 10)))
    plt.xticks(x_ticks, [f"{i*grid_resolution:.1f}" for i in x_ticks])
    plt.yticks(y_ticks, [f"{i*grid_resolution:.1f}" for i in y_ticks])
    plt.gca().invert_yaxis()
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1)) # Điều chỉnh vị trí legend
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Điều chỉnh layout để legend không bị che
    plt.show()

# --- Vẽ bản đồ cơ sở và các heatmap ---

# 1. Vẽ bản đồ cơ sở với vị trí APs
def plot_base_map_with_aps(current_grid_map, aps_list, res):
    plt.figure(figsize=(current_grid_map.shape[1] / (10 / res) / 1.5 , current_grid_map.shape[0] / (10 / res) / 1.5 ))
    plt.imshow(current_grid_map, cmap='viridis', origin='lower', interpolation='nearest') # Sử dụng viridis cho map cơ sở
    for i, (r_ap, c_ap) in enumerate(aps_list):
        plt.scatter(c_ap, r_ap, marker='o', color='red', s=100, label=f'AP {i}' if i==0 else "")
    plt.title("Bản đồ Siêu thị Cơ sở với Vị trí APs")
    plt.xlabel(f"Chiều rộng (ô lưới - 1 ô = {grid_resolution}m)")
    plt.ylabel(f"Chiều cao (ô lưới - 1 ô = {grid_resolution}m)")
    x_ticks = np.arange(0, num_cols, step=max(1, int(num_cols / 10)))
    y_ticks = np.arange(0, num_rows, step=max(1, int(num_rows / 10)))
    plt.xticks(x_ticks, [f"{i*grid_resolution:.1f}" for i in x_ticks])
    plt.yticks(y_ticks, [f"{i*grid_resolution:.1f}" for i in y_ticks])
    plt.gca().invert_yaxis()
    if aps_list: # Chỉ hiển thị legend nếu có AP
        plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.show()

plot_base_map_with_aps(grid_map.copy(), access_points, grid_resolution)


# 2. Hiển thị heatmap RSSI cho từng AP
if access_points:
    for i in range(len(access_points)):
        plot_rssi_heatmap_for_ap(i, rssi_fingerprints, grid_map.copy(), access_points, grid_resolution)

# In ra một vài giá trị fingerprint để kiểm tra
print("\nMột vài giá trị RSSI fingerprint (vị trí thực tế mét): [RSSI_AP0, RSSI_AP1, RSSI_AP2, RSSI_AP3]")
count = 0
for (r, c), rssi_vals in rssi_fingerprints.items():
    if count < 10 and (r % (num_rows//5) == 0 and c % (num_cols//5) == 0) : # In thưa hơn
        print(f"({r*grid_resolution:.1f}m, {c*grid_resolution:.1f}m): {[round(val, 1) for val in rssi_vals]}")
        count += 1
    if count >=10:
        break



    # --- Tham số cho KNN ---
K_NEIGHBORS = 3 # Số láng giềng gần nhất để xem xét
USE_WEIGHTED_KNN = True # Sử dụng KNN có trọng số hay không
EPSILON_WEIGHT = 1e-6 # Giá trị nhỏ để tránh chia cho 0 trong weighted KNN

# --- Hàm cho KNN ---
def rssi_distance_euclidean(rssi_vec1, rssi_vec2):
    """Tính khoảng cách Euclide giữa hai vector RSSI."""
    if len(rssi_vec1) != len(rssi_vec2):
        raise ValueError("Các vector RSSI phải có cùng độ dài")
    squared_diff_sum = sum([(v1 - v2)**2 for v1, v2 in zip(rssi_vec1, rssi_vec2)])
    return math.sqrt(squared_diff_sum)

def predict_location_knn(observed_rssi, fingerprints_data, k, weighted=False, epsilon=1e-6):
    """
    Dự đoán vị trí dựa trên KNN.
    observed_rssi: Danh sách các giá trị RSSI quan sát được từ xe đẩy.
    fingerprints_data: Dictionary (row, col) -> [RSSI_AP1, RSSI_AP2, ...]
    k: Số láng giềng.
    weighted: True nếu sử dụng KNN có trọng số.
    epsilon: Giá trị nhỏ cho trọng số.
    """
    if not fingerprints_data:
        print("Lỗi: Dữ liệu fingerprint trống.")
        return None

    distances_to_fingerprints = []
    for (r_fp, c_fp), rssi_fp_values in fingerprints_data.items():
        dist = rssi_distance_euclidean(observed_rssi, rssi_fp_values)
        distances_to_fingerprints.append(((r_fp, c_fp), dist))

    # Sắp xếp theo khoảng cách RSSI
    distances_to_fingerprints.sort(key=lambda item: item[1])

    # Lấy K láng giềng gần nhất
    k_nearest = distances_to_fingerprints[:k]

    if not k_nearest:
        print("Lỗi: Không tìm thấy láng giềng nào.")
        return None

    # Ước tính vị trí
    if not weighted:
        # KNN thông thường (trung bình cộng tọa độ)
        sum_r, sum_c = 0, 0
        for (r_n, c_n), _ in k_nearest:
            sum_r += r_n
            sum_c += c_n
        estimated_r = sum_r / k
        estimated_c = sum_c / k
    else:
        # KNN có trọng số
        weighted_sum_r, weighted_sum_c, sum_weights = 0, 0, 0
        for (r_n, c_n), dist_rssi in k_nearest:
            weight = 1 / (dist_rssi + epsilon)
            weighted_sum_r += r_n * weight
            weighted_sum_c += c_n * weight
            sum_weights += weight
        if sum_weights == 0: # Trường hợp hiếm gặp nếu tất cả khoảng cách rất lớn
            # Quay lại KNN thông thường cho các láng giềng này
            sum_r, sum_c = 0, 0
            for (r_n, c_n), _ in k_nearest:
                sum_r += r_n
                sum_c += c_n
            estimated_r = sum_r / k
            estimated_c = sum_c / k
            print("Cảnh báo: Tổng trọng số bằng 0, sử dụng KNN không trọng số.")
        else:
            estimated_r = weighted_sum_r / sum_weights
            estimated_c = weighted_sum_c / sum_weights

    return (estimated_r, estimated_c) # Trả về tọa độ có thể là số thực

# --- Mô phỏng vị trí xe đẩy và định vị ---
# Chọn một vị trí thực tế cho xe đẩy (phải là ô lối đi)
cart_actual_pos_grid = (num_rows // 2 + 5, num_cols // 2 - 10) # Ví dụ
if grid_map[cart_actual_pos_grid[0], cart_actual_pos_grid[1]] == 1:
    print(f"Lỗi: Vị trí xe đẩy {cart_actual_pos_grid} nằm trên kệ hàng. Vui lòng chọn vị trí khác.")
    # Có thể chọn một vị trí ngẫu nhiên là lối đi ở đây
    walkable_cells = np.argwhere(grid_map != 1)
    if len(walkable_cells) > 0:
        cart_actual_pos_grid = tuple(walkable_cells[np.random.choice(len(walkable_cells))])
        print(f"Đã chọn vị trí xe đẩy ngẫu nhiên mới: {cart_actual_pos_grid}")
    else:
        print("Không có ô lối đi nào trên bản đồ!")
        exit()


# 1. Lấy RSSI "quan sát được" tại vị trí thực của xe đẩy
#   Để đơn giản, chúng ta lấy từ rssi_fingerprints đã tạo (có nhiễu sẵn)
#   Trong thực tế, bạn có thể gọi calculate_rssi với một seed nhiễu khác.
if cart_actual_pos_grid in rssi_fingerprints:
    cart_observed_rssi = rssi_fingerprints[cart_actual_pos_grid]
else:
    # Nếu vị trí xe đẩy không có trong fingerprint (ví dụ: độ phân giải khác)
    # thì cần tính toán lại RSSI tại đó
    print(f"Cảnh báo: Vị trí xe đẩy {cart_actual_pos_grid} không có trong fingerprints. Tính toán lại RSSI...")
    cart_observed_rssi = []
    for ap_id, ap_pos in enumerate(access_points):
        rssi_val = calculate_rssi(ap_pos, cart_actual_pos_grid, grid_map, grid_resolution)
        cart_observed_rssi.append(rssi_val)

print(f"\nXe đẩy ở vị trí thực tế (ô lưới): {cart_actual_pos_grid} ({cart_actual_pos_grid[0]*grid_resolution:.1f}m, {cart_actual_pos_grid[1]*grid_resolution:.1f}m)")
print(f"RSSI quan sát được của xe đẩy: {[round(val, 1) for val in cart_observed_rssi]}")

# 2. Dự đoán vị trí bằng KNN
cart_estimated_pos_grid_float = predict_location_knn(cart_observed_rssi, rssi_fingerprints, K_NEIGHBORS, USE_WEIGHTED_KNN, EPSILON_WEIGHT)

if cart_estimated_pos_grid_float:
    # Làm tròn về ô lưới gần nhất để hiển thị (tùy chọn)
    cart_estimated_pos_grid_int = (round(cart_estimated_pos_grid_float[0]), round(cart_estimated_pos_grid_float[1]))

    print(f"Vị trí xe đẩy ước tính (ô lưới, float): ({cart_estimated_pos_grid_float[0]:.2f}, {cart_estimated_pos_grid_float[1]:.2f})")
    print(f"Vị trí xe đẩy ước tính (ô lưới, int): {cart_estimated_pos_grid_int}")
    print(f"   ({cart_estimated_pos_grid_float[0]*grid_resolution:.1f}m, {cart_estimated_pos_grid_float[1]*grid_resolution:.1f}m)")


    # 3. Tính sai số định vị
    error_pixels = euclidean_distance(cart_actual_pos_grid, cart_estimated_pos_grid_float, 1) # Sai số theo pixel/ô
    error_meters = euclidean_distance(cart_actual_pos_grid, cart_estimated_pos_grid_float, grid_resolution) # Sai số theo mét
    print(f"Sai số định vị: {error_pixels:.2f} ô lưới, tương đương {error_meters:.2f} mét")

    # 4. Trực quan hóa (Thêm vào hàm plot_map hoặc tạo hàm mới)
    def plot_map_with_localization(current_grid_map, aps_list, actual_pos, estimated_pos_float, res, k_val, weighted_knn):
        plt.figure(figsize=(current_grid_map.shape[1] / (10 / res) / 1.5 , current_grid_map.shape[0] / (10 / res) / 1.5 ))
        plt.imshow(current_grid_map, cmap='Greys', origin='lower', alpha=0.6, interpolation='nearest')

        # Đánh dấu APs
        for i, (r_ap, c_ap) in enumerate(aps_list):
            plt.scatter(c_ap, r_ap, marker='o', color='red', s=100, label='AP' if i==0 else "")

        # Vị trí thực
        plt.scatter(actual_pos[1], actual_pos[0], marker='X', color='blue', s=200, label='Vị trí thực xe đẩy')
        # Vị trí ước tính
        plt.scatter(estimated_pos_float[1], estimated_pos_float[0], marker='P', color='green', s=200, label=f'Vị trí KNN (K={k_val}, {"Weighted" if weighted_knn else "Normal"})')

        # Nối vị trí thực và ước tính
        plt.plot([actual_pos[1], estimated_pos_float[1]], [actual_pos[0], estimated_pos_float[0]], 'm--', linewidth=1.5, label=f'Sai số: {error_meters:.2f}m')


        plt.title(f"Định vị Xe đẩy (K={k_val}, {'Weighted KNN' if weighted_knn else 'KNN'})")
        plt.xlabel(f"Chiều rộng (ô lưới - 1 ô = {grid_resolution}m)")
        plt.ylabel(f"Chiều cao (ô lưới - 1 ô = {grid_resolution}m)")
        x_ticks = np.arange(0, num_cols, step=max(1, int(num_cols / 10)))
        y_ticks = np.arange(0, num_rows, step=max(1, int(num_rows / 10)))
        plt.xticks(x_ticks, [f"{i*grid_resolution:.1f}" for i in x_ticks])
        plt.yticks(y_ticks, [f"{i*grid_resolution:.1f}" for i in y_ticks])
        plt.gca().invert_yaxis()
        plt.legend(loc='upper right', bbox_to_anchor=(1.45, 1))
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        plt.show()

    plot_map_with_localization(grid_map.copy(), access_points, cart_actual_pos_grid, cart_estimated_pos_grid_float, grid_resolution, K_NEIGHBORS, USE_WEIGHTED_KNN)