import math
import numpy as np
import config # Import các hằng số từ config.py
from supermarket_model import PATHWAY_ID, AP_ID # Import ID cần thiết

def euclidean_distance_m(p1_grid_rc, p2_grid_rc): # Đổi tên tham số cho rõ ràng (row, col)
    """Tính khoảng cách Euclide giữa hai điểm trên lưới (tính bằng mét)."""
    dist_cells = math.sqrt((p1_grid_rc[0] - p2_grid_rc[0])**2 + (p1_grid_rc[1] - p2_grid_rc[1])**2)
    return dist_cells * config.GRID_RESOLUTION_M

def get_line_cells_rc(r1, c1, r2, c2): # Đổi tên và tham số để rõ ràng (row, col)
    """Sử dụng thuật toán Bresenham. Trả về list of (row, col)."""
    points = []
    # Bresenham thường dùng (x,y), chúng ta sẽ map (c,r) -> (x,y) nội bộ
    x_start, y_start = c1, r1
    x_end, y_end = c2, r2

    dx = abs(x_end - x_start)
    dy = abs(y_end - y_start)
    sx = 1 if x_start < x_end else -1
    sy = 1 if y_start < y_end else -1
    err = dx - dy

    current_x, current_y = x_start, y_start
    while True:
        points.append((current_y, current_x)) # Lưu trữ lại là (row, col)
        if current_x == x_end and current_y == y_end:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            current_x += sx
        if e2 < dx:
            err += dy
            current_y += sy
    return points

def count_obstacle_intersections_on_map(supermarket_map_obj, ap_pos_rc, cell_pos_rc):
    """
    Đếm số lượng ô vật cản (không phải lối đi và không phải ô AP)
    mà đường thẳng từ AP đến cell_pos_rc đi qua.
    supermarket_map_obj: instance của SupermarketMap.
    ap_pos_rc: (hàng_ap, cột_ap)
    cell_pos_rc: (hàng_ô, cột_ô)
    """
    line_cells = get_line_cells_rc(ap_pos_rc[0], ap_pos_rc[1], cell_pos_rc[0], cell_pos_rc[1])
    obstacle_crossings = 0
    grid = supermarket_map_obj.grid_map # Truy cập grid_map từ đối tượng
    num_r, num_c = grid.shape

    # Bỏ qua điểm bắt đầu (AP) và điểm kết thúc (ô đang xét)
    for r, c in line_cells[1:-1]:
        if 0 <= r < num_r and 0 <= c < num_c:
            cell_id = grid[r, c]
            # Coi tất cả những gì không phải PATHWAY_ID và không phải AP_ID là vật cản tín hiệu
            if cell_id != PATHWAY_ID and cell_id != AP_ID:
                obstacle_crossings += 1
    return obstacle_crossings

def calculate_single_rssi_on_map(supermarket_map_obj, ap_pos_rc, cell_pos_rc):
    """
    Tính toán RSSI mô phỏng tại cell_pos_rc từ một AP cụ thể trên supermarket_map_obj.
    """
    distance_m = euclidean_distance_m(ap_pos_rc, cell_pos_rc)

    if distance_m < config.GRID_RESOLUTION_M / 2: # Ở rất gần hoặc trùng AP
        return config.P_TX_MAX_RSSI + np.random.normal(0, config.NOISE_STD_DEV_DB / 3)

    path_loss_db = 10 * config.PATH_LOSS_EXPONENT_N * math.log10(distance_m)
    
    num_obstacles = count_obstacle_intersections_on_map(supermarket_map_obj, ap_pos_rc, cell_pos_rc)
    total_obstacle_attenuation_db = num_obstacles * config.SHELF_ATTENUATION_DB # Sử dụng chung suy hao

    noise_db = np.random.normal(0, config.NOISE_STD_DEV_DB)
    
    rssi = config.P_TX_MAX_RSSI - path_loss_db - total_obstacle_attenuation_db + noise_db
    return max(rssi, config.MIN_RSSI_THRESHOLD)

def generate_rssi_fingerprints_from_map(supermarket_map_obj):
    grid = supermarket_map_obj.grid_map
    num_rows, num_cols = supermarket_map_obj.num_rows, supermarket_map_obj.num_cols
    access_points = supermarket_map_obj.access_points
    num_aps = len(access_points)

    # Khởi tạo với một giá trị đặc biệt (ví dụ NaN hoặc giá trị không thể có của RSSI)
    # để biết ô nào không có fingerprint
    fingerprints_array = np.full((num_rows, num_cols, num_aps), np.nan, dtype=np.float32)

    for r_idx in range(num_rows):
        for c_idx in range(num_cols):
            if grid[r_idx, c_idx] == PATHWAY_ID or grid[r_idx, c_idx] == AP_ID:
                for ap_idx, ap_pos in enumerate(access_points):
                    rssi_val = calculate_single_rssi_on_map(supermarket_map_obj, ap_pos, (r_idx, c_idx))
                    fingerprints_array[r_idx, c_idx, ap_idx] = rssi_val
    return fingerprints_array # Trả về mảng NumPy

def get_observed_rssi_at_cart_on_map(supermarket_map_obj, cart_pos_rc):
    """
    Tính toán RSSI 'quan sát được' tại vị trí xe đẩy trên supermarket_map_obj.
    """
    observed_rssi = []
    access_points = supermarket_map_obj.access_points
    for ap_pos in access_points:
        rssi_val = calculate_single_rssi_on_map(supermarket_map_obj, ap_pos, cart_pos_rc)
        observed_rssi.append(rssi_val)
    return observed_rssi