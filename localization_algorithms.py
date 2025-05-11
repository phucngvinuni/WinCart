# localization_algorithms.py
import math
import config
from supermarket_model import PATHWAY_ID # Import ID cần thiết

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid as PathfindingGrid # Đổi tên để rõ ràng
from pathfinding.finder.a_star import AStarFinder

def rssi_distance_euclidean(rssi_vec1, rssi_vec2):
    """Tính khoảng cách Euclide giữa hai vector RSSI."""
    if len(rssi_vec1) != len(rssi_vec2):
        raise ValueError("Các vector RSSI phải có cùng độ dài")
    squared_diff_sum = sum([(v1 - v2)**2 for v1, v2 in zip(rssi_vec1, rssi_vec2)])
    return math.sqrt(squared_diff_sum)

def predict_location_knn(observed_rssi, fingerprints_data, k, weighted=False, epsilon=1e-6):
    """Dự đoán vị trí dựa trên KNN."""
    if not fingerprints_data:
        print("Lỗi KNN: Dữ liệu fingerprint trống.")
        return None

    distances_to_fingerprints = []
    for (r_fp, c_fp), rssi_fp_values in fingerprints_data.items():
        dist = rssi_distance_euclidean(observed_rssi, rssi_fp_values)
        distances_to_fingerprints.append(((r_fp, c_fp), dist))

    if not distances_to_fingerprints: # Không có điểm nào trong fingerprint map (rất lạ)
        print("Lỗi KNN: Không có điểm nào trong fingerprint map.")
        return None

    distances_to_fingerprints.sort(key=lambda item: item[1])
    
    # Đảm bảo k không lớn hơn số lượng fingerprints có sẵn
    actual_k = min(k, len(distances_to_fingerprints))
    if actual_k == 0:
        print("Lỗi KNN: Không có láng giềng nào để chọn (actual_k = 0).")
        return None
    k_nearest = distances_to_fingerprints[:actual_k]


    if not weighted:
        sum_r, sum_c = 0, 0
        for (r_n, c_n), _ in k_nearest:
            sum_r += r_n
            sum_c += c_n
        estimated_r = sum_r / actual_k
        estimated_c = sum_c / actual_k
    else:
        weighted_sum_r, weighted_sum_c, sum_weights = 0, 0, 0
        for (r_n, c_n), dist_rssi in k_nearest:
            weight = 1 / (dist_rssi + epsilon)
            weighted_sum_r += r_n * weight
            weighted_sum_c += c_n * weight
            sum_weights += weight
        if sum_weights == 0:
            sum_r, sum_c = 0, 0
            for (r_n, c_n), _ in k_nearest: # Fallback to non-weighted if all weights are zero
                sum_r += r_n
                sum_c += c_n
            estimated_r = sum_r / actual_k
            estimated_c = sum_c / actual_k
            print("Cảnh báo KNN: Tổng trọng số bằng 0, sử dụng KNN không trọng số.")
        else:
            estimated_r = weighted_sum_r / sum_weights
            estimated_c = weighted_sum_c / sum_weights
    return (estimated_r, estimated_c)

def find_path_astar(supermarket_map_obj, start_node_rc, end_node_rc):
    """
    Tìm đường đi ngắn nhất bằng thuật toán A*.
    supermarket_map_obj: instance của SupermarketMap.
    start_node_rc: (hàng, cột) của điểm bắt đầu.
    end_node_rc: (hàng, cột) của điểm kết thúc.
    """
    grid_data_for_pathfinding = supermarket_map_obj.grid_map
    matrix = []
    for r_idx in range(grid_data_for_pathfinding.shape[0]):
        row_data = []
        for c_idx in range(grid_data_for_pathfinding.shape[1]):
            # Thư viện pathfinding: >0 là đi được (trọng số), <=0 là vật cản
            if grid_data_for_pathfinding[r_idx, c_idx] == PATHWAY_ID or \
               grid_data_for_pathfinding[r_idx, c_idx] == AP_ID: # Coi ô AP cũng là lối đi
                row_data.append(1)
            else: # Tất cả các ID khác (OBSTACLE, STALL, ITEM) là vật cản
                row_data.append(0)
        matrix.append(row_data)

    try:
        path_grid = PathfindingGrid(matrix=matrix) # Sử dụng tên đã import

        start_pf_node = path_grid.node(start_node_rc[1], start_node_rc[0]) # (col, row)
        end_pf_node = path_grid.node(end_node_rc[1], end_node_rc[0])     # (col, row)

        if not start_pf_node.walkable:
            print(f"Lỗi tìm đường: Điểm bắt đầu ({start_node_rc}) là vật cản trong pathfinding matrix.")
            return None
        if not end_pf_node.walkable:
            print(f"Lỗi tìm đường: Điểm kết thúc ({end_node_rc}) là vật cản trong pathfinding matrix.")
            return None

        finder = AStarFinder(diagonal_movement=DiagonalMovement.never) # Chỉ đi ngang/dọc cho đơn giản
        path, runs = finder.find_path(start_pf_node, end_pf_node, path_grid)

        if path:
            return [(node.y, node.x) for node in path] # Chuyển lại (hàng, cột)
        else:
            print(f"A* không tìm thấy đường từ {start_node_rc} đến {end_node_rc}.")
            return None
    except IndexError as e:
        print(f"Lỗi IndexError khi tìm đường (có thể điểm ra ngoài biên): {e}")
        print(f"  Start: {start_node_rc}, End: {end_node_rc}, Map dims: {supermarket_map_obj.num_rows}x{supermarket_map_obj.num_cols}")
        return None
    except Exception as e:
        print(f"Lỗi không xác định khi tìm đường: {e}")
        return None