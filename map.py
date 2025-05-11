import numpy as np
import matplotlib.pyplot as plt

# Kích thước siêu thị (mét)
supermarket_width = 50
supermarket_height = 30
# Độ phân giải lưới (mét/ô)
grid_resolution = 0.5

# Số lượng ô theo chiều rộng và chiều cao
num_cols = int(supermarket_width / grid_resolution)
num_rows = int(supermarket_height / grid_resolution)

# Tạo bản đồ lưới, khởi tạo tất cả là lối đi (ví dụ: giá trị 0)
# 0: Lối đi
# 1: Kệ hàng/Vật cản
# 2: AP
# 3+: Mã loại mặt hàng
grid_map = np.zeros((num_rows, num_cols), dtype=int)

# Ví dụ: Vẽ một kệ hàng hình chữ nhật
# Tọa độ góc trên trái (theo ô lưới)
shelf_row_start, shelf_col_start = 5, 10
# Kích thước kệ hàng (số ô)
shelf_rows, shelf_cols = 20, 2

grid_map[shelf_row_start : shelf_row_start + shelf_rows,
         shelf_col_start : shelf_col_start + shelf_cols] = 1 # Đánh dấu là kệ hàng

access_points = [
    (2, 2),   # Tọa độ (hàng, cột) của AP 1
    (2, num_cols - 3),
    (num_rows - 3, 2),
    (num_rows - 3, num_cols - 3),
    (num_rows // 2, num_cols // 2) # AP ở giữa
]

# Tùy chọn: Đánh dấu AP trên bản đồ lưới
for r_ap, c_ap in access_points:
    if 0 <= r_ap < num_rows and 0 <= c_ap < num_cols:
        grid_map[r_ap, c_ap] = 2 # Đánh dấu là AP



access_points = [
    (2, 2),   # Tọa độ (hàng, cột) của AP 1
    (2, num_cols - 3),
    (num_rows - 3, 2),
    (num_rows - 3, num_cols - 3),
    (num_rows // 2, num_cols // 2) # AP ở giữa
]

# Tùy chọn: Đánh dấu AP trên bản đồ lưới
for r_ap, c_ap in access_points:
    if 0 <= r_ap < num_rows and 0 <= c_ap < num_cols:
        grid_map[r_ap, c_ap] = 2 # Đánh dấu là AP

item_map = {} # Dictionary: (row, col) -> item_id
item_definitions = {
    101: "Sữa",
    102: "Bánh mì",
    201: "Nước ngọt",
    # ...
}

# Ví dụ: Đặt "Sữa" trên một phần của kệ hàng đã vẽ ở trên
# Giả sử kệ đó dành cho sữa
for r_item in range(shelf_row_start, shelf_row_start + shelf_rows // 2): # Nửa trên của kệ
    for c_item in range(shelf_col_start, shelf_col_start + shelf_cols):
        grid_map[r_item, c_item] = 101 # Mã mặt hàng sữa
        item_map[(r_item, c_item)] = 101


def plot_map(current_grid_map, aps, cart_pos=None, path=None, target_item_pos=None):
    plt.figure(figsize=(supermarket_width / 5, supermarket_height / 5)) # Điều chỉnh kích thước
    cmap = plt.cm.get_cmap('viridis', np.max(current_grid_map) + 1) # Tạo colormap
    plt.imshow(current_grid_map, cmap=cmap, origin='lower', interpolation='nearest')

    # Đánh dấu APs
    for r_ap, c_ap in aps:
        plt.scatter(c_ap, r_ap, marker='o', color='red', s=100, label='AP' if (r_ap, c_ap) == aps[0] else "")

    # Đánh dấu vị trí xe đẩy
    if cart_pos:
        plt.scatter(cart_pos[1], cart_pos[0], marker='s', color='blue', s=150, label='Xe đẩy')

    # Đánh dấu vị trí món hàng mục tiêu
    if target_item_pos:
        plt.scatter(target_item_pos[1], target_item_pos[0], marker='*', color='yellow', s=200, label='Món hàng')

    # Vẽ đường đi
    if path:
        path_rows, path_cols = zip(*path)
        plt.plot(path_cols, path_rows, color='cyan', linewidth=2, label='Đường đi')


    # Cài đặt hiển thị
    plt.xticks(np.arange(0, num_cols, step=max(1, num_cols // 10)),
               [f"{i*grid_resolution:.1f}" for i in np.arange(0, num_cols, step=max(1, num_cols // 10))])
    plt.yticks(np.arange(0, num_rows, step=max(1, num_rows // 10)),
               [f"{i*grid_resolution:.1f}" for i in np.arange(0, num_rows, step=max(1, num_rows // 10))])
    plt.xlabel("Chiều rộng (mét)")
    plt.ylabel("Chiều cao (mét)")
    plt.title("Bản đồ Siêu thị Mô phỏng")
    plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)

    # Tạo legend cho các loại ô (nếu cần)
    # patches = [plt.Rectangle((0,0),1,1,fc=cmap(i)) for i in range(np.max(current_grid_map)+1)]
    # labels = ['Lối đi', 'Kệ hàng', 'AP'] # Thêm các loại mặt hàng nếu cần
    # plt.legend(patches, labels, loc='upper right', bbox_to_anchor=(1.35, 1))
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
    plt.gca().invert_yaxis() # Để gốc tọa độ (0,0) ở góc trên bên trái như thường lệ của ảnh
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Điều chỉnh để legend không bị che
    plt.show()

# Vẽ bản đồ ban đầu
plot_map(grid_map, access_points)


