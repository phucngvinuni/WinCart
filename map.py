# config.py

# --- Kích thước và Độ phân giải Bản đồ ---
SUPERMARKET_WIDTH_M = 50
SUPERMARKET_HEIGHT_M = 30
GRID_RESOLUTION_M = 0.5 # mét/ô

# --- Mã cho các loại ô trên bản đồ ---
CELL_TYPE_PATH = 0
CELL_TYPE_SHELF = 1

# --- Vị trí AP (tọa độ ô lưới) ---
AP_MARGIN_CELLS = 2 # Số ô cách mép

# --- Tham số Mô phỏng RSSI ---
P_TX_MAX_RSSI = -30     # dBm (RSSI tối đa khi ở rất gần AP, không vật cản)
PATH_LOSS_EXPONENT_N = 2.8 # Hệ số suy hao đường truyền
SHELF_ATTENUATION_DB = 4.0 # Suy hao qua mỗi đơn vị kệ hàng (dB)
NOISE_STD_DEV_DB = 3.0     # Độ lệch chuẩn của nhiễu Gaussian (dB)
MIN_RSSI_THRESHOLD = -95   # Ngưỡng RSSI tối thiểu có thể phát hiện

# --- Tham số KNN ---
K_NEIGHBORS = 3
USE_WEIGHTED_KNN = True
EPSILON_WEIGHT = 1e-6 # Giá trị nhỏ để tránh chia cho 0 trong weighted KNN

# --- Màu sắc cho trực quan hóa ---
COLOR_PATH_LINE = 'cyan'
COLOR_TARGET_ITEM_MARKER = 'yellow'
COLOR_CART_ACTUAL_MARKER = 'blue'
COLOR_CART_ESTIMATED_MARKER = 'green'
COLOR_ERROR_LINE = 'magenta'
COLOR_AP_MARKER = 'red'
COLOR_SHELF_ON_MAP = 'gray' # Màu cho kệ hàng trên bản đồ chính
COLOR_PATH_ON_MAP = 'white' # Màu cho lối đi trên bản đồ chính
COLOR_TARGET_ITEM_MARKER = 'yellow'