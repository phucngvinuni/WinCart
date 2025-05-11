# config.py

# --- Kích thước và Độ phân giải Bản đồ ---
SUPERMARKET_WIDTH_M = 50
SUPERMARKET_HEIGHT_M = 30
GRID_RESOLUTION_M = 2 # mét/ô

# --- Mã ID cho các loại ô trên bản đồ (import từ supermarket_model) ---
# Chúng ta sẽ import trực tiếp từ supermarket_model trong các file khác khi cần
# để tránh phụ thuộc vòng tròn nếu supermarket_model cũng cần config.

# --- Vị trí AP (tọa độ ô lưới) ---
AP_MARGIN_CELLS = 2 # Số ô cách mép

# --- Tham số Mô phỏng RSSI ---
P_TX_MAX_RSSI = -30     # dBm (RSSI tối đa khi ở rất gần AP, không vật cản)
PATH_LOSS_EXPONENT_N = 2.8 # Hệ số suy hao đường truyền
SHELF_ATTENUATION_DB = 4.0 # Suy hao qua mỗi đơn vị kệ hàng/vật cản (dB)
NOISE_STD_DEV_DB = 0.2     # Độ lệch chuẩn của nhiễu Gaussian (dB)
MIN_RSSI_THRESHOLD = -95   # Ngưỡng RSSI tối thiểu có thể phát hiện

# --- Tham số KNN ---
K_NEIGHBORS = 3
USE_WEIGHTED_KNN = True
EPSILON_WEIGHT = 1e-6 # Giá trị nhỏ để tránh chia cho 0 trong weighted KNN

# --- Màu sắc cho trực quan hóa ---
COLOR_PATH_ON_MAP = 'white'
COLOR_OBSTACLE_ON_MAP = 'dimgray'
COLOR_AP_CELL_ON_MAP = 'lightcoral' # Màu nền cho ô chứa AP (nếu vẽ)
COLOR_STALL_BASE = 'skyblue'      # Màu cơ sở cho gian hàng (sẽ được điều chỉnh)
COLOR_ITEM_BASE = 'lightgreen'    # Màu cơ sở cho item (sẽ được điều chỉnh)

COLOR_AP_MARKER = 'red'
COLOR_CART_ACTUAL_MARKER = 'blue'
COLOR_CART_ESTIMATED_MARKER = 'green'
COLOR_TARGET_ITEM_MARKER = 'magenta' # Đổi màu để phân biệt với AP
COLOR_PATH_LINE = 'cyan'
COLOR_ERROR_LINE = 'orange'

# --- Speech Recognition ---
SPEECH_RECOGNITION_TIMEOUT = 5 # giây
SPEECH_RECOGNITION_PHRASE_LIMIT = 10 # giây