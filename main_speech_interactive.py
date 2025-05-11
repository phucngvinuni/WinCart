# main_speech_interactive.py
import numpy as np
import time
import matplotlib.pyplot as plt # Cần cho plt.show() và plt.pause()
import speech_recognition as sr
import re # Cho hàm extract_keywords
import sys
import config # Các hằng số cấu hình chung
from supermarket_model import SupermarketMap # Lớp quản lý bản đồ
import rssi_simulation
import localization_algorithms
import interactive_visualization # Lớp quản lý plot tương tác

# --- Biến trạng thái toàn cục của mô phỏng ---
supermarket = None
rssi_fingerprints_data = None
interactive_plotter = None # Instance của InteractiveSupermarketPlotter

# Biến lưu trạng thái xe đẩy hiện tại
# (Vị trí thực tế sẽ được cập nhật bởi click chuột hoặc mô phỏng di chuyển)
# Vị trí ước tính sẽ được tính toán sau mỗi lần có vị trí thực tế mới
current_cart_actual_rc = None
current_cart_estimated_rc_float = None


# --- HÀM NHẬN DIỆN GIỌNG NÓI ---
def recognize_speech_from_mic(recognizer, microphone):
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")
    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    with microphone as source:
        # print("Điều chỉnh theo tiếng ồn xung quanh (1 giây)...")
        try:
            recognizer.adjust_for_ambient_noise(source, duration=0.5) # Giảm thời gian điều chỉnh
        except Exception as e:
            print(f"Lỗi khi adjust_for_ambient_noise: {e}")
            # Tiếp tục mà không điều chỉnh nếu microphone có vấn đề
        # print(f"Ngưỡng năng lượng: {recognizer.energy_threshold:.2f}")
        print("\n🎤 Hãy nói yêu cầu của bạn (ví dụ: 'tìm táo', 'dẫn đến quầy thịt tươi')...")
        try:
            audio = recognizer.listen(source, timeout=config.SPEECH_RECOGNITION_TIMEOUT,
                                      phrase_time_limit=config.SPEECH_RECOGNITION_PHRASE_LIMIT)
        except sr.WaitTimeoutError:
            print("🔇 Không có âm thanh nào được phát hiện.")
            return {"success": False, "error": "timeout", "transcription": None}
        except Exception as e:
            print(f"Lỗi khi listen: {e}")
            return {"success": False, "error": "listen_error", "transcription": None}


    # print("Đã ghi nhận! Đang xử lý...")
    response = {"success": True, "error": None, "transcription": None}
    try:
        response["transcription"] = recognizer.recognize_google(audio, language="vi-VN")
    except sr.RequestError:
        response["success"] = False
        response["error"] = "API unavailable (mất kết nối mạng?)"
    except sr.UnknownValueError:
        response["error"] = "Không thể nhận diện được giọng nói" # Không phải lỗi API
    except Exception as e:
        response["success"] = False
        response["error"] = f"Lỗi không xác định khi nhận diện: {e}"
    return response

# --- HÀM TRÍCH XUẤT TỪ KHÓA MẶT HÀNG/GIAN HÀNG ---
def extract_target_from_speech(text, supermarket_obj: SupermarketMap):
    """
    Trích xuất từ khóa mặt hàng hoặc gian hàng từ văn bản.
    Trả về (loại_mục_tiêu, tên_chuẩn_hóa_của_mục_tiêu)
    loại_mục_tiêu: "item", "stall", hoặc None
    """
    if not text: return None, None
    text_lower = text.lower()

    # Ưu tiên tìm tên mặt hàng đầy đủ trước
    sorted_item_defs = sorted(supermarket_obj.item_definitions.items(), key=lambda x: len(x[1]['name']), reverse=True)
    for item_id, item_props in sorted_item_defs:
        item_name_defined = item_props['name']
        if item_name_defined.lower() in text_lower:
            print(f"Tìm thấy từ khóa mặt hàng (khớp cụm): '{item_name_defined}'")
            return "item", item_name_defined

    # Tiếp theo, tìm tên gian hàng đầy đủ
    sorted_stall_defs = sorted(supermarket_obj.stall_definitions.items(), key=lambda x: len(x[1]['name']), reverse=True)
    for stall_id, stall_props in sorted_stall_defs:
        stall_name_defined = stall_props['name']
        if stall_name_defined.lower() in text_lower:
            print(f"Tìm thấy từ khóa gian hàng (khớp cụm): '{stall_name_defined}'")
            return "stall", stall_name_defined

    # Nếu không khớp cụm, thử khớp từng từ (đơn giản hơn)
    words = re.findall(r'\b\w+\b', text_lower)
    for word in words:
        for item_id, item_props in supermarket_obj.item_definitions.items():
            if word == item_props['name'].lower():
                print(f"Tìm thấy từ khóa mặt hàng (khớp 1 từ): '{item_props['name']}'")
                return "item", item_props['name']
        for stall_id, stall_props in supermarket_obj.stall_definitions.items():
            if word == stall_props['name'].lower():
                print(f"Tìm thấy từ khóa gian hàng (khớp 1 từ): '{stall_props['name']}'")
                return "stall", stall_props['name']

    print(f"Không tìm thấy từ khóa nào phù hợp trong: '{text}'")
    return None, None

# --- HÀM XỬ LÝ LOGIC KHI CLICK LÊN BẢN ĐỒ ---
def handle_map_click_event(clicked_cart_actual_rc):
    global supermarket, rssi_fingerprints_data, interactive_plotter
    global current_cart_actual_rc, current_cart_estimated_rc_float

    print(f"handle_map_click_event được gọi với vị trí: {clicked_cart_actual_rc}")
    current_cart_actual_rc = clicked_cart_actual_rc # Cập nhật vị trí thực tế mới

    # 1. Thực hiện định vị
    observed_rssi = rssi_simulation.get_observed_rssi_at_cart_on_map(
        supermarket, current_cart_actual_rc
    )
    # print(f"  RSSI quan sát được (từ click): {[round(val, 1) for val in observed_rssi]}")

    estimated_pos = localization_algorithms.predict_location_knn(
        observed_rssi,
        rssi_fingerprints_data,
        config.K_NEIGHBORS,
        config.USE_WEIGHTED_KNN,
        config.EPSILON_WEIGHT
    )

    if estimated_pos:
        current_cart_estimated_rc_float = estimated_pos
        error_m = rssi_simulation.euclidean_distance_m(current_cart_actual_rc, current_cart_estimated_rc_float)
        print(f"  Vị trí ước tính (ô): ({estimated_pos[0]:.2f}, {estimated_pos[1]:.2f}), Sai số: {error_m:.2f}m")
        interactive_plotter.update_cart_location(
            current_cart_actual_rc, current_cart_estimated_rc_float, error_m,
            message="Đã đặt xe đẩy. Sẵn sàng nhận lệnh thoại."
        )
    else:
        print("  Không thể định vị xe đẩy.")
        current_cart_estimated_rc_float = None
        interactive_plotter.update_cart_location(
            current_cart_actual_rc, None, None,
            message="Lỗi định vị sau khi click."
        )
    # Không tự động hỏi món hàng ở đây, để vòng lặp chính xử lý giọng nói


# --- HÀM MÔ PHỎNG DI CHUYỂN XE ĐẨY ---
def simulate_cart_movement_along_path(path_rc_nodes):
    global supermarket, rssi_fingerprints_data, interactive_plotter
    global current_cart_actual_rc, current_cart_estimated_rc_float

    if not path_rc_nodes or interactive_plotter is None:
        print("Không có đường đi hoặc plotter để mô phỏng di chuyển.")
        return

    print("\n🚗 Bắt đầu mô phỏng di chuyển xe đẩy...")
    initial_estimated_pos_for_path = current_cart_estimated_rc_float # Giữ lại vị trí ước tính ban đầu

    for i, step_rc in enumerate(path_rc_nodes):
        current_cart_actual_rc = step_rc # Cập nhật vị trí thực tế của xe đẩy

        # Định vị lại xe đẩy tại vị trí mới này
        observed_rssi_at_step = rssi_simulation.get_observed_rssi_at_cart_on_map(
            supermarket, current_cart_actual_rc
        )
        estimated_pos_at_step = localization_algorithms.predict_location_knn(
            observed_rssi_at_step, rssi_fingerprints_data,
            config.K_NEIGHBORS, config.USE_WEIGHTED_KNN, config.EPSILON_WEIGHT
        )

        error_m_at_step = None
        if estimated_pos_at_step:
            current_cart_estimated_rc_float = estimated_pos_at_step # Cập nhật vị trí ước tính
            error_m_at_step = rssi_simulation.euclidean_distance_m(current_cart_actual_rc, current_cart_estimated_rc_float)
            # print(f"  Bước {i+1}: Xe ở {step_rc}, Ước tính: ({estimated_pos_at_step[0]:.1f},{estimated_pos_at_step[1]:.1f}), Sai số: {error_m_at_step:.2f}m")

        # Cập nhật plot
        interactive_plotter.update_cart_location(
            current_cart_actual_rc, current_cart_estimated_rc_float, error_m_at_step
        )
        # Giữ nguyên đường đi mục tiêu và tên mục tiêu (nếu có)
        interactive_plotter.current_path_rc_nodes = path_rc_nodes # Vẽ lại toàn bộ đường
        # interactive_plotter.target_item_name_display và target_approachable_pos_rc không đổi

        interactive_plotter.fig.canvas.flush_events() # Cập nhật giao diện
        plt.pause(0.3) # Tạm dừng để mô phỏng tốc độ, plt.pause thay vì time.sleep cho matplotlib

    print("✅ Hoàn thành di chuyển.")
    interactive_plotter.clear_path_and_target(message="Đã đến nơi! Sẵn sàng nhận lệnh mới.")
    # Giữ lại vị trí xe đẩy cuối cùng
    interactive_plotter.update_cart_location(
        current_cart_actual_rc, current_cart_estimated_rc_float,
        interactive_plotter.localization_error_m, # Giữ lại lỗi cuối cùng
        message="Đã đến nơi! Sẵn sàng."
    )

# --- HÀM CHÍNH ĐỂ CHẠY MÔ PHỎNG ---
def run_interactive_simulation():
    global supermarket, rssi_fingerprints_data, interactive_plotter
    global current_cart_actual_rc, current_cart_estimated_rc_float

    # 1. Khởi tạo đối tượng SupermarketMap
    supermarket = SupermarketMap(
        width_m=config.SUPERMARKET_WIDTH_M,
        height_m=config.SUPERMARKET_HEIGHT_M,
        resolution_m=config.GRID_RESOLUTION_M
    )

    # 2. Thêm các thành phần vào bản đồ (tường, gian hàng, vật phẩm, AP)
    # --- TƯỜNG BAO ---
    supermarket.add_general_obstacle(0, 0, 1, supermarket.num_cols) # Tường trên
    supermarket.add_general_obstacle(supermarket.num_rows - 1, 0, 1, supermarket.num_cols) # Tường dưới
    supermarket.add_general_obstacle(0, 0, supermarket.num_rows, 1) # Tường trái
    supermarket.add_general_obstacle(0, supermarket.num_cols - 1, supermarket.num_rows, 1) # Tường phải

    # --- GIAN HÀNG VÀ MẶT HÀNG ---
    # (Đây là phần bạn cần tùy chỉnh nhiều nhất cho bản đồ của mình)
    # Kệ Trái (chứa Sữa và Bánh mì)
    shelf1_r, shelf1_c = supermarket.num_rows // 4, supermarket.num_cols // 5
    shelf1_h, shelf1_w = supermarket.num_rows // 2, 3 # Kệ cao, rộng 3 ô
    stall_id_shelf1 = supermarket.add_stall_area(shelf1_r, shelf1_c, shelf1_h, shelf1_w, "Kệ Đồ Khô")
    if stall_id_shelf1 != -1:
        supermarket.add_item_to_grid(shelf1_r, shelf1_c, shelf1_h // 2, shelf1_w, "Sữa", on_stall_id=stall_id_shelf1)
        supermarket.add_item_to_grid(shelf1_r + shelf1_h // 2, shelf1_c, shelf1_h - (shelf1_h // 2), shelf1_w, "Bánh mì", on_stall_id=stall_id_shelf1)

    # Kệ Phải (chứa Nước ngọt)
    shelf2_r, shelf2_c = supermarket.num_rows // 4, (supermarket.num_cols // 5) * 3
    shelf2_h, shelf2_w = supermarket.num_rows // 2, 3
    stall_id_shelf2 = supermarket.add_stall_area(shelf2_r, shelf2_c, shelf2_h, shelf2_w, "Kệ Nước Giải Khát")
    if stall_id_shelf2 != -1:
        supermarket.add_item_to_grid(shelf2_r, shelf2_c, shelf2_h, shelf2_w, "Nước ngọt", on_stall_id=stall_id_shelf2)
    
    # Thêm một kệ nữa cho đa dạng
    shelf3_r, shelf3_c = supermarket.num_rows // 2 + 5 , supermarket.num_cols - 15
    shelf3_h, shelf3_w = 10, 4
    stall_id_shelf3 = supermarket.add_stall_area(shelf3_r, shelf3_c, shelf3_h, shelf3_w, "Kệ Gia Vị")
    if stall_id_shelf3 != -1:
        supermarket.add_item_to_grid(shelf3_r, shelf3_c, shelf3_h, shelf3_w, "Nước mắm", on_stall_id=stall_id_shelf3)


    # 3. Thêm Access Points
    ap_coords = [
        (config.AP_MARGIN_CELLS, config.AP_MARGIN_CELLS),
        (config.AP_MARGIN_CELLS, supermarket.num_cols - 1 - config.AP_MARGIN_CELLS),
        (supermarket.num_rows - 1 - config.AP_MARGIN_CELLS, config.AP_MARGIN_CELLS),
        (supermarket.num_rows - 1 - config.AP_MARGIN_CELLS, supermarket.num_cols - 1 - config.AP_MARGIN_CELLS)
        # (supermarket.num_rows // 2, supermarket.num_cols // 2) # AP ở giữa (tùy chọn)
    ]
    for r_ap, c_ap in ap_coords:
        supermarket.add_access_point(r_ap, c_ap)
    print(f"Đã thêm {len(supermarket.access_points)} APs.")


    # 4. Tạo bản đồ fingerprint RSSI
    print("Đang tạo bản đồ RSSI fingerprints...")
    rssi_fingerprints_data = rssi_simulation.generate_rssi_fingerprints_from_map(supermarket)
    print(f"Kích thước của rssi_fingerprints_data trong bộ nhớ: {sys.getsizeof(rssi_fingerprints_data)} bytes")
    if rssi_fingerprints_data:
        # Kiểm tra kích thước của một vài value (là list hoặc numpy array)
        example_key = next(iter(rssi_fingerprints_data))
        example_value = rssi_fingerprints_data[example_key]
        print(f"Kích thước của một value trong fingerprints: {sys.getsizeof(example_value)} bytes, kiểu: {type(example_value)}")
        if isinstance(example_value, np.ndarray):
            print(f"  Kiểu dữ liệu của array: {example_value.dtype}, số phần tử: {example_value.size}")
    print(f"Hoàn thành tạo bản đồ RSSI fingerprints với {len(rssi_fingerprints_data)} điểm.")

    # 5. Khởi tạo và hiển thị bản đồ tương tác
    print("\nĐang khởi tạo giao diện đồ họa...")
    interactive_plotter = interactive_visualization.InteractiveSupermarketPlotter(
        supermarket,
        on_map_click_callback_func=handle_map_click_event # Gán hàm callback
    )
    interactive_plotter.current_message_on_plot = "Click vào lối đi để đặt xe đẩy hoặc nói lệnh."
    interactive_plotter.update_dynamic_plot_elements() # Vẽ trạng thái ban đầu

    # 6. Khởi tạo Nhận dạng Giọng nói
    recognizer = sr.Recognizer()
    try:
        microphone = sr.Microphone()
        # Kiểm tra microphone một lần
        with microphone as source:
            print("Kiểm tra microphone...")
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
        print("Microphone sẵn sàng.")
        speech_enabled = True
    except Exception as e:
        print(f"Lỗi khởi tạo microphone: {e}. Chức năng giọng nói sẽ bị tắt.")
        print("Bạn vẫn có thể tương tác bằng cách click chuột để đặt xe đẩy (nhưng không có tìm đường).")
        speech_enabled = False
        interactive_plotter.current_message_on_plot = "Lỗi Mic. Chỉ click để đặt xe."
        interactive_plotter.update_dynamic_plot_elements()

    # 7. Vòng lặp chính của chương trình (kết hợp plt.pause và xử lý giọng nói)
    # plt.show(block=False) # Hiển thị non-blocking
    interactive_plotter.fig.show() # Cách khác để hiển thị non-blocking

    try:
        while True: # Vòng lặp chính của ứng dụng
            # Xử lý sự kiện của Matplotlib để giữ cho cửa sổ tương tác
            # và cho phép hàm onclick được gọi
            plt.pause(0.1) # Quan trọng: cho phép GUI cập nhật và xử lý sự kiện
                           # Đồng thời không làm CPU chạy 100%

            if speech_enabled and current_cart_actual_rc: # Chỉ lắng nghe nếu xe đẩy đã được đặt
                print("\n------------------------------------------")
                print(f"Xe đẩy đang ở vị trí thực tế: {current_cart_actual_rc}, ước tính: {current_cart_estimated_rc_float}")
                speech_response = recognize_speech_from_mic(recognizer, microphone)
                
                plot_msg = None
                target_approachable_rc = None
                target_name_for_plot = None
                path_for_plot = None

                if speech_response["transcription"]:
                    spoken_text = speech_response["transcription"]
                    print(f"Người dùng nói: \"{spoken_text}\"")
                    interactive_plotter.current_message_on_plot = f"Đã nghe: \"{spoken_text}\". Đang xử lý..."
                    interactive_plotter.update_dynamic_plot_elements()
                    plt.pause(0.01)


                    keyword_type, found_name = extract_target_from_speech(spoken_text, supermarket)

                    if found_name:
                        target_name_for_plot = found_name
                        if keyword_type == "item":
                            target_approachable_rc = supermarket.get_approachable_item_location_by_name(
                                found_name, current_cart_estimated_rc_float # Ưu tiên điểm gần xe đẩy ước tính
                            )
                        elif keyword_type == "stall":
                            target_approachable_rc = supermarket.get_stall_approachable_location(
                                found_name, current_cart_estimated_rc_float
                            )
                        
                        if target_approachable_rc:
                            print(f"  Mục tiêu '{found_name}' ({keyword_type}) có điểm tiếp cận tại: {target_approachable_rc}")
                            # Điểm bắt đầu tìm đường là vị trí *ước tính* của xe đẩy, đã làm tròn
                            if current_cart_estimated_rc_float:
                                start_node_path = (round(current_cart_estimated_rc_float[0]),
                                                   round(current_cart_estimated_rc_float[1]))
                                # Đảm bảo start_node_path là ô đi được
                                if supermarket.grid_map[start_node_path[0],start_node_path[1]] != PATHWAY_ID and \
                                   supermarket.grid_map[start_node_path[0],start_node_path[1]] != AP_ID :
                                    print(f"  Cảnh báo: Vị trí ước tính {start_node_path} không phải lối đi. Dùng vị trí thực tế.")
                                    start_node_path = current_cart_actual_rc

                                print(f"  Tìm đường từ {start_node_path} đến {target_approachable_rc}...")
                                path_nodes = localization_algorithms.find_path_astar(
                                    supermarket, start_node_path, target_approachable_rc
                                )
                                if path_nodes:
                                    path_for_plot = path_nodes
                                    plot_msg = f"Đang dẫn đường đến: {found_name}"
                                    interactive_plotter.update_path_to_target(found_name, target_approachable_rc, path_nodes, plot_msg)
                                    simulate_cart_movement_along_path(path_nodes) # Mô phỏng di chuyển
                                else:
                                    plot_msg = f"Không tìm thấy đường đến: {found_name}"
                            else:
                                plot_msg = "Chưa định vị được xe đẩy để tìm đường."
                        else:
                            plot_msg = f"Không tìm thấy điểm tiếp cận cho: {found_name}"
                    else:
                        plot_msg = f"Không hiểu rõ yêu cầu: \"{spoken_text}\""
                
                elif speech_response["error"] and speech_response["error"] != "timeout" and speech_response["error"] != "Unable to recognize speech":
                    # Chỉ hiển thị lỗi API nghiêm trọng, bỏ qua lỗi không nghe thấy hoặc không nhận diện
                    plot_msg = f"Lỗi nhận diện: {speech_response['error']}"

                if plot_msg: # Cập nhật plot nếu có thay đổi hoặc thông báo
                    interactive_plotter.current_message_on_plot = plot_msg
                    interactive_plotter.update_dynamic_plot_elements()

            elif speech_enabled and not current_cart_actual_rc:
                interactive_plotter.current_message_on_plot = "Vui lòng click lên bản đồ để đặt vị trí ban đầu cho xe đẩy."
                interactive_plotter.update_dynamic_plot_elements()
                plt.pause(0.5) # Chờ một chút để người dùng đọc


    except KeyboardInterrupt:
        print("\nThoát chương trình mô phỏng.")
    finally:
        plt.close('all') # Đảm bảo đóng tất cả cửa sổ plot khi thoát


if __name__ == "__main__":
    run_interactive_simulation()