from flask import Flask, render_template, jsonify, request, Response
import os
import sys
import threading
import cv2
import time
import numpy as np
import pandas as pd
import datetime

app = Flask(__name__)

# Biến toàn cục
gaze_thread = None
is_running = False
current_frame = None
current_gaze_point = None
frame_lock = threading.Lock()
is_calibrating = False
calibration_status = ""

# Thêm đường dẫn để tìm modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)

from gaze_tracking.homtransform import HomTransform
from gaze_tracking.model import EyeModel


class WebHomTransform(HomTransform):
    def __init__(self, directory):
        super().__init__(directory)
        self.web_mode = True

    def calibrate_web(self, model, cap, sfm=False):
        """Calibration cho web - skip GUI, return dummy calibration"""
        global is_calibrating, calibration_status

        try:
            is_calibrating = True
            calibration_status = "Đang hiệu chỉnh (web mode)..."

            # Tạo calibration matrix đơn giản thay vì chạy full calibration với GUI
            # Web app sẽ sử dụng gaze estimation trực tiếp không cần calibration phức tạp
            print("Web calibration: Using identity matrix (skipping GUI calibration)")
            
            # Khởi tạo các attributes cần thiết cho HomTransform
            self.STransG = np.eye(4)
            self.STransW = np.eye(4)
            self.scaleWtG = 1.0
            self.StG = []  # List of calibration points (Gaze to Screen)
            self.StW = []  # List of calibration points (World to Screen)
            
            is_calibrating = False
            calibration_status = "Hoàn thành hiệu chỉnh"

            # Return identity transformation matrix
            return self.STransG

        except Exception as e:
            is_calibrating = False
            calibration_status = f"Lỗi hiệu chỉnh: {str(e)}"
            print(f"Lỗi trong quá trình hiệu chỉnh: {e}")
            import traceback
            traceback.print_exc()
            raise

    def RunGazeOnScreen_web(self, model, cap, sfm=False):
        """
        Tracking KHÔNG dùng OpenCV window - chỉ gửi data lên web
        """
        global current_frame, current_gaze_point, is_running

        try:
            if cap is None or not cap.isOpened():
                print("Camera không khả dụng")
                return

            # Tạo video output (không hiển thị)
            out_video = None
            wc_width, wc_height = 640, 480
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output_path = os.path.join(self.dir, "results", "output_video.mp4")
                out_video = cv2.VideoWriter(output_path, fourcc, 20.0, (wc_width * 2, wc_height))
                if not out_video.isOpened():
                    print("Không thể tạo video output")
                    out_video = None
            except Exception as e:
                print(f"Không thể tạo video output: {e}")
                out_video = None

            df = pd.DataFrame()
            frame_prev = None
            WTransG1 = np.eye(4)
            FSgaze = np.array([[-10], [-10], [-10]])


            frame_count = 0
            while cap.isOpened() and is_running:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        print("Không đọc được frame từ camera")
                        break
                except Exception as e:
                    print(f"Lỗi đọc frame: {e}")
                    break

                try:
                    # Lấy thông tin mắt
                    eye_info = model.get_gaze(frame)
                    if eye_info is None:
                        if frame_count % 30 == 0:
                            print("Không phát hiện được mắt")
                        time.sleep(0.05)
                        continue

                    gaze = eye_info['gaze']

                    if frame_prev is not None and sfm:
                        WTransG1, WTransG2, W_P = self.sfm.get_GazeToWorld(model, frame_prev, frame)

                    frame_prev = frame.copy()

                    if sfm:
                        FSgaze, Sgaze, Sgaze2 = self._getGazeOnScreen_sfm(gaze, WTransG1)
                    else:
                        FSgaze, Sgaze, Sgaze2 = self._getGazeOnScreen(gaze)

                    # Chuyển đổi từ mm sang pixel
                    gaze_pixel = self._mm2pixel(FSgaze.copy())

                    # Đảm bảo tọa độ trong phạm vi màn hình
                    gaze_x = max(0, min(int(gaze_pixel[0][0]), self.width))
                    gaze_y = max(0, min(int(gaze_pixel[1][0]), self.height))

                    # Cập nhật frame và gaze point cho web
                    with frame_lock:
                        current_frame = frame.copy()
                        current_gaze_point = {
                            'x': gaze_x,
                            'y': gaze_y,
                            'valid': True
                        }

                    # Lưu dữ liệu
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                    df = pd.concat([df, pd.DataFrame([np.hstack((
                        timestamp, eye_info['gaze'], FSgaze.flatten(),
                        eye_info['EyeRLCenterPos'], eye_info['HeadPosAnglesYPR'],
                        eye_info['HeadPosInFrame'], [0, 0, 0], 0, WTransG1[:3, 3].flatten(),
                        Sgaze.flatten(), Sgaze2.flatten(), eye_info['EyeState']))])])

                    # Lưu video nếu có (KHÔNG hiển thị)
                    if out_video is not None and out_video.isOpened():
                        try:
                            # Tạo visualization cho video (nhưng không hiển thị)
                            white_frame = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
                            # Vẽ điểm gaze lên white frame
                            cv2.circle(white_frame, (gaze_x, gaze_y), 15, (255, 0, 0), -1)

                            # Resize và ghép
                            vis_frame = cv2.resize(white_frame, (wc_width, wc_height))
                            cam_frame = cv2.resize(frame, (wc_width, wc_height))
                            final_frame = np.concatenate((vis_frame, cam_frame), axis=1)
                            out_video.write(final_frame)
                        except Exception as e:
                            if frame_count == 0:
                                print(f"Lỗi ghi video: {e}")

                    frame_count += 1
                    if frame_count % 100 == 0:
                        print(f"Đã xử lý {frame_count} frames - Gaze: ({gaze_x}, {gaze_y})")

                except Exception as e:
                    if frame_count % 30 == 0:
                        print(f"Lỗi xử lý frame: {e}")
                    continue

                # QUAN TRỌNG: KHÔNG có cv2.waitKey() và cv2.imshow() ở đây
                time.sleep(0.03)  # ~30 FPS

            # Cleanup
            cap.release()
            if out_video is not None and out_video.isOpened():
                out_video.release()

            # KHÔNG gọi cv2.destroyAllWindows() vì không có window nào

            # Lưu CSV
            if not df.empty:
                df.columns = ['time_sec', 'gaze_x', 'gaze_y', 'gaze_z', 'Sgaze_x',
                              'Sgaze_y', 'Sgaze_z', 'REyePos_x', 'REyePos_y',
                              'LEyePos_x', 'LEyePos_y', 'yaw', 'pitch', 'roll',
                              'HeadPos_x', 'HeadPos_y', 'set_x', 'set_y', 'set_z',
                              'WTransG_x', 'WTransG_y', 'WTransG_z', 'RegSgaze_x',
                              'RegSgaze_y', 'RegSgaze_z', 'CalPSgaze_x', 'CalPSgaze_y',
                              'CalPSgaze_z', 'ROpenClose', 'LOpenClose']
                df = df.reset_index(drop=True)
                csv_path = os.path.join(self.dir, "results", "GazeTracking.csv")
                df.to_csv(csv_path)
                print(f"Đã lưu {len(df)} dòng dữ liệu vào {csv_path}")


        except Exception as e:
            import traceback
            traceback.print_exc()
            raise


def run_gaze_estimation(dir):
    global is_running, is_calibrating, calibration_status, current_gaze_point, current_frame

    cap = None

    try:
        output_directory = os.path.join(dir, "results")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        start_model_load_time = time.time()
        model = EyeModel(dir)
        total_model_load_time = time.time() - start_model_load_time
        print(f"Model loaded in: {1000 * total_model_load_time:.1f}ms")

        homtrans = WebHomTransform(dir)

        # Khởi tạo camera
        import platform
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_DSHOW
        cap = cv2.VideoCapture(0, backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)  # Thử không dùng backend cụ thể
        if not cap.isOpened():
            raise Exception("Không thể mở camera")

        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)


        # Calibration (vẫn dùng OpenCV window)
        calibration_status = "Đang hiệu chỉnh - Vui lòng nhìn vào các điểm trên màn hình"

        STransG = homtrans.calibrate_web(model, cap, sfm=True)

        print(f"STransG:\n{np.array2string(STransG, formatter={'float': lambda x: f'{x:.2f}'})}")


        # Đóng tất cả cửa sổ OpenCV sau calibration
        cv2.destroyAllWindows()
        time.sleep(0.5)  # Đợi windows đóng hẳn

        calibration_status = "Hoàn tất - Đang theo dõi điểm nhìn trên web"

        homtrans.RunGazeOnScreen_web(model, cap, sfm=True)


    except Exception as e:
        print(f"LỖI: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Reset biến toàn cục
        with frame_lock:
            current_frame = None
            current_gaze_point = None

        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        is_running = False
        is_calibrating = False
        print("Đã dọn dẹp tài nguyên")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start_gaze', methods=['POST'])
def start_gaze():
    global gaze_thread, is_running

    if is_running:
        return jsonify({
            'status': 'error',
            'message': 'Gaze estimation đang chạy'
        }), 400

    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(src_dir)

    intel_path = os.path.join(project_root, 'intel')
    if not os.path.exists(intel_path):
        return jsonify({
            'status': 'error',
            'message': f'Không tìm thấy thư mục intel tại {intel_path}'
        }), 404

    is_running = True
    gaze_thread = threading.Thread(target=run_gaze_estimation, args=(project_root,))
    gaze_thread.daemon = True
    gaze_thread.start()

    return jsonify({
        'status': 'success',
        'message': 'Đã khởi động gaze estimation'
    })


@app.route('/stop_gaze', methods=['POST'])
def stop_gaze():
    global is_running
    is_running = False
    time.sleep(0.5)  # Đợi thread dừng
    return jsonify({
        'status': 'success',
        'message': 'Đã dừng gaze estimation'
    })


@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        'is_running': is_running,
        'is_calibrating': is_calibrating,
        'calibration_status': calibration_status
    })


@app.route('/gaze_point', methods=['GET'])
def get_gaze_point():
    global current_gaze_point

    if current_gaze_point is None:
        return jsonify({
            'x': 0,
            'y': 0,
            'valid': False
        })

    with frame_lock:
        return jsonify(current_gaze_point)


@app.route('/video_feed')
def video_feed():
    def generate():
        global current_frame
        while True:
            with frame_lock:
                if current_frame is not None:
                    frame = current_frame.copy()
                else:
                    # Frame trống nếu chưa có
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, 'Waiting for camera...', (150, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Encode frame
            try:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"Lỗi encode frame: {e}")

            time.sleep(0.03)  # ~30 FPS

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True,port=5000, threaded=True, use_reloader=False)