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
current_gaze_point = {'x': 0, 'y': 0, 'valid': False}
frame_lock = threading.Lock()
is_calibrating = False
calibration_status = ""
camera_cap = None
camera_thread = None
pause_camera_feed = False

# Web calibration variables
calibration_points = []  # List of (x, y) screen coordinates for calibration
current_calibration_index = 0
calibration_gaze_data = []  # Collected gaze data at each point
model_instance = None
homtrans_instance = None

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
        """Web-based calibration - không dùng OpenCV GUI"""
        global is_calibrating, calibration_status, pause_camera_feed
        global calibration_points, current_calibration_index, calibration_gaze_data

        try:
            is_calibrating = True
            calibration_status = "Sẵn sàng - Đợi người dùng bắt đầu calibration trên web"
            
            print(f"Web calibration: Đợi người dùng nhấn 'Bắt đầu hiệu chỉnh'... Cần thu thập {len(calibration_points)} điểm")
            
            # Đợi cho đến khi calibration hoàn tất (được set từ endpoint)
            # Mỗi điểm: 3s hiển thị + 0.5s collect = 3.5s
            # 9 điểm × 3.5s = 31.5s + buffer → 60s timeout
            max_wait = 600  # 60 giây timeout (600 * 0.1s = 60s)
            waited = 0
            while is_calibrating and len(calibration_gaze_data) < len(calibration_points) and waited < max_wait:
                time.sleep(0.1)
                waited += 1
                if waited % 50 == 0:  # Print every 5 seconds
                    print(f"Calibration progress: {len(calibration_gaze_data)}/{len(calibration_points)}")
            
            if not is_calibrating:
                raise Exception("Calibration bị hủy")
            
            if len(calibration_gaze_data) < len(calibration_points):
                raise Exception(f"Không đủ dữ liệu calibration: chỉ có {len(calibration_gaze_data)}/{len(calibration_points)}")
            
            # Tính toán transformation matrix từ collected data
            print(f"Đã thu thập {len(calibration_gaze_data)} điểm calibration")
            
            # Tính transformation matrix (simplified version)
            STransG = self._compute_transformation(calibration_gaze_data, calibration_points)
            
            # Lưu vào instance
            self.STransG = STransG
            
            is_calibrating = False
            calibration_status = "Hoàn thành hiệu chỉnh - Bắt đầu tracking"
            
            print(f"Calibration hoàn tất. STransG matrix:\n{STransG}")
            return STransG

        except Exception as e:
            pause_camera_feed = False
            is_calibrating = False
            calibration_status = f"Lỗi hiệu chỉnh: {str(e)}"
            print(f"Lỗi trong quá trình hiệu chỉnh: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _compute_transformation(self, gaze_data, screen_points):
        """Tính toán transformation matrix từ gaze data và screen points"""
        try:
            print(f"Computing transformation from {len(gaze_data)} gaze samples and {len(screen_points)} screen points")
            
            gaze_vectors = np.array([d['gaze'] for d in gaze_data])
            screen_coords = np.array(screen_points)
            
            print(f"Gaze vectors shape: {gaze_vectors.shape}")
            print(f"Screen coords shape: {screen_coords.shape}")
            
            # Loại bỏ outliers bằng cách kiểm tra distance từ median
            if len(gaze_vectors) > 4:
                # Tính median của gaze vectors
                median_gaze = np.median(gaze_vectors[:, :2], axis=0)
                distances = np.linalg.norm(gaze_vectors[:, :2] - median_gaze, axis=1)
                median_dist = np.median(distances)
                
                # Loại bỏ điểm có distance > 3 * median (outliers)
                threshold = 3 * median_dist + 0.1  # +0.1 để tránh threshold = 0
                inliers = distances < threshold
                
                outliers_count = np.sum(~inliers)
                if outliers_count > 0:
                    print(f"⚠️ Loại bỏ {outliers_count} outliers khỏi {len(gaze_vectors)} điểm")
                    gaze_vectors = gaze_vectors[inliers]
                    screen_coords = screen_coords[inliers]
            
            # Tạo transformation matrix 4x4
            STransG = np.eye(4)
            
            if len(gaze_vectors) >= 3:
                # Dùng full affine transformation (6 parameters): 
                # screen_x = a*gaze_x + b*gaze_y + c
                # screen_y = d*gaze_x + e*gaze_y + f
                
                gaze_x = gaze_vectors[:, 0]
                gaze_y = gaze_vectors[:, 1]
                screen_x = screen_coords[:, 0]
                screen_y = screen_coords[:, 1]
                
                # Tạo matrix A cho least squares: [gaze_x, gaze_y, 1]
                A = np.column_stack([gaze_x, gaze_y, np.ones(len(gaze_x))])
                
                # Giải hệ phương trình: A @ params_x = screen_x
                try:
                    params_x = np.linalg.lstsq(A, screen_x, rcond=None)[0]
                    params_y = np.linalg.lstsq(A, screen_y, rcond=None)[0]
                    
                    # Fill transformation matrix với affine parameters
                    STransG[0, 0] = params_x[0]  # a (scale x theo gaze_x)
                    STransG[0, 1] = params_x[1]  # b (skew/rotation)
                    STransG[0, 3] = params_x[2]  # c (offset x)
                    STransG[1, 0] = params_y[0]  # d (skew/rotation)
                    STransG[1, 1] = params_y[1]  # e (scale y theo gaze_y)
                    STransG[1, 3] = params_y[2]  # f (offset y)
                    
                    # Tính error để đánh giá quality
                    pred_x = A @ params_x
                    pred_y = A @ params_y
                    error_x = np.mean(np.abs(pred_x - screen_x))
                    error_y = np.mean(np.abs(pred_y - screen_y))
                    
                    print(f"✅ Affine transformation computed:")
                    print(f"   X: a={params_x[0]:.2f}, b={params_x[1]:.2f}, c={params_x[2]:.2f} (error: {error_x:.1f}px)")
                    print(f"   Y: d={params_y[0]:.2f}, e={params_y[1]:.2f}, f={params_y[2]:.2f} (error: {error_y:.1f}px)")
                    
                except np.linalg.LinAlgError:
                    print("⚠️ Least squares failed, dùng simple scaling")
                    # Fallback: simple scaling
                    scale_x = (screen_x.max() - screen_x.min()) / (gaze_x.max() - gaze_x.min() + 1e-6)
                    scale_y = (screen_y.max() - screen_y.min()) / (gaze_y.max() - gaze_y.min() + 1e-6)
                    offset_x = screen_x.mean() - scale_x * gaze_x.mean()
                    offset_y = screen_y.mean() - scale_y * gaze_y.mean()
                    
                    STransG[0, 0] = scale_x
                    STransG[1, 1] = scale_y
                    STransG[0, 3] = offset_x
                    STransG[1, 3] = offset_y
            else:
                print("WARNING: Không đủ dữ liệu để tính transformation, sử dụng identity matrix")
            
            return STransG
        except Exception as e:
            print(f"Lỗi tính transformation: {e}")
            import traceback
            traceback.print_exc()
            return np.eye(4)

    def RunGazeOnScreen_web(self, model, cap, sfm=False):
        """
        Tracking KHÔNG dùng OpenCV window - chỉ gửi data lên web
        Không đọc camera vì đã có thread riêng đọc camera
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
                # Lấy frame từ biến toàn cục thay vì đọc từ camera
                with frame_lock:
                    if current_frame is None:
                        time.sleep(0.05)
                        continue
                    frame = current_frame.copy()

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

                    # Initialize default values
                    FSgaze = np.zeros((3, 1))
                    Sgaze = np.zeros((3, 1))
                    Sgaze2 = np.zeros((3, 1))

                    # Apply transformation to get screen coordinates
                    if hasattr(self, 'STransG') and self.STransG is not None:
                        # Use calibrated transformation
                        gaze_4d = np.array([gaze[0], gaze[1], gaze[2], 1.0])
                        screen_coords = self.STransG @ gaze_4d
                        gaze_x = max(0, min(int(screen_coords[0]), self.width))
                        gaze_y = max(0, min(int(screen_coords[1]), self.height))
                        # Set FSgaze to screen coords for logging
                        FSgaze = np.array([[screen_coords[0]], [screen_coords[1]], [0]])
                    else:
                        # Fallback if no calibration
                        if sfm:
                            FSgaze, Sgaze, Sgaze2 = self._getGazeOnScreen_sfm(gaze, WTransG1)
                        else:
                            FSgaze, Sgaze, Sgaze2 = self._getGazeOnScreen(gaze)
                        
                        gaze_pixel = self._mm2pixel(FSgaze.copy())
                        gaze_x = max(0, min(int(gaze_pixel[0][0]), self.width))
                        gaze_y = max(0, min(int(gaze_pixel[1][0]), self.height))

                    # Cập nhật gaze point cho web (không cần cập nhật frame vì đã có thread riêng)
                    with frame_lock:
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

            # Cleanup (KHÔNG release camera_cap vì nó là global và đang dùng cho camera feed)
            # cap.release()  # COMMENTED OUT - camera_cap sẽ được giữ cho lần chạy tiếp theo
            if out_video is not None and out_video.isOpened():
                out_video.release()

            # KHÔNG gọi cv2.destroyAllWindows() vì không có window nào

            # Lưu CSV
            if not df.empty:
                try:
                    df.columns = ['time_sec', 'gaze_x', 'gaze_y', 'gaze_z', 'Sgaze_x',
                                  'Sgaze_y', 'Sgaze_z', 'REyePos_x', 'REyePos_y',
                                  'LEyePos_x', 'LEyePos_y', 'yaw', 'pitch', 'roll',
                                  'HeadPos_x', 'HeadPos_y', 'set_x', 'set_y', 'set_z',
                                  'WTransG_x', 'WTransG_y', 'WTransG_z', 'RegSgaze_x',
                                  'RegSgaze_y', 'RegSgaze_z', 'CalPSgaze_x', 'CalPSgaze_y',
                                  'CalPSgaze_z', 'ROpenClose', 'LOpenClose']
                except ValueError as e:
                    print(f"Lỗi đặt tên cột: {e}. DataFrame có {len(df.columns)} cột")
                df = df.reset_index(drop=True)
                csv_path = os.path.join(self.dir, "results", "GazeTracking.csv")
                df.to_csv(csv_path)
                print(f"Đã lưu {len(df)} dòng dữ liệu vào {csv_path}")


        except Exception as e:
            import traceback
            traceback.print_exc()
            raise


def run_gaze_estimation(dir):
    global is_running, is_calibrating, calibration_status, current_gaze_point, current_frame, camera_cap
    global model_instance, homtrans_instance, calibration_points

    try:
        print("=== Bắt đầu run_gaze_estimation ===")
        output_directory = os.path.join(dir, "results")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        print("Đang load models...")
        start_model_load_time = time.time()
        model = EyeModel(dir)
        model_instance = model
        total_model_load_time = time.time() - start_model_load_time
        print(f"Model loaded in: {1000 * total_model_load_time:.1f}ms")

        homtrans = WebHomTransform(dir)
        homtrans_instance = homtrans

        # Sử dụng camera đã được khởi tạo sẵn
        if camera_cap is None or not camera_cap.isOpened():
            raise Exception("Camera chưa được khởi tạo hoặc không khả dụng")

        # Setup calibration points (9 điểm)
        screen_width = homtrans.width
        screen_height = homtrans.height
        margin_x = screen_width // 6
        margin_y = screen_height // 6
        
        calibration_points = [
            (margin_x, margin_y),                      # Top left
            (screen_width // 2, margin_y),             # Top center
            (screen_width - margin_x, margin_y),       # Top right
            (margin_x, screen_height // 2),            # Middle left
            (screen_width // 2, screen_height // 2),   # Center
            (screen_width - margin_x, screen_height // 2),  # Middle right
            (margin_x, screen_height - margin_y),      # Bottom left
            (screen_width // 2, screen_height - margin_y),  # Bottom center
            (screen_width - margin_x, screen_height - margin_y)  # Bottom right
        ]
        
        print(f"Calibration points setup: {len(calibration_points)} points")
        print("Camera OK, bắt đầu web calibration...")
        calibration_status = "Sẵn sàng - Nhấn 'Bắt đầu hiệu chỉnh' trên trang web"

        STransG = homtrans.calibrate_web(model, camera_cap, sfm=True)

        print(f"STransG:\n{np.array2string(STransG, formatter={'float': lambda x: f'{x:.2f}'})}")

        # Lưu transformation matrix vào homtrans
        homtrans.STransG = STransG

        cv2.destroyAllWindows()
        time.sleep(0.5)

        calibration_status = "Hoàn tất - Đang theo dõi điểm nhìn trên web"
        
        print("Bắt đầu tracking...")
        homtrans.RunGazeOnScreen_web(model, camera_cap, sfm=True)


    except Exception as e:
        print(f"LỖI trong run_gaze_estimation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Reset biến toàn cục
        with frame_lock:
            current_gaze_point = {'x': 0, 'y': 0, 'valid': False}

        cv2.destroyAllWindows()
        is_running = False
        is_calibrating = False
        model_instance = None
        homtrans_instance = None
        print("Đã dọn dẹp tài nguyên và thoát run_gaze_estimation")


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

    with frame_lock:
        if current_gaze_point is None:
            return jsonify({
                'x': 0,
                'y': 0,
                'valid': False
            })
        return jsonify(current_gaze_point)


@app.route('/calibration/start', methods=['POST'])
def start_calibration():
    """Bắt đầu quá trình calibration"""
    global current_calibration_index, calibration_gaze_data, is_calibrating, calibration_points
    
    if not is_running or model_instance is None:
        return jsonify({
            'status': 'error',
            'message': 'Vui lòng khởi động hệ thống trước'
        }), 400
    
    # Nhận window size từ frontend
    data = request.get_json()
    if data and 'width' in data and 'height' in data:
        screen_width = data['width']
        screen_height = data['height']
        
        # Recalculate calibration points dựa trên actual window size
        margin_x = screen_width // 6
        margin_y = screen_height // 6
        
        calibration_points = [
            (margin_x, margin_y),                      # Top left
            (screen_width // 2, margin_y),             # Top center
            (screen_width - margin_x, margin_y),       # Top right
            (margin_x, screen_height // 2),            # Middle left
            (screen_width // 2, screen_height // 2),   # Center
            (screen_width - margin_x, screen_height // 2),  # Middle right
            (margin_x, screen_height - margin_y),      # Bottom left
            (screen_width // 2, screen_height - margin_y),  # Bottom center
            (screen_width - margin_x, screen_height - margin_y)  # Bottom right
        ]
        
        print(f"Calibration points recalculated for window size {screen_width}x{screen_height}")
        print(f"Points: {calibration_points}")
    
    current_calibration_index = 0
    calibration_gaze_data = []
    
    return jsonify({
        'status': 'success',
        'message': 'Bắt đầu calibration',
        'total_points': len(calibration_points)
    })


@app.route('/calibration/current_point', methods=['GET'])
def get_current_calibration_point():
    """Lấy điểm calibration hiện tại"""
    global current_calibration_index, calibration_points, calibration_gaze_data
    
    # Số điểm đã thu thập
    collected = len(calibration_gaze_data)
    total = len(calibration_points)
    
    print(f"[current_point] DEBUG: collected={collected}, total={total}, gaze_data_len={len(calibration_gaze_data)}")
    
    # Nếu đã thu đủ, trả về done
    if collected >= total:
        print(f"[current_point] Returning done=True (collected {collected}/{total})")
        return jsonify({
            'done': True,
            'index': collected,
            'total': total
        })
    
    # Trả về điểm tiếp theo cần thu thập
    point = calibration_points[collected]
    print(f"[current_point] Returning point {collected}/{total}: ({point[0]}, {point[1]})")
    return jsonify({
        'done': False,
        'index': collected,
        'total': total,
        'x': point[0],
        'y': point[1]
    })


@app.route('/calibration/collect', methods=['POST'])
def collect_calibration_data():
    """Thu thập gaze data tại điểm calibration hiện tại"""
    global current_calibration_index, calibration_gaze_data, model_instance, current_frame, is_calibrating
    
    collected_before = len(calibration_gaze_data)
    print(f"[collect] Starting collection. Currently have {collected_before} points")
    print(f"[collect] is_calibrating={is_calibrating}, model_instance={model_instance is not None}, current_frame={current_frame is not None}")
    
    if model_instance is None:
        print(f"[collect] CRITICAL ERROR: model_instance is None! Calibration may have been interrupted.")
        return jsonify({
            'status': 'error',
            'message': 'Model bị mất - có thể hệ thống bị dừng. Vui lòng reload trang và thử lại!'
        }), 500
        
    if current_frame is None:
        print(f"[collect] ERROR: current_frame is None")
        return jsonify({
            'status': 'error',
            'message': 'Camera chưa sẵn sàng'
        }), 400
    
    try:
        # Collect multiple samples at this point
        samples = []
        for i in range(10):  # 10 samples
            with frame_lock:
                if current_frame is not None:
                    frame = current_frame.copy()
                    eye_info = model_instance.get_gaze(frame)
                    if eye_info is not None:
                        samples.append(eye_info['gaze'])
                        print(f"[collect] Sample {i+1}/10: gaze={eye_info['gaze'][:2]}")
            time.sleep(0.05)
        
        print(f"[collect] Collected {len(samples)}/10 samples")
        
        if len(samples) > 0:
            # Average the gaze vectors
            avg_gaze = np.mean(samples, axis=0)
            
            # Lưu gaze data với index là số điểm hiện tại
            point_index = len(calibration_gaze_data)
            calibration_gaze_data.append({
                'gaze': avg_gaze,
                'point_index': point_index
            })
            
            collected = len(calibration_gaze_data)
            total = len(calibration_points)
            
            print(f"[collect] SUCCESS: Added point {point_index}. Now have {collected}/{total} points total")
            print(f"[collect] Gaze data array length: {len(calibration_gaze_data)}")
            
            return jsonify({
                'status': 'success',
                'collected': collected,
                'total': total,
                'done': collected >= total
            })
        else:
            print(f"[collect] ERROR: No valid eye data in {len(samples)} samples")
            return jsonify({
                'status': 'error',
                'message': 'Không phát hiện được mắt'
            }), 400
            
    except Exception as e:
        print(f"[collect] EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


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


def start_camera_feed():
    """Bắt đầu camera feed ngay khi app khởi động"""
    global camera_cap, current_frame, camera_thread, pause_camera_feed
    
    try:
        import platform
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_DSHOW
        camera_cap = cv2.VideoCapture(0, backend)
        if not camera_cap.isOpened():
            camera_cap = cv2.VideoCapture(0)
        
        if camera_cap.isOpened():
            camera_cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            camera_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
            
            def update_camera():
                global current_frame, pause_camera_feed
                while camera_cap is not None and camera_cap.isOpened():
                    if not pause_camera_feed:  # Chỉ đọc khi không bị pause
                        ret, frame = camera_cap.read()
                        if ret:
                            with frame_lock:
                                current_frame = frame.copy()
                    time.sleep(0.03)
            
            camera_thread = threading.Thread(target=update_camera, daemon=True)
            camera_thread.start()
            print("Camera feed started successfully")
        else:
            print("Failed to open camera")
    except Exception as e:
        print(f"Error starting camera: {e}")

if __name__ == '__main__':
    # Khởi động camera trước khi chạy Flask
    start_camera_feed()
    app.run(debug=True, port=5000, threaded=True, use_reloader=False)