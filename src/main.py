import os
import sys
import cv2
import time
import numpy as np

# Thêm đường dẫn để tìm modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Lên thư mục gốc
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)

from gaze_tracking.homtransform import HomTransform
from gaze_tracking.model import EyeModel


def main(dir):
    try:
        output_directory = os.path.join(dir, "results")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        print(f"Loading models from: {dir}")

        start_model_load_time = time.time()
        model = EyeModel(dir)
        total_model_load_time = time.time() - start_model_load_time
        print(f"Total time to load model: {1000 * total_model_load_time:.1f}ms")

        homtrans = HomTransform(dir)
        # cap=cv2.VideoCapture(0)
        """ for higher resolution (max available: 1920x1080) """
        # Use CAP_AVFOUNDATION on macOS instead of CAP_DSHOW (Windows)
        import platform
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_DSHOW
        cap = cv2.VideoCapture(0, backend)
        # cap.set(cv2.CAP_PROP_SETTINGS, 1)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

        """ Calibration """
        STransG = homtrans.calibrate(model, cap, sfm=True)

        print("============================")
        print(f"STransG\n{np.array2string(STransG, formatter={'float': lambda x: f'{x:.2f}'})}")

        homtrans.RunGazeOnScreen(model, cap, sfm=True)

        # gocv.PlotPupils(gray_image, prediction, morphedMask, falseColor, centroid)

    except Exception as e:
        print(f"Something wrong when running EyeModel: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Lấy thư mục gốc project (WebCamGazeEstimation/)
    current_file = os.path.abspath(__file__)  # src/main.py
    src_dir = os.path.dirname(current_file)  # src/
    project_root = os.path.dirname(src_dir)  # WebCamGazeEstimation/

    # Thư mục chứa models (intel/ nằm ở root)
    dir = project_root

    print(f"Project root: {project_root}")
    print(f"Model directory: {dir}")
    print(f"Intel folder exists: {os.path.exists(os.path.join(dir, 'intel'))}")

    main(dir)