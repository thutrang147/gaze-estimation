# Webcam based Gaze Estimation
Estimate gaze on computer screen

## Installation Guidelines

### 1. Set Up Python Virtual Environment

Create a virtual Python environment (Python 3.8-3.11 recommended):

```bash
# Using conda (recommended)
conda create --name gaze_estimation python=3.10
conda activate gaze_estimation

# OR using venv
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies

All dependencies have been consolidated into a single requirements.txt file.

#### macOS ARM (M1/M2/M3/M4) - Special Setup Required

OpenVINO pip wheels are corrupted on Apple Silicon. Use conda-forge instead:

```bash
# Install OpenVINO from conda-forge first
conda install -c conda-forge openvino

# Then install remaining dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### Windows/Linux/Intel Mac

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: Installation may take several minutes as it downloads PyTorch, OpenVINO, and other large packages.

**For Windows users**: If you need H.264 video codec support, download `openh264-1.8.0-win64.dll` from [here](https://github.com/cisco/openh264/releases/tag/v1.8.0) and copy it to your environment's root folder (e.g., `C:\Anaconda3\envs\gaze_estimation`).

### 3. Grant Camera Permission (macOS only)

On macOS, the Terminal needs camera access. When you first run the application, you'll be prompted to grant permission:
- Go to **System Settings** → **Privacy & Security** → **Camera**
- Enable checkbox for **Terminal** (or **iTerm2** if you use that)
- Restart your terminal after granting permission

### 4. Run the Application

**Option A: OpenVINO-based model (recommended for speed)**
```bash
python src/main.py
```

**Option B: PyTorch pl_gaze model**
```bash
python src/main_pl.py
```

This model is based on [pytorch_mpiigaze_demo](https://github.com/hysts/pytorch_mpiigaze_demo). Training code available at [pl_gaze_estimation](https://github.com/hysts/pl_gaze_estimation/tree/main).

**Option C: Web interface (Flask app)**
```bash
python src/app.py
```
Then open http://localhost:5000 in your browser.

**Option D: Compare with Tobii Eye Tracker**
```bash
python src/main_compareWithTobii.py
```
*Requires Tobii Eye Tracker 5 SDK and a custom executable file.*

## Modifications

This repository is based on the original work by Lucas Falch and Katrin Solveig Lohan. Key modifications include:

- **macOS ARM (Apple Silicon) compatibility**: Fixed OpenVINO integration to work with M1/M2/M3/M4 chips
- **Screen size estimation**: Added fallback for physical screen dimensions on macOS where screeninfo returns None
- **OpenVINO 2025 support**: Updated code to support both OpenVINO 2024 and 2025 API versions
- **Removed deprecated imports**: Fixed turtle module import issue
- **Updated installation guide**: Added specific instructions for macOS ARM setup

## Credits & Original Work

This project is based on:
- **Original Repository**: [WebCamGazeEstimation](https://github.com/FalchLucas/WebCamGazeEstimation) by Lucas Falch
- **Research Paper**: Lucas Falch and Katrin Solveig Lohan, "Webcam-based gaze estimation for computer screen interaction", Frontiers in Robotics and AI, Volume 11 - 2024 | https://doi.org/10.3389/frobt.2024.1369566

If you use this code in an academic context, please cite the original paper above.

### Additional Credits

The PyTorch pl_gaze model is based on:
- [pytorch_mpiigaze_demo](https://github.com/hysts/pytorch_mpiigaze_demo)
- [pl_gaze_estimation](https://github.com/hysts/pl_gaze_estimation/tree/main)
