# paw_statistics

is a Python framework for analysis of static hind paw postures.

## Paw keypoint segmentation and Analysis Pipeline

A complete, modular pipeline for **keypoint segmentation and analysis** built on Detectron2.
This toolkit lets you train custom keypoint detection models, run inference, correct predictions with a simple UI, and run morphological and statistical analyses using common Python libraries.

---

## Features
- Inference and visualization of predicted keypoints
- UI for post-hoc correction of predicted keypoints
- Quantitative analysis (distances, angles, regression, clustering, circular stats)
- Example Jupyter notebook for quick demonstration

---

## Full install (default — Detectron2 + GPU or CPU)

These instructions are the default installation path and cover installing PyTorch, Detectron2, and all project dependencies so you can run inference and the full example pipeline. Make sure you choose the PyTorch wheel that matches your CUDA version (or use the CPU wheel if you do not have a GPU).

1) Clone the repository
```bash
git clone https://github.com/ChristianPritz/paw_statistics
cd paw_statistics
```

2) Create and activate a virtual environment (recommended)
- Unix / macOS:
```bash
python -m venv .venv
source .venv/bin/activate
```
- Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
- Or use conda (recommended if you use GPU):
```bash
conda create -n paw_statistics python=3.11
conda activate paw_statistics
```

3) Upgrade pip and install PyTorch matching your CUDA (examples)
- CUDA 12.1 example (adjust to your CUDA):
```bash
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
- CPU-only example:
```bash
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

4) Install Detectron2
We recommend using a pre-built wheel that matches your PyTorch and CUDA versions. A simple (git) install is provided but may build from source on your machine:
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
If the install tries to build and fails, follow the Detectron2 installation guide for the correct prebuilt wheel for your environment.

5) Install project dependencies
```bash
pip install -r requirements.txt
```
Note: requirements.txt contains core analysis deps (numpy, pandas, scipy, matplotlib, scikit-learn, pycircstat, opencv-python-headless). If you plan to run the README verification command, install statsmodels as well:
```bash
pip install statsmodels
```

6) Fetch the model and data from the OSF.io repository
```bash
python fetch_from_osf.py
```

7) Verify installation and GPU (if applicable)
```bash
python -c "import torch, detectron2, cv2, sklearn, statsmodels; print('Packages complete!')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); \nprint(f'Device count: {torch.cuda.device_count()}'); \nprint(f'Current device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

---

## Quick CPU-only smoke test (alternative)

If you want to get started quickly without Detectron2 or GPU support (useful for running analysis notebooks and tests that do not require model inference), follow this shorter path: 

1) Clone and create/activate environment (same as above)
2) Install CPU-only PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
3) Install project dependencies:
```bash
pip install -r requirements.txt
```
4) Quick verification:
```bash
python -c "import torch, cv2, sklearn; print('Basic packages OK', torch.__version__)"
```

This path avoids Detectron2 and CUDA and is ideal for first tests of data analysis and plotting features.

---

## Troubleshooting (common issues)

| Issue | Solution |
|---|---|
| torch not found | Reinstall PyTorch with the official command for your CUDA or CPU configuration. |
| detectron2 build failed | Ensure PyTorch and CUDA versions match and a C/C++ build toolchain (nvcc, gcc/clang, build essentials) is available. Consider using a pre-built wheel for your platform. |
| opencv import error | The repo uses opencv-python-headless; try `pip install opencv-python-headless` or `pip install --upgrade opencv-python-headless`. |
| GPU not detected | Check NVIDIA drivers and CUDA toolkit; run `python -c "import torch; print(torch.cuda.is_available())"` |

---

## Requirements
The repository includes a `requirements.txt` with core Python dependencies (numpy, pandas, scipy, matplotlib, scikit-learn, pycircstat, opencv-python-headless). Use the virtual environment steps above to isolate installs.

## Usage
- Example notebooks and example scripts demonstrate how to run inference (once Detectron2 is installed), correct keypoints, and run statistical analyses.
- For development, use an editable install (if you add setup files):
```bash
pip install -e .
pip install -e .[dev]   # if dev extras are provided
```

---

## License & Contact
Please add a LICENSE file if you plan to publish. For questions or issues, open an issue on GitHub: https://github.com/ChristianPritz/paw_statistics/issues
