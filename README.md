# paw_statistics

is a Python framework for analysis of static hind paw postures.

## Paw keypoint segmentation and Analysis Pipeline

A complete, modular pipeline for keypoint segmentation and analysis built on Detectron2.
This toolkit, implemented in PyTorch, enables you to run inference on images of mouse hindpaws, review and correct keypoint predictions through an intuitive user interface, and perform morphological and statistical analyses using common Python libraries. The models can be custom-trained within a Detectron2 environment.

---

## Features
- Inference and visualization of predicted keypoints
- UI for post-hoc correction of predicted keypoints
- Quantitative analysis (distances, angles, regression, clustering, circular stats)
- Example Jupyter notebook for quick demonstration

---

## Requirements

Inference requires PyTorch and the packages listed in requirements.txt, including core Python dependencies such as numpy, pandas, scipy, matplotlib, scikit-learn, pycircstat2, and opencv-python-headless.

Detectron2 **is not required for inference**, but custom training of the model does require a full Detectron2 installation.

Use the virtual environment setup instructions above to keep dependencies isolated.

---


## Installation (Pytorch + GPU or CPU)

These instructions are the default installation path and cover installing PyTorch, and all project dependencies so you can run inference and the full example pipeline. Make sure you choose the PyTorch wheel that matches your CUDA version (or use the CPU wheel if you do not have a GPU).

1) Clone the repository
```bash
git clone https://github.com/ChristianPritz/paw_statistics
cd paw_statistics
```

2) Create and activate a virtual environment (recommended)
- Use conda (recommended if you use GPU):
```bash
conda create -n paw_statistics python=3.11
conda activate paw_statistics
```
- or alternatively virtualenv in Unix / macOS:
```bash
python -m venv .venv
source .venv/bin/activate
```
- or virtualenv in Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
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
4) Install project dependencies
```bash
pip install -r requirements.txt
```
5) Install into the environment
- Example notebooks and example scripts demonstrate how to run inference (once Detectron2 is installed), correct keypoints, and run statistical analyses.
- For development, use an editable install (if you add setup files):
```bash
pip install -e .
```

6) Fetch the model and data from the OSF.io repository
```bash
python fetch_from_osf.py
```

8) Verify installation and GPU (if applicable)
```bash
python verify_installation.py
```

---
## Known issues

* The deployed model (model_torch.pt) performs less accurately than the Detectron2 version (model.pth) due to limitations in model tracing, resulting in:
  
  - a higher false-negative rate
  
  - increased keypoint and bounding box placement error
  
* reduced keypoint placement accuracy when paws are closely spaced



## Optional Detectron2 installation
If you want use the detectron2 model (model.pth) for custom keypoint segmentation please install detectron2 according to the installation page: https://detectron2.readthedocs.io/en/latest/tutorials/install.html 

The pth file can be custom trained using detectron2. 

---

## Troubleshooting (common issues)

| Issue | Solution |
|---|---|
| torch not found | Reinstall PyTorch with the official command for your CUDA or CPU configuration. |
| detectron2 build failed | Ensure PyTorch and CUDA versions match and a C/C++ build toolchain (nvcc, gcc/clang, build essentials) is available. Consider using a pre-built wheel for your platform. |
| opencv import error | The repo uses opencv-python-headless; try `pip install opencv-python-headless` or `pip install --upgrade opencv-python-headless`. |
| GPU not detected | Check NVIDIA drivers and CUDA toolkit; run `python -c "import torch; print(torch.cuda.is_available())"` |

---

## License & Contact
Please add a LICENSE file if you plan to publish. For questions or issues, open an issue on GitHub: https://github.com/ChristianPritz/paw_statistics/issues
