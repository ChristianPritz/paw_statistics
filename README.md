# paw_statistics

is a Python framework for analysis of static hind paw postures.
NOTE: The repository is being updated: first update will be fully completed by 30/08/2026, second update mid September 2026. We apologize for any inconvenience. 

## Paw keypoint segmentation and Analysis Pipeline

A complete, modular pipeline for keypoint segmentation and analysis built on Detectron2.
This toolkit, implemented in PyTorch, enables you to run inference on images of mouse hindpaws, review and correct keypoint predictions through an intuitive user interface, and perform morphological and statistical analyses using common Python libraries. The models can be custom-trained within a Detectron2 environment.

---

## Features
- Inference and visualization of predicted keypoints
- UI for post-hoc correction of predicted keypoints
- Quantitative analysis (distances, angles, regression, clustering, circular stats)
- Local Tkinter application and an all-in-one Google Colab notebook
- YOLO object detection plus hind/front specialist pose models

---

## Requirements

Inference requires Python 3.10 or newer, Tkinter, PyTorch, Ultralytics, and the packages listed in `requirements.txt`. Tkinter is normally supplied by Python itself; Conda users can install it with `conda install tk` if needed.

The current inference pipeline uses YOLO models through Ultralytics.

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
- For development, use an editable install:
```bash
pip install -e .
```

6) Fetch the model and data from the OSF.io repository
```bash
python fetch_from_osf.py
```

The default public OSF resource ID used by the example is `dc745`.

8) Verify installation and GPU (if applicable)
```bash
python verify_installation.py
```


---
## Known issues

* reduced keypoint placement accuracy when paws are closely spaced
* Colab sessions are temporary. Download saved ZIP/CSV results before disconnecting the runtime.



---

## License & Contact
Please add a LICENSE file if you plan to publish. For questions or issues, open an issue on GitHub: https://github.com/ChristianPritz/paw_statistics/issues
