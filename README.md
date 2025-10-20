# paw_statistics
is Python framework for analysis of static hind paw postures 

## Paw keypoint segmentation and Analysis Pipeline

A complete, modular pipeline for **keypoint segmentation and analysis** built on [Detectron2](https://github.com/facebookresearch/detectron2).  
This toolkit allows you to train custom keypoint detection models, run inference, and perform advanced morphological and statistical analyses using standard python libraries

---

## Features
- **Inference and visualization** of predicted keypoints 
- **UI for Post-hoc corrections** of predicted keypoints
- **Quantitative analysis** (distance, angles, regression, clustering, etc.)
- Example Jupyter notebook for quick demonstration

---

## Installation



### download the code
```bash
    git clone https://github.com/ChristianPritz/paw_statistics
    cd paw_statistics
````

### create a venv or conda environment
  

```bash
	python -m venv .venv
    source .venv/bin/activate        # on Linux/Mac
    .venv\Scripts\activate           # on Windows
```
    
    conda (recommended)
```bash 
   conda create -n paw_statistics python=3.11
   conda activate paw_statistics
```
	

### install PyTorch and Detectron2 as specified in the Detectron2 release (link)
```bash  
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### install dependencies
```bash  
   pip install -r requirements.txt
```
### download model
```bash 
   python download_model.py
```

### verify installation
#### Package installation 
```bash
   python -c "import torch, detectron2, cv2, sklearn, statsmodels; print('Packages complete!')"
```
#### GPU 
```bash
   python -c "import torch; print(torch.cuda.is_available())"
 	
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); \
   print(f'Device count: {torch.cuda.device_count()}'); \
   print(f'Current device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```
## Trouble shoot 

| Issue | Solution |
|:------|:----------|
| **torch not found** | Reinstall PyTorch using the official command for your CUDA version. |
| **detectron2 build failed** | Make sure your PyTorch and CUDA versions are compatible. |
| **opencv import error** | Try reinstalling with `pip install opencv-python-headless`. |
| **GPU not detected** | Check `torch.cuda.is_available()` — update your NVIDIA drivers or CUDA toolkit. |

---


## 1. Clone the repository
```bash
    git clone https://github.com/ChristianPritz/paw_statistics
    cd paw_statistics
```
