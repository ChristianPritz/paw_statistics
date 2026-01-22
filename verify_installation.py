#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_installation.py
Checks imports of required packages and prints useful diagnostic messages.
"""

packages = {
    "cv2": "OpenCV (computer vision)",
    "sklearn": "Scikit-Learn (machine learning/statistics)",
    "pycricstat2": "pycricstat2 (custom / statistical)",
    "statsmodels": "Statsmodels (regression / statistics)"
    "torch": "PyTorch (machine learning)",
}

print("\n=== PACKAGE IMPORT VERIFICATION ===\n")

for pkg, description in packages.items():
    try:
        mod = __import__(pkg)
        print(f"✔ Successfully imported {pkg} — {description}")
    except ImportError as e:
        print(f"✖ ERROR importing {pkg} — {description}")
        print(f"  → {e}\n")
        print("Note the some torch erros do not break the paw_statistics")

# -------- CUDA CHECK --------
print("\n=== CUDA / GPU CHECK (PyTorch) ===\n")
try:
    import torch

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        print(f"GPU device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ No CUDA device detected. (CPU-only mode)")

except Exception as e:
    print("✖ Could not run CUDA check because PyTorch failed to import.")
    print(f"  → {e}")

print("\n✅ Verification finished.\n")
