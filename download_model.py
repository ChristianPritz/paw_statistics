#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 16:48:07 2025

@author: Christian Pritz
"""

import os
import urllib.request

# to get the download link get the file page url and add '/download' to the end
# i.e. url = 'https://osf.io/dc745/files/vcgmb/download'



def download_model(
    url="https://osf.io/dc745/files/vcgmb/download",
    dest="models/keypoint_model.pth"
):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
         print(f"✅ Model already present at {dest}")
         return dest
    print("⬇️ Downloading model weights...")
    urllib.request.urlretrieve(url, dest)
    print(f"✅ Model downloaded to {dest}")
    return dest

if __name__ == "__main__":
    download_model()

