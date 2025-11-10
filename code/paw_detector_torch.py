#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:37:31 2024

@author: Christian Pritz
"""
import cv2
import torch
import numpy as np
from torchvision.ops import nms



class paw_detector:
    """
    Torchvision based detector for deployment
    """

    def __init__(self, model_path: str, device: str = "cpu",threshold=0.9):
        self.device = torch.device(device)
        self.model = torch.jit.load(model_path).to(self.device)
        self.model.eval()
        self.threshold = threshold

    # ---------------------------
    # internal helper
    

    def _preprocess(self,img_bgr):
        # detectron2 like normalization values 
        # ---------------------------
        
        
        # 1. convert to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        #2. Resize shortest side to 1344px (initial size limit in training)
        h, w = img_rgb.shape[:2]
        scale = 1344 / min(h, w)
        new_w, new_h = int(scale * w), int(scale * h)
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
        # 3. Convert to CHW tensor
        tensor = torch.as_tensor(img_rgb).permute(2, 0, 1).float()
    
        # 4. Detectron normalization (RGB order)
        PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        PIXEL_STD = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1)
        tensor = (tensor - PIXEL_MEAN) / PIXEL_STD
    
        return img_rgb, tensor, scale  


    # ---------------------------
    # PUBLIC API (preserves old syntax)
    # ---------------------------
    @torch.no_grad()
    # standardized  detector 
    def detect(self, img_bgr,threshold=None):
        if threshold is None:
            threshold = self.threshold
        """
        Runs inference on a single image.
        Returns: dict exactly like old detector:
        {
          "boxes": np.ndarray (N x 4),
          "classes": np.ndarray (N,),
          "logits": np.ndarray (N,),
          "keypoints": np.ndarray (K x 2),
          "prob": np.ndarray (N,)
        }
        """
        img_rgb, tensor,scale = self._preprocess(img_bgr)
        #outputs = self.model([{"image": tensor}])
        outputs = self.model(tensor)
        

    
        # These indices are from your TorchScript head output format...........
        bxs = outputs[0].cpu()
        cls = outputs[1].cpu()
        logits = outputs[2].cpu()
        pts = outputs[3].cpu()
        prob = outputs[4].cpu()
        
        #thresholding of the output............................................
        
        bxs = bxs[prob>threshold]
        cls = cls[prob>threshold]
        logits = logits[prob>threshold]
        pts = pts[prob>threshold]
        prob = prob[prob>threshold]
        
        # reshape keypoints
        #pts = pts.numpy().reshape(pts.shape[0],pts.shape[1], pts.shape[2])

        # optional NMS for eliminating overlapping boxes
        keep = nms(bxs, prob, iou_threshold=0.9)

        result = {
            "boxes": bxs[keep].numpy()/scale,
            "classes": cls[keep].numpy(),
            "logits": logits[keep].numpy(),
            "keypoints": pts[keep].numpy()/scale,
            "prob": prob[keep].numpy(),
            "image_rgb": img_rgb,     # convenience for visualization
        }

        return result

    @torch.no_grad()

    #Wrapper for UI embedding 
    def detect_4_UI(self, img_bgr,threshold=None):
        if threshold is None:
            threshold = self.threshold
        results = self.detect(img_bgr)
        return results["boxes"],results["keypoints"],results["classes"],results["prob"]
    
    def detect_batch(self, image_list):
        """
        Same logic as detect_single, just supports multiple images.
        image_list: list of np.ndarray (BGR images)
        """
        results = []
        for img_bgr in image_list:
            results.append(self.detect_single(img_bgr))
        return results





