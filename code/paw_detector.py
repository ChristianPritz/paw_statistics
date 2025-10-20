#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:37:31 2024

@author: wormulon
"""
#TESTING THE TRAINED DETECTOR//////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////


from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random, glob
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.data import MetadataCatalog
import random
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from scipy.spatial import distance_matrix
from PIL import Image
from IPython import embed
import os


class paw_detector():
    def __init__(self,model_path,base_path,threshold = 0.8,device="cuda",
                 keypoint_names = None, keypoint_logic = None,
                 visualize = True,detector_settings=None):   
        
        if detector_settings is None:
            self.detector_settings = {"keypoint_names":[
                "heel","base_i","tip_i","base_ii","phal_ii","tip_ii",
                "base_iii","phal_iii","tip_iii","base_iv","phal_iv","tip_iv",
                "base_v","phal_v","tip_v"
            ],"keypoint_logic":[
                [0,1],[0,3],[0,6],[0,9],[0,12],
                [1,2],[3,4],[4,5],[6,7],[7,8],
                [9,10],[10,11],[12,13],[13,14],
                [1,3],[3,6],[6,9],[9,12]
            ],"thing_classes":["mouse_hind_paw_left", "mouse_hind_paw_right"],
            "thing_dataset_id_to_contiguous_id":{1: 0, 2: 1},
            "NUM_WORKERS":2,
            "IMS_PER_BATCH":5,
            "BATCH_SIZE_PER_IMAGE":512,}
        else: 
            self.detector_settings = detector_settings
 
      
        #self.threshold = threshold
        self.base_path = base_path    
        self.threshold = threshold
        self.device = device
        self.visualize = visualize
        #self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        if os.path.exists(model_path):
            self.model_path = model_path
        else:
            raise ValueError("[MODEL PATH ERROR] Model.pth file can't be found. Please set the model path in the detector settings: detector_settings['model_path'] = 'your/path/to/the/model/model.pth' or pass the model path directly to the paw_detector")        
        #MetadataCatalog.get("worms_train").thing_classes = ["worm",]
        self.intialize_model(keypoint_names = keypoint_names,keypoint_logic = keypoint_logic)
        
            
    def intialize_model(self,keypoint_names = None,keypoint_logic=None):
        # --- Defaults ---
        if keypoint_names is None:
            keypoint_names = self.detector_settings["keypoint_names"]
    
        if keypoint_logic is None:
            keypoint_logic = self.detector_settings["keypoint_logic"]
    
        # --- Keypoint connection rules ---
        color = np.asarray([255, 48, 12])
        keypoint_connection_rules = [
            (keypoint_names[a], keypoint_names[b], tuple(np.uint8(color)))
            for a, b in keypoint_logic
        ]
    
        # --- Manual Metadata registration (no JSON needed) ---
        dataset_name = "meta_paws_inference"
        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)
    
        # Dummy dataset function (not used, just required to register)
        DatasetCatalog.register(dataset_name, lambda: [])
    
        meta = MetadataCatalog.get(dataset_name)
        meta.thing_classes = self.detector_settings["thing_classes"]
        meta.thing_dataset_id_to_contiguous_id = self.detector_settings["thing_dataset_id_to_contiguous_id"]
        meta.keypoint_names = keypoint_names
        meta.keypoint_flip_map = ()
        meta.keypoint_connection_rules = keypoint_connection_rules
        meta.evaluator_type = "coco"
    
        # --- Save metadata for later use ---
        self.metaData = meta
        self.num_keypoints = len(keypoint_names)
    
        # --- Config setup ---
        self.cfg = get_cfg()
        self.cfg.MODEL.DEVICE = self.device
        self.cfg.merge_from_file(
            model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")
        )
        self.cfg.MODEL.WEIGHTS = self.model_path
        self.cfg.DATASETS.TRAIN = (dataset_name,)
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = self.detector_settings["NUM_WORKERS"]
        self.cfg.SOLVER.IMS_PER_BATCH = self.detector_settings["IMS_PER_BATCH"]
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = self.detector_settings["BATCH_SIZE_PER_IMAGE"]
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(meta.thing_classes)
        self.cfg.MODEL.RETINANET.NUM_CLASSES = len(meta.thing_classes)
        self.cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = self.num_keypoints
        self.cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((self.num_keypoints, 1), dtype=float).tolist()
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
    
        # --- Predictor ---
        self.predictor = DefaultPredictor(self.cfg)



    def imload(self,im_path):
        im = cv2.imread(im_path)
        
        return im
    
    def visualize_prediction(self,outputs,im):
        
        v = Visualizer(im[:,:,::-1], metadata=self.metaData, scale=1.2)
          #v = Visualizer(im[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        image = out.get_image()[:, :, ::-1]
        
        height,width = image.shape[:2]
        dpi = 300  # Define DPI (dots per inch)
        width_inch = width / dpi
        height_inch = height / dpi
        fig = plt.figure(dpi=dpi)
        fig.set_size_inches(width_inch, height_inch, forward=False)
        
        #fig =plt.subplots(figsize=(width_inch, height_inch),dpi=dpi) 
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)

        ax.imshow(image,origin='upper') 
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        plt.xlim(0, width)
        plt.ylim(height, 0) 
        fig = plt.gcf()
        fig.canvas.draw()
        plot_image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        
        # Convert the PIL Image to a NumPy array
        img = np.asarray(plot_image)

        resized_image = cv2.resize(img, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)
        return resized_image
    
    

        
    def detect(self,im,return_overlay):
        #visualize = self.visualize

        outputs = self.predictor(im)
        pts = outputs["instances"].get_fields()["pred_keypoints"].to("cpu")
        bxs = outputs["instances"].get_fields()["pred_boxes"].to("cpu")
        clss = outputs["instances"].get_fields()["pred_classes"].to("cpu")
        #pts = self.control_points(pts)
        #x,angles = self.pts_2_angles(pts)
        if self.visualize :
            self.overlay = self.visualize_prediction(outputs,im)
            
        return pts,bxs,clss
    
    def cut_image(self, im, bboxes, margin_percent):
     
        bmax = np.max(bboxes,axis=0)
        bmin = np.min(bboxes,axis=0)
        sides = np.asarray([bmax[2]-bmin[0],bmax[3]-bmin[1]])
        center_of_mass = np.asarray([np.mean([bmax[2],bmin[0]]),
                                    np.mean([bmax[3],bmin[1]])])
        half_side = np.max(sides)/2
        tolerance = np.max(im.shape[:2])*margin_percent/100
        new_x1 = center_of_mass[0] - half_side - tolerance
        new_y1 = center_of_mass[1]- half_side - tolerance
        new_x2 = center_of_mass[0] + half_side + tolerance
        new_y2 = center_of_mass[1] + half_side + tolerance

        return im[int(new_y1):int(new_y2), int(new_x1):int(new_x2),:]
    
    def detect_single(self,im_path,reFrame=False,return_overlay=False):
       #visualize = self.visualize
        im = self.imload(im_path)
        if reFrame:
            pts,bx,clss = self.detect(im,False)
            bxs = bx.tensor.numpy()

            im = self.cut_image( im, bxs, 5)
            

            
        pts,bx = self.detect(im,return_overlay)
        if return_overlay:
            img = self.overlay
        else:
            img = im
        return pts,bx,img,clss
    
    def detect_from_im(self,im,reFrame=False,return_overlay=False):
        if reFrame:
            pts,bx,clss = self.detect(im,False)
            bxs = bx.tensor.numpy()

            im = self.cut_image( im, bxs, 5)
            

            
        pts,bx,clss = self.detect(im,return_overlay)
        if return_overlay:
            img = self.overlay
        else:
            img = im

        return pts,bx,img,clss
    def detect_batch(self,ims):
        splines = np.empty((ims.shape[0],self.num_keypoints,2))
        boxes = np.empty((ims.shape[0],4))
        classes = np.empty(ims.shape[0],2)
        for idx,i in enumerate(ims):
            pts,bx,_,clss = self.detect(i)
            splines[idx,:,:] = pts
            boxes[idx,:] = bx
            classes[idx,:] = clss
        return splines,boxes,classes
 

def from_directory(directory,detector,plotting=True):

    image_extensions = ['tif', 'tiff', 'jpg','JPG']
    image_files = []
    image_names = []

    directory_path = os.path.abspath(directory)

    for ext in image_extensions:
        # Create a pattern for each extension using glob
        pattern = os.path.join(directory_path, f'*.{ext}')
        # Use glob to get a list of files matching the pattern
        image_files.extend(glob.glob(pattern, recursive=True))
    A,B,C = [],[],[]
    for i in image_files:

        try:
           
            
            a,b,_,c = detector.detect_single(i,reFrame=False)
            A.append(a)
            B.append(b)
            C.append(c)
            if plotting :
                
                fig,ax = plt.subplots(dpi=300)
                ax.set_title(i)
       
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    return A,B,C





