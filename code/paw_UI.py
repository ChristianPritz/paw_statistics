#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 14:14:46 2025

@author: Christian Pritz
"""

import cv2
import torch
import numpy as np
import glob, copy,scipy,math,cv2,os,posixpath
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
import matplotlib.pyplot as plt
from .interactive_plot_UI import interactive_plot_UI
from .paw_statistics import paw_statistics
from dataclasses import dataclass
import numpy as np
from ultralytics import YOLO
from enum import IntEnum


def _show_in_foreground(window, parent=None):
    """Raise a newly-created Tk window without keeping it permanently on top."""
    if parent is not None:
        window.transient(parent)

    window.lift()
    try:
        window.attributes("-topmost", True)
    except tk.TclError:
        pass

    def release_topmost():
        if not window.winfo_exists():
            return
        try:
            window.attributes("-topmost", False)
        except tk.TclError:
            pass
        window.lift()
        window.focus_force()

    window.after_idle(release_topmost)


class ImageSequenceExporter:
    def __init__(self,parent, image_dir, metadata, detector_settings,
                 width=300, factor=3, prefix="", paw_stats=None,
                 image_save_dir=None):

        if parent is not None:        
            self.root = tk.Toplevel(parent)
            self._owns_master = False
        else: 
            self._owns_master = True
            self.root = tk.Tk()

        self.image_dir = Path(image_dir)
        self.metadata = metadata
        self.dataframe = pd.DataFrame(columns=list(metadata.keys()) +
                                      ["image_name", "source_image", "crop_index"])
        
        if image_save_dir is None: 
            p,_ = os.path.split(self.image_dir)
        else: 
            p = image_save_dir
 
        self.output_dir = Path(p) # serves as default path
        os.makedirs(self.output_dir, exist_ok=True)
        self.width = width
        self.factor = factor
        self.prefix = prefix
        self.save_name = 'default'
        self.save_name_zip = 'default.zip'
        self.define_file_name()

        # --- Detect whether input is a video file or directory ---
        if self.image_dir.is_file() and self.image_dir.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
            print(f"Video mode activated: {self.image_dir.name}")
            self.is_video = True
            self.video_capture = cv2.VideoCapture(str(self.image_dir))
            self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            self.image_files = list(range(self.total_frames))  # pseudo list for indexing
        else:
            print(f"Image directory mode activated: {self.image_dir}")
            self.is_video = False
            self.image_files, _ = paw_cropper.index_dir(self, directory=str(self.image_dir),
                                                        dataFrame=False)
        print('##############################################################')    
        print(self.image_files)
        print('##############################################################')
        self.export_bbox = None
        self.export_pts = None
        self.export_side = None

        self.current_index = 0
        self.threshold = detector_settings.get('threshold', 0.7)
        self.tolerance = detector_settings.get('padding', 0.5)
        self.counter = {}     # per-image crop counter
        self.detector_settings = detector_settings
        self.crpr = paw_cropper(detector_settings,
                                video=str(self.image_dir) if self.is_video else None,
                                directory=str(self.image_dir if not self.is_video else self.output_dir),
                                output_dir=str(self.output_dir))
        if paw_stats is not None:
 
            self.paw_stats = paw_statistics(None)
 
            if os.path.exists(paw_stats):

                self.paw_stats.load_data_zip(filename = paw_stats)
            else:

                self.paw_stats.load_data_zip()
                
            
            print('----------------------------------------------------------')
            print("- DATA LOADED FROM EXISTING FILE - " + paw_stats)
            print('----------------------------------------------------------')
            
            
        else:
            columns = list(metadata.keys()) + ["image_name", "source_image",
                                               "crop_index", "predicted_side", 
                                               "image_dir","frame_number"]
            self.paw_stats = paw_statistics(None, columns=columns)
        
        
        self.create_ui()
        _show_in_foreground(self.root, parent)
        

    def create_ui(self):
        self.root.title("Image Dir Frame Exporter")
    
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True)
    
        self.video_frame = ttk.Frame(self.main_frame)
        self.video_frame.pack(side="left", fill="both", expand=True)
    
        # --- metadata frame using grid so we can do col1/col2 ---
        self.metadata_frame = ttk.Frame(self.main_frame)
        self.metadata_frame.pack(side="right", fill="y")
        
        # Outer border frame that spans top→bottom
        self.metadata_border = tk.Frame(
            self.metadata_frame,
            highlightbackground="black",
            highlightthickness=2,
            bd=0
        )
        self.metadata_border.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Inner 2-column layout
        self.metadata_border.columnconfigure(0, weight=1)
        self.metadata_border.columnconfigure(1, weight=1)
        
        # Two subframes = two columns
        self.meta_col1 = ttk.Frame(self.metadata_border)   # metadata fields
        self.meta_col2 = ttk.Frame(self.metadata_border)   # control widgets
        
        self.meta_col1.grid(row=0, column=0, sticky="nsw", padx=5, pady=5)
        self.meta_col2.grid(row=0, column=1, sticky="nse", padx=5, pady=5)
    
        # ---------------- VIDEO CANVAS ----------------
        self.canvas = tk.Canvas(self.video_frame, width=self.width,
                                height=self.width, bg="black")
        self.canvas.pack(padx=5, pady=5)
    
        # ---------------- NAVIGATION -------------------
        nav_frame = ttk.Frame(self.video_frame)
        nav_frame.pack()
    
        ttk.Button(nav_frame, text="<< Prev",
                   command=self.prev_frame).pack(side="left")
        ttk.Button(nav_frame, text="Next >>",
                   command=self.next_frame).pack(side="left")
    
        # Frame navigation bar (bottom)
        nav_frame_bottom = ttk.Frame(self.video_frame)
        nav_frame_bottom.pack(fill="x", pady=(5, 10))
    
        ttk.Label(nav_frame_bottom, text="Frame:").pack(side="left", padx=5)
    
        self.frame_scroll = ttk.Scale(
            nav_frame_bottom,
            from_=0,
            to=len(self.image_files) - 1,
            orient="horizontal",
            command=self.on_frame_scroll
        )
        self.frame_scroll.pack(side="left", fill="x", expand=True, padx=5)
    
        self.frame_index_var = tk.StringVar(value=str(self.current_index))
        self.frame_entry = ttk.Entry(nav_frame_bottom,
                                     textvariable=self.frame_index_var, width=6)
        self.frame_entry.pack(side="right", padx=5)
        self.frame_entry.bind("<Return>", self.on_frame_entry)
    
        filename_frame = ttk.Frame(self.video_frame)
        filename_frame.pack(fill="x", pady=(3, 10))
    
        self.filename_label = ttk.Label(filename_frame,
                                        text="", font=("TkDefaultFont", 10, "bold"),
                                        anchor="center")
        self.filename_label.pack(expand=True)
    
        self.export_button = ttk.Button(nav_frame, text="Export cropped",
                                        command=self.export_segmented_paw)
        self.export_button.pack(side="left")
    
        # ---------------- COLUMN 2 CONTENT ----------------
        ttk.Label(self.meta_col2, text="Select Prediction:").pack(pady=5)
        self.prediction_scale = ttk.Scale(self.meta_col2, from_=0, to=0,
                                          orient="horizontal",
                                          command=self.on_prediction_change)
        self.prediction_scale.pack(pady=5, fill="x")
    
        ttk.Label(self.meta_col2, text="Detection Threshold (0–1):").pack(pady=2)
        self.threshold_entry = ttk.Entry(self.meta_col2)
        self.threshold_entry.insert(0, str(self.threshold))
        self.threshold_entry.pack(pady=2)
        self.threshold_entry.bind("<Return>", self.on_threshold_change)
    
        ttk.Label(self.meta_col2, text="Crop Tolerance (0–1 or inf):").pack(pady=2)
        self.tolerance_entry = ttk.Entry(self.meta_col2)
        self.tolerance_entry.insert(0, str(self.tolerance))
        self.tolerance_entry.pack(pady=2)
    
        # ---------------- CONTROL BUTTON FRAME ----------------
        button_frame = ttk.Frame(self.meta_col2)
        button_frame.pack(fill="x", pady=10)
        
        # Make buttons same width
        BTN_WIDTH = 22
        
        self.btn_fix = ttk.Button(button_frame, text="Fix predictions",
                                  command=self.fix_predictions, width=BTN_WIDTH)
        self.btn_fix.pack(fill="x", pady=3)
        
        self.btn_manual = ttk.Button(button_frame, text="Place points manually",
                                     command=self.place_keypoints, width=BTN_WIDTH)
        self.btn_manual.pack(fill="x", pady=3)

        self.btn_add_roi = ttk.Button(button_frame, text="Add ROI",
                                      command=self.add_roi, width=BTN_WIDTH)
        self.btn_add_roi.pack(fill="x", pady=3)
        
        self.btn_filename = ttk.Button(button_frame, text="Define file name",
                                       command=self.define_file_name, width=BTN_WIDTH)
        self.btn_filename.pack(fill="x", pady=3)
        
        # NEW BUTTON
        self.btn_crop = ttk.Button(button_frame, text="Crop image",
                                   command=self.crop_image_mode, width=BTN_WIDTH)
        self.btn_crop.pack(fill="x", pady=3)
        
        self.btn_save_exit = ttk.Button(button_frame, text="Save & Exit",
                                        command=self.save_and_exit, width=BTN_WIDTH)
        self.btn_save_exit.pack(fill="x", pady=10)
            
        # ---------------- COLUMN 1: METADATA FIELDS ----------------
        ttk.Label(self.meta_col1, text="Metadata:",
                  font=("TkDefaultFont", 10, "bold")).pack(pady=5)
    
        self.metadata_inputs = {}
        for key, value in self.metadata.items():
            ttk.Label(self.meta_col1, text=key).pack(anchor="w")
    
            if isinstance(value, list):
                cb = ttk.Combobox(self.meta_col1, values=value)
                cb.set(value[0])
                cb.pack(fill="x", pady=1)
                self.metadata_inputs[key] = cb
            else:
                ent = ttk.Entry(self.meta_col1)
                ent.insert(0, value)
                ent.pack(fill="x", pady=1)
                self.metadata_inputs[key] = ent
    
        # Init
        self.update_frame()
        if self._owns_master:
            self.root.mainloop()
    
    def update_filename_label(self):
        """Display only the filename (no path) depending on image/video mode."""
        if self.is_video:
            _,vid_name = os.path.split(self.image_dir)
            fname = f"{vid_name} frame_{self.current_index:05d}"
        else:
            fname = Path(self.image_files[self.current_index]).name
    
        self.filename_label.config(text=fname)

    def on_threshold_change(self, event=None):
        """Update the detection threshold and refresh frame."""
        try:
            val = float(self.threshold_entry.get())
            if 0.0 <= val <= 1.0:
                self.threshold = val
                # Update detector threshold
                self.crpr.detector.threshold = self.threshold
                self.update_frame()
            else:
                messagebox.showwarning("Invalid Input", "Enter a float between 0 and 1.", parent=self.root)
                self.threshold_entry.delete(0, tk.END)
                self.threshold_entry.insert(0, str(self.threshold))
        except ValueError:
            messagebox.showwarning("Invalid Input", "Enter a valid float between 0 and 1.", parent=self.root)
            self.threshold_entry.delete(0, tk.END)
            self.threshold_entry.insert(0, str(self.threshold))
        
    def on_prediction_change(self, value):
        """Triggered when the prediction selection scale changes."""
        try:
            self.current_box_selection = int(float(value))
        except ValueError:
            self.current_box_selection = 0
        self.redraw_bboxes()

    def _canvas_to_frame(self, x, y):
        """Convert displayed canvas coordinates to original-frame coordinates."""
        scale_y, scale_x = self.scaler
        height, width = self.current_frame.shape[:2]
        frame_x = float(np.clip(x * scale_x, 0, width))
        frame_y = float(np.clip(y * scale_y, 0, height))
        return frame_x, frame_y

    def _refresh_selected_crop(self, run_specialist=False):
        """Synchronize crop, offsets, and optionally pose output for the selected ROI."""
        sel = self.current_box_selection
        height, width = self.current_frame.shape[:2]
        box = np.asarray(self.current_boxes[sel], dtype=np.float32).copy()
        box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
        x1, y1, x2, y2 = np.rint(box).astype(int)
        if x2 <= x1 or y2 <= y1:
            return False

        box = np.array([x1, y1, x2, y2], dtype=np.float32)
        crop = self.current_frame[y1:y2, x1:x2].copy()
        self.current_boxes[sel] = box
        self.current_paw_images[sel] = crop
        self.offset_x[sel] = x1
        self.offset_y[sel] = y1

        if run_specialist and np.isfinite(self.current_classes[sel]):
            cls = int(self.current_classes[sel])
            detector = self.crpr.detector
            mirrored = detector._needs_flip(cls)
            specialist_input = detector._mirror_image(crop) if mirrored else crop
            output = detector._run_specialist(specialist_input, cls)
            if output is not None:
                pose_box, points, _ = output
                if mirrored:
                    pose_box = detector._mirror_xyxy(pose_box, crop.shape[1])
                    points = detector._mirror_keypoints(points, crop.shape[1])
                offset = box[:2]
                self.current_orig_boxes[sel] = detector._translate_xyxy(pose_box, offset)
                self.current_pts[sel] = detector._translate_keypoints(points, offset)
        return True

    def _bind_roi_handle(self, item, corner):
        self.canvas.tag_bind(item, "<ButtonPress-1>",
                             lambda event, c=corner: self._start_roi_resize(event, c))

    def _start_roi_resize(self, event, corner):
        self._resizing_corner = corner
        # Canvas-level bindings remain active even though redraw replaces the
        # handle item currently under the pointer.
        self.canvas.bind("<B1-Motion>", self._drag_roi_handle)
        self.canvas.bind("<ButtonRelease-1>", self._finish_roi_resize)
        return "break"

    def _drag_roi_handle(self, event):
        if not hasattr(self, "_resizing_corner"):
            return "break"
        sel = self.current_box_selection
        box = np.asarray(self.current_boxes[sel], dtype=np.float32).copy()
        x, y = self._canvas_to_frame(event.x, event.y)
        minimum = 2.0
        if self._resizing_corner == "ul":
            box[0] = min(x, box[2] - minimum)
            box[1] = min(y, box[3] - minimum)
        else:
            box[2] = max(x, box[0] + minimum)
            box[3] = max(y, box[1] + minimum)
        self.current_boxes[sel] = box
        self._refresh_selected_crop(run_specialist=False)
        self.redraw_bboxes()
        return "break"

    def _finish_roi_resize(self, event):
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        if hasattr(self, "_resizing_corner"):
            del self._resizing_corner
        if not self._refresh_selected_crop(run_specialist=True):
            messagebox.showwarning("Invalid ROI", "The resized ROI is empty.", parent=self.root)
        self.redraw_bboxes()
        return "break"

    def redraw_bboxes(self):
        """Redraw image and bounding boxes according to current selection."""
        # Draw image
        #img_path = self.image_files[self.current_index]
        #frame = cv2.imread(str(img_path))
        if hasattr(self,"current_frame"):
            self.canvas.delete("all")
            self.resize_image(self.current_frame)

        # Draw boxes
            for idx, box in enumerate(self.current_boxes):
                x1, y1, x2, y2 = box
                x1 /= self.scaler[1]; x2 /= self.scaler[1]
                y1 /= self.scaler[0]; y2 /= self.scaler[0]
                color = "red" if idx == self.current_box_selection else "blue"
                self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
                if idx == self.current_box_selection:
                    radius = 6
                    ul = self.canvas.create_oval(
                        x1 - radius, y1 - radius, x1 + radius, y1 + radius,
                        fill="yellow", outline="black", width=1,
                    )
                    lr = self.canvas.create_oval(
                        x2 - radius, y2 - radius, x2 + radius, y2 + radius,
                        fill="yellow", outline="black", width=1,
                    )
                    self._bind_roi_handle(ul, "ul")
                    self._bind_roi_handle(lr, "lr")
    
    def resize_image(self,frame):
        
        old_h, old_w = frame.shape[:2]
        if old_h < old_w:
            self.height = round(self.width * old_h / old_w)
            self.display_width = self.width
            self.scaler = np.array([old_h / self.height, old_w / self.width])
        
            # Draw placeholder image
            frame_rgb = cv2.cvtColor(cv2.resize(frame, (self.width, self.height)), cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
            self.canvas.image = imgtk
        else: 
            self.height = self.width
            width = round(self.width * old_w/old_h)
            self.display_width = width
            self.scaler = np.array([old_h / self.height, old_w / width])
        
            # Draw placeholder image
            frame_rgb = cv2.cvtColor(cv2.resize(frame, (width, self.height)), cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
            self.canvas.image = imgtk
        
    
    def update_frame(self,override=None):

        
        # --- Load image or video frame depending on mode ---
        if override is None:
            if self.is_video:
        
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_index)
                ret, frame = self.video_capture.read()
                if not ret:
                    messagebox.showerror("Error", f"Cannot read frame {self.current_index}", parent=self.root)
                    return
                img_name = f"frame_{self.current_index:05d}.png"
            else:
                
                print("These are the image files:",self.image_files)
                img_path = self.image_files[self.current_index]
                frame = cv2.imread(str(img_path))
                img_name = Path(img_path).name
                if frame is None:
                    messagebox.showerror("Error", f"Cannot load {img_path}", parent=self.root)
                    return
            self.current_frame = frame
        else:
            frame = override
        
        self.update_nav_controls()
        self.update_filename_label()
        
        # --- Handle tolerance input ---
        tol = self.tolerance_entry.get().lower()
        try:
            self.tolerance = float(tol) if tol != 'inf' else float('inf')
        except ValueError:
            messagebox.showerror("Error", "Invalid tolerance (enter float or inf)", parent=self.root)
            return

        # --- Run detection ---
        if override is None:
            if self.is_video:
                pts, bboxes, crops, clss, orig_bboxes = self.crpr.live_cropper_img(
                    frame, tolerance=self.tolerance
                ) #if hasattr(self.crpr, 'live_cropper_img_from_frame') else self.crpr.live_cropper_img(
                    #str(img_name), tolerance=self.tolerance
                #)
            else:
                pts, bboxes, crops, clss, orig_bboxes = self.crpr.live_cropper_img(
                    str(img_path), tolerance=self.tolerance, 
                    threshold = self.threshold
                )
        else:
            pts, bboxes, crops, clss, orig_bboxes = self.crpr.live_cropper_img(
                frame, tolerance=self.tolerance, 
                threshold = self.threshold
            )
        

        
        if bboxes is None or len(bboxes) == 0: # incase of no predictions.... 
            self.current_pts = []
            self.current_boxes = []
            self.current_orig_boxes = []
            self.current_classes = []
            self.current_paw_images = []
            self.offset_x = []
            self.offset_y = []
            self.canvas.delete("all")
            
            self.resize_image(frame)
        
            # Overlay text
            self.canvas.create_text(
                self.width // 2, self.height // 2,
                text="No predictions found",
                fill="yellow", font=("Arial", 20, "bold")
            )
        
            # Disable dependent UI controls
            self.prediction_scale.configure(to=0)
            self.prediction_scale.set(0)
            self.prediction_scale.state(["disabled"])
            self.export_button.state(["disabled"])
            return
        
        # --- successful preditions enable the controls ----------
        self.prediction_scale.state(["!disabled"])
        self.export_button.state(["!disabled"])
        # --- Store predictions for later editing/export ---
        self.current_pts = pts
        self.current_boxes = bboxes
        self.current_orig_boxes = orig_bboxes
        self.current_classes = clss
        self.current_paw_images = crops
        self.current_box_selection = 0
        self.offset_x = [b[0] for b in bboxes] # ELIMINATE DEPRECATED
        self.offset_y = [b[1] for b in bboxes] # ELIMINATE DEPRECATED

        # --- Adjust UI scaling ---
        if len(bboxes) > 0:
            self.prediction_scale.configure(to=len(bboxes) - 1)
        else:
            self.prediction_scale.configure(to=0)
        self.prediction_scale.set(0)

        self.redraw_bboxes()

    def correct_prediction(self,fname):
        img = self.current_paw_images[self.current_box_selection]
        pts_in = np.asarray(self.current_pts[self.current_box_selection]).copy()
        bbox_in = np.asarray(self.current_orig_boxes[self.current_box_selection]).copy()
        class_out = self.current_classes[self.current_box_selection]
        
        bbox_in[0] = bbox_in[0]-self.offset_x[self.current_box_selection]
        bbox_in[1] = bbox_in[1]-self.offset_y[self.current_box_selection]
        bbox_in[2] = bbox_in[2]-self.offset_x[self.current_box_selection]
        bbox_in[3] = bbox_in[3]-self.offset_y[self.current_box_selection]
        
        pts_in = pts_in - np.tile([self.offset_x[self.current_box_selection],
                                   self.offset_y[self.current_box_selection],0],
                                  (pts_in.shape[0],1))


        app = interactive_plot_UI(self.root, img, pts_in[:, [0, 1]], bbox_in,
                                  self.detector_settings["connect_logic"],
                                  self.detector_settings["colors_ui"],
                                  title='Inspect and correct the predictions',
                                  window_size=[1000, 1000])

   
        pts_out,bxs_out = app.return_data()
        # construct metadata here 


        fname = fname[0:-4] + '.mat'
        dict_out = self.dataframe.iloc[-1].to_dict()
        dict_out["height"] = img.shape[0]
        dict_out["width"] = img.shape[1]
        dict_out["visibility"] = np.ones((15,1)) #np.tile([2],(15,1)) 
        dict_out["truncated"] = 0
        dict_out["useful"] = 'yes'
        dict_out["remark"] = 'empty'
        dict_out["animal_ID"] = dict_out['animal_id'] #lazy work around
        # updating current bbox and pts

        self.export_bbox = bxs_out
        self.export_pts = pts_out
        self.export_side = class_out
        
        self.save_to_mat(pts_out, bxs_out, [dict_out], fname)
    
    def fix_predictions(self):
        """Rotate current image in 10° increments and reload UI state with best rotation."""
       
    
        frame = self.current_frame.copy()
        best_sum_prob = -1
        best_result = None
    
        # Rotate 0–180° in 30° steps
        random_rots = [10,11,12,13,14,15]
        rot = np.random.choice(random_rots,size=1)
        for angle in range(0, 181, int(rot[0])):
            rot_mat = cv2.getRotationMatrix2D(
                (frame.shape[1] // 2, frame.shape[0] // 2), angle, 1.0
            )
            rotated = cv2.warpAffine(frame, rot_mat, (frame.shape[1], frame.shape[0]))
    
            pts, bboxes, crops, clss, orig_bboxes = self.crpr.live_cropper_img(
                rotated, tolerance=self.tolerance, threshold=self.threshold
            )
    
            if bboxes is None or len(bboxes) == 0:
                continue
    
            scores = getattr(self.crpr, "last_scores", [])
            prob_sum = float(np.sum(scores)) if len(scores) else float(len(bboxes))
    
            if prob_sum > best_sum_prob:
                best_sum_prob = prob_sum
                best_result = (rotated, pts, bboxes, crops, clss, orig_bboxes)
    
        if best_result is None:
            messagebox.showinfo("Fix predictions", "No prediction found for any rotation.", parent=self.root)
            return
    
        #Unpack best result

        rotated, pts, bboxes, crops, clss, orig_bboxes = best_result
    
        #Replace current image and prediction results
        self.current_frame = rotated
        self.current_pts = pts
        self.current_boxes = bboxes
        self.current_orig_boxes = orig_bboxes
        self.current_classes = clss
        self.current_paw_images = crops
        self.current_box_selection = 0
    
        #IMPORTANT: Synchronize UI elements and run full update pipeline
        self.prediction_scale.configure(to=len(bboxes) - 1)
        self.prediction_scale.set(0)
        self.prediction_scale.state(["!disabled"])
        self.export_button.state(["!disabled"])
    
        #updating the frame
        self.update_frame(override=rotated)

    # def save_to_mat(self,pts, bxs, data_dict, image_name):
    #     """
    #     Saves Python objects into a .mat file in a specified cell array format.
    
    #     Parameters:
    #         pts (np.ndarray): Array to be placed in cell (1,2)
    #         bxs (np.ndarray): Array to be placed in cell (2,2)
    #         data_dict (dict): Dictionary containing 'genotype', 'paw_posture', and 'side'
    #         filename (str): Name of the output .mat file
    #     """
        
    #     filename = image_name[0:-4] + '.mat'
    #     cell_array = [["points", '', '', '', '', '', ''],  # 1
    #                   ["rois", '', '', '', '', '', ''],  # 2
    #                   ["visability", '', '', '', '', '', ''],  # 3
    #                   ["truncated", '', '', '', '', '', ''],  # 4
    #                   ["height", '', '', '', '', '', ''],  # 5
    #                   ["width", '', '', '', '', '', ''],  # 6
    #                   ["genotype", '', '', '', '', '', ''],  # 7
    #                   ["gender", '', '', '', '', '', ''],  # 8
    #                   ["side", '', '', '', '', '', ''],  # 9
    #                   ["treatment", '', '', '', '', '', ''],  # 10
    #                   ["paw_posture", '', '', '', '', '', ''],  # 11
    #                   ["pain_status", '', '', '', '', '', ''],  # 12
    #                   ["useful", '', '', '', '', '', ''],  # 13
    #                   ["remark", '', '', '', '', '', ''],  # 14
    #                   ["animal_ID", '','', '', '', '', '']]  # 15
        
    #     for i in range(len(data_dict)):
    #     # Create an empty cell array (MATLAB cell array is represented as a list of lists in Python)
    
    
    #         if len(pts.shape) == 3:
    #             cell_array[0][i+1] = pts[i]        # (1,2) -> cell_array[0][1]
    #             cell_array[1][i+1] = bxs[i]        # (2,2) -> cell_array[1][1]
    #         else:
    #             cell_array[0][i+1] = pts        # (1,2) -> cell_array[0][1]
    #             cell_array[1][i+1] = bxs        # (2,2) -> cell_array[1][1]
                
    #         cell_array[2][i+1] = data_dict[i]["visibility"]    # (3,2) -> cell_array[2][1]
    #         cell_array[3][i+1] = data_dict[i]["truncated"] # (4,2) -> cell_array[3][1]
    #         cell_array[4][i+1] = data_dict[i]["height"]        # (5,2) -> cell_array[4][1]
    #         cell_array[5][i+1] = data_dict[i]["width"]
    #         cell_array[6][i+1] = data_dict[i]["genotype"]
    #         cell_array[7][i+1] = data_dict[i]["gender"]
    #         cell_array[8][i+1] = data_dict[i]["side"]
    #         cell_array[9][i+1] = data_dict[i]["treatment"]
    #         cell_array[10][i+1] = data_dict[i]["paw_posture"]
    #         cell_array[11][i+1] = data_dict[i]["pain_status"]
    #         cell_array[12][i+1] = data_dict[i]["useful"]
    #         cell_array[13][i+1] = data_dict[i]["remark"]
    #         cell_array[14][i+1] = data_dict[i]["animal_ID"]
    
            
    #     for j in cell_array:
    #         for i in np.arange(len(data_dict)+1,7)[::-1]:
    #             j.pop(i)

    #     mat_dict = {'varList': cell_array}
    
    #     # Save to .mat file
    #     scipy.io.savemat(filename, mat_dict)

    def save_to_mat(self, pts, bxs, data_dict, image_name):
        """
        Save annotations to MATLAB varList format.
    
        Parameters
        ----------
        pts : ndarray
            Keypoints (Nx2) or (M,N,2)
        bxs : ndarray
            Bounding boxes in XYWH format.
        data_dict : list[dict]
            Annotation dictionaries.
        image_name : str
            Output image filename.
        """
    
        import numpy as np
        import scipy.io
    
        filename = image_name[:-4] + ".mat"
    
        # ------------------------------------------------------------------
        # varList layout
        # ------------------------------------------------------------------
        cell_array = [
            ["points", "", "", "", "", "", ""],        # 1
            ["rois", "", "", "", "", "", ""],          # 2
            ["visability", "", "", "", "", "", ""],    # 3
            ["truncated", "", "", "", "", "", ""],     # 4
            ["height", "", "", "", "", "", ""],        # 5
            ["width", "", "", "", "", "", ""],         # 6
            ["genotype", "", "", "", "", "", ""],      # 7
            ["gender", "", "", "", "", "", ""],        # 8
            ["side", "", "", "", "", "", ""],          # 9
            ["treatment", "", "", "", "", "", ""],     # 10
            ["paw_posture", "", "", "", "", "", ""],   # 11
            ["pain_status", "", "", "", "", "", ""],   # 12
            ["useful", "", "", "", "", "", ""],        # 13
            ["remark", "", "", "", "", "", ""],        # 14
            ["animal_ID", "", "", "", "", "", ""],     # 15
            ["ant_or_post", "", "", "", "", "", ""],   # 16
        ]
    
        # --------------------------------------------------------------
        # Helper
        # --------------------------------------------------------------
        def get_value(d, key, default="UKN"):
            val = d.get(key, default)
            if val is None or val == "":
                return default
            return val
    
        # --------------------------------------------------------------
        # Fill annotations
        # --------------------------------------------------------------
        for i, ann in enumerate(data_dict):
    
            # ----- points / bbox -----
            if pts.ndim == 3:
                kp = pts[i]
                box = bxs[i]
            else:
                kp = pts
                box = bxs
    
            cell_array[0][i + 1] = kp
            cell_array[1][i + 1] = box
    
            # ----------------------------------------------------------
            # Default visibility = ones
            # ----------------------------------------------------------
            visibility = np.ones((kp.shape[0], 1), dtype=np.uint32)
    
            # ----------------------------------------------------------
            # Image dimensions
            # ----------------------------------------------------------
            height = ann.get("height", "UKN")
            width = ann.get("width", "UKN")
    
            # ----------------------------------------------------------
            # Determine anterior/posterior from class id
            #
            # classes:
            # 0 = post
            # 1 = post
            # 2 = ant
            # 3 = ant
            # ----------------------------------------------------------
            cls = ann.get("class", ann.get("class_id", None))
    
            if cls in [0, 1]:
                ant_post = "post"
            elif cls in [2, 3]:
                ant_post = "ant"
            else:
                ant_post = "UKN"
    
            # ----------------------------------------------------------
            # Fill varList
            # ----------------------------------------------------------
            cell_array[2][i + 1] = visibility
            cell_array[3][i + 1] = 0                    # always
            cell_array[4][i + 1] = height
            cell_array[5][i + 1] = width
    
            cell_array[6][i + 1] = get_value(ann, "genotype")
            cell_array[7][i + 1] = get_value(ann, "gender")
            cell_array[8][i + 1] = get_value(ann, "side")
            cell_array[9][i + 1] = get_value(ann, "treatment")
            cell_array[10][i + 1] = get_value(ann, "paw_posture")
            cell_array[11][i + 1] = get_value(ann, "pain_status")
            cell_array[12][i + 1] = get_value(ann, "useful")
            cell_array[13][i + 1] = get_value(ann, "remark")
            cell_array[14][i + 1] = get_value(ann, "animal_ID")
            cell_array[15][i + 1] = ant_post
    
        # --------------------------------------------------------------
        # Remove unused columns
        # --------------------------------------------------------------
        for row in cell_array:
            while len(row) > len(data_dict) + 1:
                row.pop()
    
        scipy.io.savemat(filename, {"varList": cell_array})

    def prev_frame(self):
        self.current_index = max(0, self.current_index - 1)
        self.update_frame()
        self.update_nav_controls()

    def next_frame(self):
        self.current_index = min(len(self.image_files) - 1, self.current_index + 1)
        self.update_frame()
        self.update_nav_controls()
    def on_frame_scroll(self, value):
        """Handle scrollbar movement."""
        try:
            idx = int(float(value))
            if idx != self.current_index:
                self.current_index = idx
                self.update_frame()
        except ValueError:
            pass
    def update_nav_controls(self):
        """Sync scrollbar and entry with current frame index."""
        self.frame_scroll.set(self.current_index)
        self.frame_index_var.set(str(self.current_index))
        
    def on_frame_entry(self, event=None):
        """Handle manual frame index input."""
        try:
            idx = int(self.frame_index_var.get())
            if 0 <= idx < len(self.image_files):
                self.current_index = idx
                self.update_frame()
            else:
                messagebox.showwarning("Invalid Index", f"Enter a number between 0 and {len(self.image_files)-1}", parent=self.root)
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid integer.", parent=self.root)

    def set_prediction_controls_state(self, state: str):
        self.prediction_scale.state([state])
        self.export_button.state([state])
    

    
    def export_segmented_paw(self):
        if not hasattr(self, "current_boxes") or len(self.current_boxes) == 0:
            messagebox.showwarning("No predictions", "No predictions available for this frame.", parent=self.root)
            return
   
        # -------------------------------------------------------------
        #  DUPLICATE ENTRY CHECK
        # -------------------------------------------------------------
        # Read current metadata values from UI
        current_meta = {}
        for key, widget in self.metadata_inputs.items():
            val = widget.get()
            current_meta[key] = val
         
        # Only compare metadata columns that the user can change
        columns_to_check = list(self.metadata_inputs.keys())
         
        # Check against existing entries in the label database
        if hasattr(self.paw_stats, "label_db") and len(self.paw_stats.label_db) > 0:
         
            # Extract only the relevant metadata columns
            db_subset = self.paw_stats.label_db[columns_to_check]
         
            # Convert the row values into a comparable tuple
            current_tuple = tuple(current_meta[col] for col in columns_to_check)
         
            # Generate tuples for every row
            db_tuples = [tuple(row[col] for col in columns_to_check)
                         for _, row in db_subset.iterrows()]
         
            if current_tuple in db_tuples:
                messagebox.showinfo(
                    "Duplicate Entry",
                    "!! - This metadata combination already exists in the database!!\n"
                    "Please check to avoid creating double entries.",
                    parent=self.root,
                )
                return  # NO EXPROT - DUPLICATE
        # -------------------------------------------------------------
        #        END DUPLICATE CHECK — EXPORT CONTINUES AS BEFORE
        # -------------------------------------------------------------
        
        
        
        

        p = self.image_files[self.current_index]

        if self.is_video:
            base,name = os.path.split(self.image_dir)
            ext = '.png'
 
            
        else:
            base,name = os.path.split(p)
            ext = name[-4:]
            
        name = name[0:-4]
        base = self.output_dir
        sel = self.current_box_selection
        count = self.counter.get(base, 0)
        self.counter[base] = count + 1

        
        out_name = f"{name}_crop{count}{ext}"
        out_path = self.output_dir / out_name
        cv2.imwrite(str(out_path), self.current_paw_images[sel])
        

        md = {k: w.get() for k, w in self.metadata_inputs.items()}
  
        md.update({"image_name": out_name, "source_image": name, 
                    "crop_index": sel,"predicted_side":self.current_classes[sel],
                    "image_dir":self.output_dir,
                    "frame_number":self.current_index})
        #This is redundant with paw_stats.label_db
        self.dataframe = pd.concat([self.dataframe, pd.DataFrame([md])], ignore_index=True)
        self.correct_prediction(str(out_path))
        md["predicted_side"] = self.export_side
        
        # update side fromt he prediction
        
        
        # add the paw to the paw_statistics file..... #
       
        self.export_pts = np.hstack((self.export_pts,np.ones((self.export_pts.shape[0],1))))
        pts = self.export_pts.reshape((1,self.export_pts.shape[0],self.export_pts.shape[1]))
   
        bxs = self.export_bbox.reshape((1,self.export_bbox.shape[0],1))

        self.paw_stats.add_data(pts,bxs)

            # Append the dictionary to the DataFrame

        self.paw_stats.label_db = pd.concat([self.paw_stats.label_db,
                                              pd.DataFrame([md])],
                                            ignore_index=True)
        
        self.export_bbox,self.export_pts,self.export_side = None,None,None
        # Saving the data on the hard disk 
        
        self.save_data()
    
    def crop_image_mode(self):
        """
        Activates 2-click ROI tool to crop the current frame.
        After second click:
            - ROI is mapped to original resolution
            - Image is cropped
            - update_frame(override=crop) is called
        """
    
        self.manual_roi = None
        self.manual_roi_start = None
        self.roi_rect = None
    
        def on_mouse_click(event):
    
            # FIRST CLICK
            if self.manual_roi_start is None:
                self.manual_roi_start = (event.x, event.y)
    
                if self.roi_rect is not None:
                    self.canvas.delete(self.roi_rect)
                    self.roi_rect = None
    
            # SECOND CLICK
            else:
                x0, y0 = self.manual_roi_start
                x1, y1 = event.x, event.y
    
                if x1 < x0:
                    x0, x1 = x1, x0
                if y1 < y0:
                    y0, y1 = y1, y0
    
                # Draw final rectangle
                if self.roi_rect is not None:
                    self.canvas.delete(self.roi_rect)
    
                self.roi_rect = self.canvas.create_rectangle(
                    x0, y0, x1, y1, outline="orange", width=2
                )
    
                # Map ROI to original image coordinates
                scale_y, scale_x = self.scaler
    
                X0 = int(x0 * scale_x)
                X1 = int(x1 * scale_x)
                Y0 = int(y0 * scale_y)
                Y1 = int(y1 * scale_y)
    
                frame = self.current_frame
                H, W = frame.shape[:2]
    
                X0 = max(0, min(W - 1, X0))
                X1 = max(0, min(W - 1, X1))
                Y0 = max(0, min(H - 1, Y0))
                Y1 = max(0, min(H - 1, Y1))
    
                if X1 <= X0 or Y1 <= Y0:
                    messagebox.showwarning("Invalid ROI", "ROI outside bounds.", parent=self.root)
                    return
    
                cropped = frame[Y0:Y1, X0:X1].copy()
    
                # Cleanup bindings
                self.canvas.unbind("<Button-1>")
                self.canvas.unbind("<Motion>")
    
                self.manual_roi_start = None
    
                # IMPORTANT: override current frame
                self.current_frame = cropped
                self.update_frame(override=cropped)
    
        def on_mouse_move(event):
            if self.manual_roi_start is None:
                return
    
            x0, y0 = self.manual_roi_start
            x1, y1 = event.x, event.y
    
            if self.roi_rect is not None:
                self.canvas.delete(self.roi_rect)
    
            self.roi_rect = self.canvas.create_rectangle(
                x0, y0, x1, y1, outline="orange", width=2
            )
    
        # Activate bindings
        self.canvas.bind("<Button-1>", on_mouse_click)
        self.canvas.bind("<Motion>", on_mouse_move)
    
        print("Crop mode: click once to start, click again to crop.")
   
    def _draw_roi(self, on_complete, color="lime"):
        """Collect a two-click canvas ROI, then call the requested workflow."""
        self.manual_roi = None
        self.manual_roi_start = None
        self.roi_rect = None

        def on_mouse_click(event):
            if self.manual_roi_start is None:
                self.manual_roi_start = (event.x, event.y)
                if self.roi_rect is not None:
                    self.canvas.delete(self.roi_rect)
                    self.roi_rect = None
            else:
                x0, y0 = self.manual_roi_start
                x1, y1 = event.x, event.y
                self.manual_roi = (x0, y0, x1, y1)
                if self.roi_rect is not None:
                    self.canvas.delete(self.roi_rect)
                self.roi_rect = self.canvas.create_rectangle(
                    x0, y0, x1, y1, outline=color, width=2
                )
                self.manual_roi_start = None
                self.canvas.unbind("<Button-1>")
                self.canvas.unbind("<Motion>")
                on_complete()

        def on_mouse_move(event):
            if self.manual_roi_start is None:
                return
            x0, y0 = self.manual_roi_start
            x1, y1 = event.x, event.y
            if self.roi_rect is not None:
                self.canvas.delete(self.roi_rect)
            self.roi_rect = self.canvas.create_rectangle(
                x0, y0, x1, y1, outline=color, width=2
            )

        self.canvas.bind("<Button-1>", on_mouse_click)
        self.canvas.bind("<Motion>", on_mouse_move)
        print("Draw ROI: Click once for start, click again for end.")

    def place_keypoints(self):
        """Draw an ROI and seed it with placeholder keypoints for manual editing."""
        self._draw_roi(self.prepare_manual_keypoints, color="lime")

    def add_roi(self):
        """Draw an ROI and obtain its keypoints from the selected specialist model."""
        self._draw_roi(self.prepare_model_roi, color="cyan")

    def _choose_paw_class(self):
        labels = {
            "Hind left": PawClass.HIND_LEFT,
            "Hind right": PawClass.HIND_RIGHT,
            "Front left": PawClass.FRONT_LEFT,
            "Front right": PawClass.FRONT_RIGHT,
        }
        popup = tk.Toplevel(self.root)
        popup.title("Select paw type")
        popup.resizable(False, False)
        _show_in_foreground(popup, self.root)
        choice = tk.StringVar(value=next(iter(labels)))
        ttk.Label(popup, text="Specialist model for the new ROI:").pack(
            padx=15, pady=(15, 5)
        )
        ttk.Combobox(
            popup, textvariable=choice, values=list(labels), state="readonly"
        ).pack(fill="x", padx=15)
        result = {"class": None}

        def confirm():
            result["class"] = int(labels[choice.get()])
            popup.destroy()

        ttk.Button(popup, text="Run model", command=confirm).pack(pady=15)
        popup.protocol("WM_DELETE_WINDOW", popup.destroy)
        popup.grab_set()
        self.root.wait_window(popup)
        return result["class"]

    def _manual_roi_crop(self):
        """Return the drawn ROI as an original-frame box and image crop."""
        x0, y0, x1, y1 = self.manual_roi
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        X0, Y0 = self._canvas_to_frame(x0, y0)
        X1, Y1 = self._canvas_to_frame(x1, y1)
        X0, Y0, X1, Y1 = map(int, (X0, Y0, X1, Y1))
        if X1 <= X0 or Y1 <= Y0:
            messagebox.showwarning("Invalid ROI", "ROI is outside bounds.", parent=self.root)
            return None, None
        box = np.array([X0, Y0, X1, Y1], dtype=np.float32)
        return box, self.current_frame[Y0:Y1, X0:X1].copy()

    def prepare_model_roi(self):
        """Run the chosen specialist on a drawn crop and append its prediction."""
        box, crop = self._manual_roi_crop()
        if crop is None:
            return
        cls = self._choose_paw_class()
        if cls is None:
            self.redraw_bboxes()
            return

        detector = self.crpr.detector
        mirrored = detector._needs_flip(cls)
        model_input = detector._mirror_image(crop) if mirrored else crop
        output = detector._run_specialist(model_input, cls)
        if output is None:
            messagebox.showwarning(
                "No prediction", "The specialist model found no paw in that ROI.",
                parent=self.root,
            )
            self.redraw_bboxes()
            return

        pose_box, points, _ = output
        if mirrored:
            pose_box = detector._mirror_xyxy(pose_box, crop.shape[1])
            points = detector._mirror_keypoints(points, crop.shape[1])
        pose_box = detector._translate_xyxy(pose_box, box[:2])
        points = detector._translate_keypoints(points, box[:2])

        # Normalize model outputs to mutable lists before appending the new ROI.
        self.current_boxes = list(self.current_boxes) + [box]
        self.current_paw_images = list(self.current_paw_images) + [crop]
        self.current_orig_boxes = list(self.current_orig_boxes) + [pose_box]
        self.current_pts = list(self.current_pts) + [points]
        self.current_classes = list(self.current_classes) + [cls]
        self.offset_x = list(self.offset_x) + [box[0]]
        self.offset_y = list(self.offset_y) + [box[1]]
        self.current_box_selection = len(self.current_boxes) - 1
        self.prediction_scale.configure(to=self.current_box_selection)
        self.prediction_scale.set(self.current_box_selection)
        self.prediction_scale.state(["!disabled"])
        self.export_button.state(["!disabled"])
        self.redraw_bboxes()
        
    def prepare_manual_keypoints(self):
        #excise image
        x0, y0, x1, y1 = self.manual_roi
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
    
        # rescale using self.scaler = [orig_h/new_h , orig_w/new_w]
        scale_y, scale_x = self.scaler
    
        X0 = int(x0 * scale_x)
        X1 = int(x1 * scale_x)
        Y0 = int(y0 * scale_y)
        Y1 = int(y1 * scale_y)
    
        frame = self.current_frame
        H, W = frame.shape[:2]
    
        X0 = max(0, min(W - 1, X0))
        X1 = max(0, min(W - 1, X1))
        Y0 = max(0, min(H - 1, Y0))
        Y1 = max(0, min(H - 1, Y1))
        if X1 <= X0 or Y1 <= Y0:
            messagebox.showwarning("Invalid ROI", "ROI is outside image bounds.", parent=self.root)
            return
        crop = frame[Y0:Y1, X0:X1].copy()

        # -------------------------------
        # Generate empty predictions
        # -------------------------------
        pts, bxs = self.generate_empty_prediction(crop, ask_for_input=False)
        if len(pts) == 0:
            messagebox.showinfo("Cancelled", "No manual paws to generate.", parent=self.root)
            return
    
        # -------------------------------
        # Store manual prediction buffers so correction can use them
        # -------------------------------
        offset = np.array([X0, Y0], dtype=np.float32)
        global_pts = np.asarray(pts, dtype=np.float32).copy()
        global_pts[:, :, 0] += offset[0]
        global_pts[:, :, 1] += offset[1]
        global_bboxes = np.asarray(bxs, dtype=np.float32).copy()
        global_bboxes[:, [0, 2]] += offset[0]
        global_bboxes[:, [1, 3]] += offset[1]
        roi_box = np.array([X0, Y0, X1, Y1], dtype=np.float32)

        self.manual_pts = global_pts
        self.manual_bboxes = global_bboxes
        self.manual_crops = [crop]
        self.manual_side = "unknown"
    
        # Disable prediction slider so user cannot change selection
        self.prediction_scale.configure(state="disabled")
    
        # Temporarily override correction inputs
        self.current_pts = global_pts
        self.current_boxes = [roi_box]
        self.current_orig_boxes = [global_bboxes[0].copy()]
        self.current_paw_images = [crop]
        self.current_classes = [np.nan]
        self.current_box_selection = 0
        self.offset_x = [X0]
        self.offset_y = [Y0]
        self.prediction_scale.configure(to=0)
        self.prediction_scale.set(0)
        self.export_button.state(["!disabled"])
        self.redraw_bboxes()
       

    def define_file_name(self):
        # Ask for save path (includes filename)
        file_path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Please choose filename for the data (zip file)",
            defaultextension=".zip",
            filetypes=[("ZIP files", "*.zip")],
            initialdir=str(self.output_dir),
            initialfile=self.save_name if self.save_name else "export.zip"
        )
    
        if not file_path:
            return  # user cancelled
    
        # Normalize extension to .zip
        file_path = str(file_path)
        if not file_path.lower().endswith(".zip"):
            file_path += ".zip"
    
        # Split into folder + filename
        folder = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
    
        # Save internally
        self.output_dir = Path(folder)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
        self.save_name_zip = filename
        self.save_name = filename[0:-4]
    
        # messagebox.showinfo(
        #     "Filename Set",
        #     f"Data will be saved to:\n{self.output_dir}\n\nFile: {self.save_name}"
        # )


    def save_data(self):

        csv = self.output_dir / f"{self.save_name}.csv"
        self.dataframe.to_csv(csv, index=False)
        zip_name = self.output_dir / f"{self.save_name_zip}"
        self.paw_stats.all_angles()
        self.paw_stats.save_data_zip(filename=zip_name)
        self.csv_name = csv
        self.zip_name = zip_name

    def save_and_exit(self):
        tol = self.tolerance_entry.get().lower()
        self.tolerance = float(tol) if tol != 'inf' else float('inf')
        self.save_data()
        #messagebox.showinfo("Saved", f"Metadata saved to {self.csv_name}, and full data to {self.zip_name}")
        self.root.destroy()

    
    def generate_empty_prediction(self,img,ask_for_input=False):
        def ask_usr():
            return simpledialog.askinteger(
                "No paw detected!",
                "Please enter the number of non-predicted paws:",
                parent=self.root,
            )
        
        if ask_for_input:
            user_input = ask_usr()
        else:
            user_input = 1
        if user_input == 0:
            return [],[]
        else:
            raw = np.asarray([[0.5,0.2,1],
                    [0.3,0.45,1],
                    [0.27,0.55,1],
                    [0.42,0.5,1],
                    [0.42,0.65,1],
                    [0.42,0.75,1],
                    [0.5,0.52,1],
                    [0.5,0.68,1],
                    [0.5,0.78,1],
                    [0.58,0.51,1],
                    [0.58,0.66,1],
                    [0.58,0.76,1],
                    [0.62,0.45,1],
                    [0.66,0.62,1],
                    [0.68,0.72,1],])
            

            pts = np.zeros((user_input,15,3))
            bxs = np.zeros((user_input,4))
            for i in range(0,user_input):
                rx = copy.copy(raw)
                rx[:,0] = rx[:,0]*img.shape[1]
                rx[:,1] = rx[:,1]*img.shape[0]
                pts[i,:,:] = rx
                bxs[i,:] = np.asarray([
                    np.min(rx[:,0]),  # x1
                    np.min(rx[:,1]),  # y1
                    np.max(rx[:,0]),  # x2
                    np.max(rx[:,1])   # y2
                ]) 
                
        return pts,bxs    
    
        
    def prepare_dict(self,df,img_size):
        def ensure_column_exists(df, column_name, default_value='not stated'):
            if column_name not in df.columns:
                df[column_name] = default_value

            return df
        

        ensure_column_exists(df, "genotype")
        ensure_column_exists(df, "gender")
        ensure_column_exists(df, "side")
        ensure_column_exists(df, "paw_posture")
        ensure_column_exists(df, "pain_status")
        ensure_column_exists(df, "useful")
        ensure_column_exists(df, "remark")
        ensure_column_exists(df, "animal_ID")
        ensure_column_exists(df, "image_name")
        dict_out = []

        for i in range(0,len(df)):
            new_row = {'genotype': df.genotype.iloc[i],
                       'gender': df.gender.iloc[i],
                       'side': df.side.iloc[i],
                       'treatment': df.treatment.iloc[i],
                       'paw_posture':df.paw_posture.iloc[i],
                       'pain_status':df.pain_status.iloc[i],
                       'useful':df.useful.iloc[i],
                       'remark':df.remark.iloc[i],
                       'animal_ID':df.animal_ID.iloc[i],
                       'image_id': df.image_name.iloc[i],
                       
                       'paw_number':i,}
            new_row['visibility'] = np.ones((15,1),'uint32')
            new_row['truncated'] = 0
            new_row['height'] = img_size[0]
            new_row['width'] = img_size[1]
            dict_out.append(new_row)
            
            # Append the dictionary to the DataFrame
        return dict_out

class paw_cropper:
    def __init__(self, detector_settings, video=None, directory=None, output_dir=None):
        self.detector = paw_detector.from_settings(detector_settings)
        self.directory = directory
        self.output_dir = output_dir or (directory + "/output")
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def index_dir(self=None, directory=None, dataFrame=True):
        directory = directory or "."
        exts = ['jpg','jpeg','png','tif','tiff']
        files = []
        for ext in exts:
            files += glob.glob(os.path.join(directory, f"*.{ext}"))
        if dataFrame:
            names = [Path(f).name for f in files]
            df = pd.DataFrame({"image_name": names})
            return files, df
        return files, None

    def live_cropper_img(self, image, tolerance=0.10,threshold=0.9):
        if isinstance(image,str):
            img = cv2.imread(image)
        else: 
            img = image
        results = self.detector.detect_for_UI(img, threshold=threshold, padding=tolerance)
        self.last_scores = [
            result.detector_score + result.pose_score
            for result in results
        ]
        pts, boxes, imgs, classes, orig_bboxes = [], [], [], [], []
        for result in results:
            pts.append(result.keypoints)
            boxes.append(result.crop_bbox)
            orig_bboxes.append(result.pose_bbox)
            imgs.append(result.crop)
            classes.append(int(result.cls))
        return pts, boxes, imgs, classes, orig_bboxes


class PawClass(IntEnum):

    HIND_LEFT  = 0
    HIND_RIGHT = 1
    FRONT_LEFT = 2
    FRONT_RIGHT = 3

@dataclass(slots=True)
class PawDetection:

    cls: PawClass

    detector_score: float

    pose_score: float

    detector_bbox: np.ndarray

    crop_bbox: np.ndarray

    pose_bbox: np.ndarray

    crop: np.ndarray

    keypoints: np.ndarray

    mirrored: bool


@dataclass(slots=True)
class ObjectDetection:

    cls: PawClass
    score: float
    bbox: np.ndarray

    @property
    def is_left(self):
        return self.cls in (
            PawClass.HIND_LEFT,
            PawClass.FRONT_LEFT
        )

    @property
    def is_right(self):
        return not self.is_left

    @property
    def is_front(self):
        return self.cls in (
            PawClass.FRONT_LEFT,
            PawClass.FRONT_RIGHT
        )

    @property
    def is_hind(self):
        return not self.is_front
    
    
class paw_detector:
    def __init__(
        self,
        detector_model,
        hind_model,
        front_model,
        device="cuda",
        padding=0.5,
        threshold=0.7,
    ):
        self.device = device
        self.threshold = threshold
        self.padding = padding
        self.detector = YOLO(detector_model)
        self.hind_model = YOLO(hind_model)
        self.front_model = YOLO(front_model)

    @classmethod
    def from_settings(cls, settings):
        detector_model = (
            settings.get("detector_model_path")
            or settings.get("object_detector_model_path")
            or settings.get("object_model_path")
            or settings.get("model_path")
        )
        hind_model = (
            settings.get("hind_model_path")
            or settings.get("hind_pose_model_path")
            or settings.get("hind_model")
        )
        front_model = (
            settings.get("front_model_path")
            or settings.get("front_pose_model_path")
            or settings.get("front_model")
        )
        missing = [
            name for name, value in (
                ("detector/object model", detector_model),
                ("hind specialist model", hind_model),
                ("front specialist model", front_model),
            )
            if not value
        ]
        if missing:
            raise ValueError("Missing YOLO model path(s): " + ", ".join(missing))
        return cls(
            detector_model=detector_model,
            hind_model=hind_model,
            front_model=front_model,
            device=settings.get("device", "cuda"),
            padding=settings.get("padding", 0.5),
            threshold=settings.get("threshold", 0.7),
        )

    def _select_model(self, cls):
        cls = PawClass(int(cls))
        if cls in (PawClass.HIND_LEFT, PawClass.HIND_RIGHT):
            return self.hind_model
        return self.front_model

    def _needs_flip(self, cls):
        return PawClass(int(cls)) in (PawClass.HIND_LEFT, PawClass.FRONT_LEFT)

    def _pad_bbox(self, bbox, image_shape, padding=None):
        if padding is None:
            padding = self.padding
        H, W = image_shape[:2]
        if not np.isfinite(padding):
            return np.array([0, 0, W, H], dtype=np.float32)
        x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
        w = x2 - x1
        h = y2 - y1
        x1 -= padding * w
        x2 += padding * w
        y1 -= padding * h
        y2 += padding * h

        x1 = max(0, int(np.floor(x1)))
        y1 = max(0, int(np.floor(y1)))
        x2 = min(W, int(np.ceil(x2)))
        y2 = min(H, int(np.ceil(y2)))
        if x2 <= x1:
            x2 = min(W, x1 + 1)
        if y2 <= y1:
            y2 = min(H, y1 + 1)
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def _crop_image(self, image, crop_bbox):
        x1, y1, x2, y2 = crop_bbox.astype(int)
        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def _mirror_image(image):
        return cv2.flip(image, 1)

    @staticmethod
    def _mirror_xyxy(bbox, width):
        bbox = np.asarray(bbox, dtype=np.float32).copy()
        x1 = bbox[0].copy()
        x2 = bbox[2].copy()
        bbox[0] = width - x2
        bbox[2] = width - x1
        return bbox

    @staticmethod
    def _mirror_keypoints(keypoints, width):
        keypoints = np.asarray(keypoints, dtype=np.float32).copy()
        keypoints[:, 0] = width - keypoints[:, 0]
        return keypoints

    @staticmethod
    def _translate_xyxy(bbox, offset):
        bbox = np.asarray(bbox, dtype=np.float32).copy()
        bbox[[0, 2]] += offset[0]
        bbox[[1, 3]] += offset[1]
        return bbox

    @staticmethod
    def _translate_keypoints(keypoints, offset):
        keypoints = np.asarray(keypoints, dtype=np.float32).copy()
        keypoints[:, 0] += offset[0]
        keypoints[:, 1] += offset[1]
        return keypoints

    @torch.no_grad()
    def _detect_objects(self, image, threshold=None):
        if threshold is None:
            threshold = self.threshold

        results = self.detector.predict(
            source=image,
            verbose=False,
            conf=threshold,
            device=str(self.device),
        )
        detections = []
        if len(results) == 0:
            return detections

        r = results[0]
        if r.boxes is None:
            return detections

        for box in r.boxes:
            cls_id = int(box.cls.item())
            if cls_id not in [int(c) for c in PawClass]:
                continue
            detections.append(
                ObjectDetection(
                    cls=PawClass(cls_id),
                    score=float(box.conf.item()),
                    bbox=box.xyxy[0].detach().cpu().numpy().astype(np.float32),
                )
            )

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections

    def _run_specialist(self, crop, cls):
        model = self._select_model(cls)
        results = model.predict(
            source=crop,
            verbose=False,
            device=str(self.device),
        )
        if len(results) == 0:
            return None

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            return None

        scores = r.boxes.conf.detach().cpu().numpy()
        best_idx = int(np.argmax(scores))
        pose_score = float(scores[best_idx])
        pose_bbox = r.boxes.xyxy[best_idx].detach().cpu().numpy().astype(np.float32)

        if hasattr(r.keypoints, "data") and r.keypoints.data is not None:
            keypoints = r.keypoints.data[best_idx].detach().cpu().numpy().astype(np.float32)
        else:
            xy = r.keypoints.xy[best_idx].detach().cpu().numpy().astype(np.float32)
            keypoints = np.concatenate(
                [xy, np.ones((xy.shape[0], 1), dtype=np.float32)],
                axis=1,
            )
        if keypoints.shape[1] == 2:
            keypoints = np.concatenate(
                [keypoints, np.ones((keypoints.shape[0], 1), dtype=np.float32)],
                axis=1,
            )
        return pose_bbox, keypoints[:, :3], pose_score

    @torch.no_grad()
    def detect(self, img_bgr, threshold=None):
        return self.detect_for_UI(img_bgr, threshold=threshold)

    @torch.no_grad()
    def detect_for_UI(self, image, threshold=None, padding=None):
        detections = self._detect_objects(image, threshold=threshold)
        results = []

        for det in detections:
            crop_bbox = self._pad_bbox(det.bbox, image.shape, padding=padding)
            crop = self._crop_image(image, crop_bbox)
            if crop.size == 0:
                continue

            mirrored = self._needs_flip(det.cls)
            specialist_input = self._mirror_image(crop) if mirrored else crop
            specialist_output = self._run_specialist(specialist_input, det.cls)
            if specialist_output is None:
                continue

            pose_bbox, keypoints, pose_score = specialist_output
            if mirrored:
                width = crop.shape[1]
                pose_bbox = self._mirror_xyxy(pose_bbox, width)
                keypoints = self._mirror_keypoints(keypoints, width)

            crop_offset = crop_bbox[:2]
            pose_bbox = self._translate_xyxy(pose_bbox, crop_offset)
            keypoints = self._translate_keypoints(keypoints, crop_offset)

            results.append(
                PawDetection(
                    cls=det.cls,
                    detector_score=det.score,
                    pose_score=pose_score,
                    detector_bbox=det.bbox,
                    crop_bbox=crop_bbox,
                    pose_bbox=pose_bbox,
                    crop=crop,
                    keypoints=keypoints,
                    mirrored=mirrored,
                )
            )
        return results

    def results_to_legacy(self, results):
        if len(results) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0, 0, 3), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                [],
            )
        boxes = np.asarray([r.crop_bbox for r in results], dtype=np.float32)
        keypoints = np.asarray([r.keypoints for r in results], dtype=np.float32)
        classes = np.asarray([int(r.cls) for r in results], dtype=np.int64)
        props = [
            {
                "detector_score": r.detector_score,
                "pose_score": r.pose_score,
                "detector_bbox": r.detector_bbox,
                "pose_bbox": r.pose_bbox,
                "mirrored": r.mirrored,
            }
            for r in results
        ]
        return boxes, keypoints, classes, props

    def detect_4_UI(self, img, threshold=None):
        results = self.detect_for_UI(img, threshold=threshold)
        return self.results_to_legacy(results)

    def detect_batch(self, image_list):
        return [self.detect(img_bgr) for img_bgr in image_list]


ImageSequenceExporter2 = ImageSequenceExporter
