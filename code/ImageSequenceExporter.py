#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 14:14:46 2025

@author: Christian Pritz
"""

import glob, copy,scipy
import cv2, os, posixpath
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
import matplotlib.pyplot as plt
from paw_detector_torch import paw_detector
from interactive_plot_UI import interactive_plot_UI
from paw_statistics import paw_statistics
from IPython import embed


class ImageSequenceExporter:
    def __init__(self, image_dir, metadata, detector_settings,
                 width=300, factor=3, output_dir=None, prefix="", paw_stats=None):
        self.root = tk.Tk()
        self.image_dir = Path(image_dir)
        self.metadata = metadata
        self.dataframe = pd.DataFrame(columns=list(metadata.keys()) +
                                      ["image_name", "source_image", "crop_index"])
        self.output_dir = Path(output_dir) if output_dir else self.image_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.factor = factor
        self.prefix = prefix
    


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

        self.export_bbox = None
        self.export_pts = None
        self.export_side = None

        self.current_index = 0
        self.threshold = 0.9
        self.tolerance = 0.1  # default
        self.counter = {}     # per-image crop counter
        self.detector_settings = detector_settings
        self.crpr = paw_cropper(detector_settings['model_path'],
                                detector_settings['device'],
                                detector_settings["connect_logic"],
                                detector_settings["keypoint_names"],
                                threshold=detector_settings['threshold'],
                                video=str(self.image_dir) if self.is_video else None,
                                directory=str(self.image_dir if not self.is_video else self.output_dir),
                                output_dir=str(self.output_dir))
        if paw_stats is not None:
            self.paw_stats = paw_statistics(None)
            self.paw_stats.load_data_zip(filename = paw_stats)
            print("DATA LOADED FROM: " + paw_stats)
        
        else:
            columns = list(metadata.keys()) + ["image_name", "source_image",
                                               "crop_index", "predicted_side", 
                                               "image_dir","frame_number"]
            self.paw_stats = paw_statistics(None, columns=columns)

        self.create_ui()
        

    def create_ui(self):
        self.root.title("Image Dir Frame Exporter")

        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True)

        self.video_frame = ttk.Frame(self.main_frame)
        self.video_frame.pack(side="left", fill="both", expand=True)
        self.metadata_frame = ttk.Frame(self.main_frame)
        self.metadata_frame.pack(side="right", fill="y")

        self.canvas = tk.Canvas(self.video_frame, width=self.width, height=self.width, bg="black")
        self.canvas.pack(padx=5, pady=5)

        nav_frame = ttk.Frame(self.video_frame)
        nav_frame.pack()
        ttk.Button(nav_frame, text="<< Prev", command=self.prev_frame).pack(side="left")
        ttk.Button(nav_frame, text="Next >>", command=self.next_frame).pack(side="left")
        # --- Frame navigation bar (scroll + entry) ---
        nav_frame_bottom = ttk.Frame(self.video_frame)
        nav_frame_bottom.pack(fill="x", pady=(5, 10))
        
        ttk.Label(nav_frame_bottom, text="Frame:").pack(side="left", padx=5)
        
        # Scrollbar to browse through frames
        self.frame_scroll = ttk.Scale(
            nav_frame_bottom,
            from_=0,
            to=len(self.image_files) - 1,
            orient="horizontal",
            command=self.on_frame_scroll
        )
        self.frame_scroll.pack(side="left", fill="x", expand=True, padx=5)
        
        # Entry field for numeric frame input
        self.frame_index_var = tk.StringVar(value=str(self.current_index))
        self.frame_entry = ttk.Entry(nav_frame_bottom, textvariable=self.frame_index_var, width=6)
        self.frame_entry.pack(side="right", padx=5)
        self.frame_entry.bind("<Return>", self.on_frame_entry)


        self.export_button = ttk.Button(nav_frame, text="Export cropped", command=self.export_segmented_paw)
        self.export_button.pack(side="left")

        # --- Scrollbar for prediction selection ---
        self.prediction_scale = ttk.Scale(
            self.metadata_frame, from_=0, to=0, orient="horizontal",
            command=self.on_prediction_change
        )
        ttk.Label(self.metadata_frame, text="Select Prediction:").pack(pady=5)
        self.prediction_scale.pack(pady=5, fill="x")
        self.prediction_scale.set(0)

        ttk.Label(self.metadata_frame, text="Detection Threshold (0–1):").pack(pady=2)
        self.threshold_entry = ttk.Entry(self.metadata_frame)
        self.threshold_entry.insert(0, str(self.threshold))
        self.threshold_entry.pack(pady=2)
        self.threshold_entry.bind("<Return>", self.on_threshold_change)        

        ttk.Label(self.metadata_frame, text="Crop Tolerance (0–1 or inf):").pack(pady=2)
        self.tolerance_entry = ttk.Entry(self.metadata_frame)
        self.tolerance_entry.insert(0, str(self.tolerance))
        self.tolerance_entry.pack(pady=2)

        ttk.Label(self.metadata_frame, text="Metadata:").pack(pady=5)
        self.metadata_inputs = {}
        for key, value in self.metadata.items():
            ttk.Label(self.metadata_frame, text=key).pack()
            if isinstance(value, list):
                cb = ttk.Combobox(self.metadata_frame, values=value)
                cb.set(value[0])
                cb.pack()
                self.metadata_inputs[key] = cb
            else:
                ent = ttk.Entry(self.metadata_frame)
                ent.insert(0, value)
                ent.pack()
                self.metadata_inputs[key] = ent

        ttk.Button(self.metadata_frame, text="Select Output Dir", command=self.select_output_directory).pack(pady=5)
        ttk.Button(self.metadata_frame, text="Save & Exit", command=self.save_and_exit).pack(pady=10)

        self.update_frame()
        self.root.mainloop()

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
                messagebox.showwarning("Invalid Input", "Enter a float between 0 and 1.")
                self.threshold_entry.delete(0, tk.END)
                self.threshold_entry.insert(0, str(self.threshold))
        except ValueError:
            messagebox.showwarning("Invalid Input", "Enter a valid float between 0 and 1.")
            self.threshold_entry.delete(0, tk.END)
            self.threshold_entry.insert(0, str(self.threshold))
        
    def on_prediction_change(self, value):
        """Triggered when the prediction selection scale changes."""
        try:
            self.current_box_selection = int(float(value))
        except ValueError:
            self.current_box_selection = 0
        self.redraw_bboxes()

    def redraw_bboxes(self):
        """Redraw image and bounding boxes according to current selection."""
        # Draw image
        #img_path = self.image_files[self.current_index]
        #frame = cv2.imread(str(img_path))
        if hasattr(self,"current_frame"):
            self.resize_image(self.current_frame)

        # Draw boxes
            for idx, box in enumerate(self.current_boxes):
                x1, y1, x2, y2 = box
                x1 /= self.scaler[1]; x2 /= self.scaler[1]
                y1 /= self.scaler[0]; y2 /= self.scaler[0]
                color = "red" if idx == self.current_box_selection else "blue"
                self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
    
    def resize_image(self,frame):
        
        old_h, old_w = frame.shape[:2]
        if old_h < old_w:
            self.height = round(self.width * old_h / old_w)
            self.scaler = np.array([old_h / self.height, old_w / self.width])
        
            # Draw placeholder image
            frame_rgb = cv2.cvtColor(cv2.resize(frame, (self.width, self.height)), cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
            self.canvas.image = imgtk
        else: 
            self.height = self.width
            width = round(self.width * old_w/old_h)
            self.scaler = np.array([old_h / self.height, old_w / width])
        
            # Draw placeholder image
            frame_rgb = cv2.cvtColor(cv2.resize(frame, (width, self.height)), cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
            self.canvas.image = imgtk
        
    
    def update_frame(self):

        
        # --- Load image or video frame depending on mode ---
        if self.is_video:
    
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_index)
            ret, frame = self.video_capture.read()
            if not ret:
                messagebox.showerror("Error", f"Cannot read frame {self.current_index}")
                return
            img_name = f"frame_{self.current_index:05d}.png"
        else:
            img_path = self.image_files[self.current_index]
            frame = cv2.imread(str(img_path))
            img_name = Path(img_path).name
            if frame is None:
                messagebox.showerror("Error", f"Cannot load {img_path}")
                return
        self.current_frame = frame
        self.update_nav_controls()
        # --- Handle tolerance input ---
        tol = self.tolerance_entry.get().lower()
        try:
            self.tolerance = float(tol) if tol != 'inf' else float('inf')
        except ValueError:
            messagebox.showerror("Error", "Invalid tolerance (enter float or inf)")
            return

        # --- Run detection ---
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
        
        if bboxes is None or len(bboxes) == 0: # incase of no predictions.... 
            self.current_pts = []
            self.current_boxes = []
            self.current_orig_boxes = []
            self.current_classes = []
            self.current_paw_images = []
            self.canvas.delete("all")
            
            self.resize_image(frame)
        
            # Overlay text
            self.canvas.create_text(
                self.width // 2, self.height // 2,
                text="⚠️ No predictions found",
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

        old_h, old_w = frame.shape[:2]
        self.height = round(self.width * old_h / old_w)
        self.scaler = np.array([old_h / self.height, old_w / self.width])

        # --- Redraw canvas ---
        frame_rgb = cv2.cvtColor(cv2.resize(frame, (self.width, self.height)), cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
        self.canvas.image = imgtk
        self.redraw_bboxes()

    def correct_prediction(self,fname):
        img = self.current_paw_images[self.current_box_selection]
        pts_in = self.current_pts[self.current_box_selection]
        bbox_in = self.current_orig_boxes[self.current_box_selection]
        pts_in = self.current_pts[self.current_box_selection]
        class_out = self.current_classes[self.current_box_selection]
        
        bbox_in[0] = bbox_in[0]-self.offset_x[self.current_box_selection]
        bbox_in[1] = bbox_in[1]-self.offset_y[self.current_box_selection]
        bbox_in[2] = bbox_in[2]-self.offset_x[self.current_box_selection]
        bbox_in[3] = bbox_in[3]-self.offset_y[self.current_box_selection]
        
        pts_in = pts_in - np.tile([self.offset_x[self.current_box_selection],
                                   self.offset_y[self.current_box_selection],0],
                                  (15,1))

        app = interactive_plot_UI(self.root, img, pts_in[:, [0, 1]], bbox_in,
                                  self.detector_settings["connect_logic"],
                                  self.detector_settings["colors_ui"],
                                  title='Correct the points',
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


    def save_to_mat(self,pts, bxs, data_dict, image_name):
        """
        Saves Python objects into a .mat file in a specified cell array format.
    
        Parameters:
            pts (np.ndarray): Array to be placed in cell (1,2)
            bxs (np.ndarray): Array to be placed in cell (2,2)
            data_dict (dict): Dictionary containing 'genotype', 'paw_posture', and 'side'
            filename (str): Name of the output .mat file
        """
        
        filename = image_name[0:-4] + '.mat'
        cell_array = [["points", '', '', '', '', '', ''],  # 1
                      ["rois", '', '', '', '', '', ''],  # 2
                      ["visability", '', '', '', '', '', ''],  # 3
                      ["truncated", '', '', '', '', '', ''],  # 4
                      ["height", '', '', '', '', '', ''],  # 5
                      ["width", '', '', '', '', '', ''],  # 6
                      ["genotype", '', '', '', '', '', ''],  # 7
                      ["gender", '', '', '', '', '', ''],  # 8
                      ["side", '', '', '', '', '', ''],  # 9
                      ["treatment", '', '', '', '', '', ''],  # 10
                      ["paw_posture", '', '', '', '', '', ''],  # 11
                      ["pain_status", '', '', '', '', '', ''],  # 12
                      ["useful", '', '', '', '', '', ''],  # 13
                      ["remark", '', '', '', '', '', ''],  # 14
                      ["animal_ID", '','', '', '', '', '']]  # 15
        
        for i in range(len(data_dict)):
        # Create an empty cell array (MATLAB cell array is represented as a list of lists in Python)
    
    
            if len(pts.shape) == 3:
                cell_array[0][i+1] = pts[i]        # (1,2) -> cell_array[0][1]
                cell_array[1][i+1] = bxs[i]        # (2,2) -> cell_array[1][1]
            else:
                cell_array[0][i+1] = pts        # (1,2) -> cell_array[0][1]
                cell_array[1][i+1] = bxs        # (2,2) -> cell_array[1][1]
                
            cell_array[2][i+1] = data_dict[i]["visibility"]    # (3,2) -> cell_array[2][1]
            cell_array[3][i+1] = data_dict[i]["truncated"] # (4,2) -> cell_array[3][1]
            cell_array[4][i+1] = data_dict[i]["height"]        # (5,2) -> cell_array[4][1]
            cell_array[5][i+1] = data_dict[i]["width"]
            cell_array[6][i+1] = data_dict[i]["genotype"]
            cell_array[7][i+1] = data_dict[i]["gender"]
            cell_array[8][i+1] = data_dict[i]["side"]
            cell_array[9][i+1] = data_dict[i]["treatment"]
            cell_array[10][i+1] = data_dict[i]["paw_posture"]
            cell_array[11][i+1] = data_dict[i]["pain_status"]
            cell_array[12][i+1] = data_dict[i]["useful"]
            cell_array[13][i+1] = data_dict[i]["remark"]
            cell_array[14][i+1] = data_dict[i]["animal_ID"]
    
            
        for j in cell_array:
            for i in np.arange(len(data_dict)+1,7)[::-1]:
                j.pop(i)

        mat_dict = {'varList': cell_array}
    
        # Save to .mat file
        scipy.io.savemat(filename, mat_dict)

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
                messagebox.showwarning("Invalid Index", f"Enter a number between 0 and {len(self.image_files)-1}")
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid integer.")

    def set_prediction_controls_state(self, state: str):
        self.prediction_scale.state([state])
        self.export_button.state([state])
    
    def export_segmented_paw(self):
        if not hasattr(self, "current_boxes") or len(self.current_boxes) == 0:
            messagebox.showwarning("No predictions", "No predictions available for this frame.")
            return


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
                   "crop_index": sel,"predicted_side":self.export_side,
                   "image_dir":self.output_dir,
                   "frame_number":self.current_index})
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
        
        messagebox.showinfo("Success", f"Saved {out_name}")
        
        
        

    def select_output_directory(self):
        d = filedialog.askdirectory(title="Select Output Directory")
        if d:
            self.output_dir = Path(d)
            self.output_dir.mkdir(exist_ok=True)
            messagebox.showinfo("Output Dir", f"Now saving to {d}")


    def save_data(self):

        csv = self.output_dir / f"{self.prefix}-{self.image_dir.name}.csv"
        self.dataframe.to_csv(csv, index=False)
        zip_name = self.output_dir / f"{self.prefix}-{self.image_dir.name}.zip"
        self.paw_stats.all_angles()
        self.paw_stats.save_data_zip(filename=zip_name)
        self.csv_name = csv
        self.zip_name = zip_name

    def save_and_exit(self):
        tol = self.tolerance_entry.get().lower()
        self.tolerance = float(tol) if tol != 'inf' else float('inf')
        self.save_data()
        messagebox.showinfo("Saved", f"Metadata saved to {self.csv_name}, and full data to {self.zip_name}")
        self.root.destroy()

    
    def generate_empty_prediction(self,img):
        def ask_usr():
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            user_input = tk.simpledialog.askinteger("No paw detected!", "Please enter the number of non-predicted paws:")
            root.destroy()  # Close the hidden main window
            return user_input
        
        user_input = ask_usr()
        if user_input == 0:
            return [],[],[]
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
                #bxs[i,:] = np.asarray([np.min(rx[:,0]),np.max(rx[:,1]),np.max(rx[:,0]),np.min(rx[:,1])]) 
                bxs[i,:] = np.asarray([np.max(rx[:,0]),np.min(rx[:,1]),np.min(rx[:,0]),np.max(rx[:,1])]) 
                
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

#keypoint names, and keypoint logic not required anymore, remove in next update
class paw_cropper:
    def __init__(self, model_path, device, keypoint_logic, keypoint_names,
                 video=None, directory=None, output_dir=None,threshold=0.9):
        self.detector = paw_detector(model_path,device,threshold=threshold)
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
        bxs,kps,cls = self.detector.detect_4_UI(img, threshold=threshold)
        pts, boxes, imgs, classes,orig_bboxes = [], [], [], [],[]
        for j in range(kps.shape[0]):
            pt = kps[j,:,0:3].reshape([15,3]) # hardcoded size
            bbox = bxs[j].reshape((4))
            clss = cls[j]
            crop, coords = self.crop_image(img, bbox, tolerance)
            pts.append(pt)
            classes.append(clss)
            boxes.append(coords)
            orig_bboxes.append(bbox)
            imgs.append(crop)
        return pts, boxes, imgs,classes,orig_bboxes

    def crop_image(self, image, bbox, tolerance):
        dx = (bbox[2] - bbox[0]) * tolerance
        dy = (bbox[3] - bbox[1]) * tolerance
        xst = int(max(bbox[0] - dx, 0))
        yst = int(max(bbox[1] - dy, 0))
        xen = int(min(bbox[2] + dx, image.shape[1]))
        yen = int(min(bbox[3] + dy, image.shape[0]))
        return image[yst:yen, xst:xen, :], [xst, yst, xen, yen]
