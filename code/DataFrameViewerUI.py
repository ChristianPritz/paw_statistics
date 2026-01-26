#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 17:15:32 2026

@author: wormulon
"""

import tkinter as tk
from paw_statistics import paw_statistics
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import os,cv2,json,inspect
from paw_statistics import paw_statistics
from PIL import Image, ImageTk
import numpy as np
import pandas as pd 
from interactive_plot_UI import interactive_plot_UI
from ImageSequenceExporter import ImageSequenceExporter

class DataFrameViewerUI:
    def __init__(self, master=None):
        self._owns_master = False
        if master is None:
            master = tk.Tk()
            self._owns_master = True
        
        pth,_= os.path.split(inspect.getfile(paw_statistics))
        settings_path = pth + '/' + "kpt_sttngs.json"
        

        if not os.path.isfile(settings_path):
            settings_path = filedialog.askopenfilename(defaultextension=".json",
                                                       filetypes=[("Json files","*.json")])
        self.detector_settings = load_settings_json(settings_path)    
        self.master = master
        self.master.title("DataFrame Viewer")
        self.master.geometry("1000x600")


        self.label_db = None          # original loaded dataframe
        self.display_df = None        # dataframe currently displayed
        self.current_idx = None
        self.current_cols = None
        self.loaded_path = None
        
        
        #--------------- Backend --------------------------
        self.backend = paw_statistics()

        # ---------- Layout: two vertical halves ----------
        self.main_pane = ttk.Panedwindow(master, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True)

        self.left_frame = ttk.Frame(self.main_pane)
        self.right_frame = ttk.Frame(self.main_pane, width=200)

        self.main_pane.add(self.left_frame, weight=4)
        self.main_pane.add(self.right_frame, weight=1)

        # ---------- Table + scrollbars ----------
        self.tree = ttk.Treeview(self.left_frame, show="headings")
        self.tree.bind("<<TreeviewSelect>>", self.on_row_selected)

        self.v_scroll = ttk.Scrollbar(self.left_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.h_scroll = ttk.Scrollbar(self.left_frame, orient=tk.HORIZONTAL, command=self.tree.xview)

        self.tree.configure(yscrollcommand=self.v_scroll.set,
                            xscrollcommand=self.h_scroll.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")

        self.left_frame.rowconfigure(0, weight=1)
        self.left_frame.columnconfigure(0, weight=1)

        # ---------- Buttons ----------
        self.btn_correct = ttk.Button(self.right_frame, text="Correct entry",
                                       command=self.correct_entry)
        self.btn_delete = ttk.Button(self.right_frame, text="Delete entry",
                                      command=self.delete_entry)
        self.btn_load = ttk.Button(self.right_frame, text="Load existing data",
                                    command=self.load_dataframe_dialog)
        self.btn_save = ttk.Button(self.right_frame, text="Save data",
                                    command=self.save_dataframe_dialog)
        self.btn_merge = ttk.Button(self.right_frame, text="Merge with existing data",
                                    command=self.merge_data)
        self.btn_add = ttk.Button(self.right_frame, text="Add paws",
                                    command=self.add_paws)
              
        self.btn_close = ttk.Button(self.right_frame, text="Close",
                                     command=self.close)

        for i, btn in enumerate([self.btn_correct, self.btn_delete,
                                  self.btn_load,self.btn_merge,
                                  self.btn_add,self.btn_save,
                                  self.btn_close]):
            
            btn.pack(fill=tk.X, padx=10, pady=8)
        
        if self._owns_master:
            self.master.mainloop()

    # ==========================================================
    # ===================== Core Methods =======================
    # ==========================================================

 
    def load_dataframe(self, df, columns=None):
        """Load (or reload) a dataframe into the UI with optional column filtering."""
    
        # -------------------------------
        # 0) Destroy and recreate Treeview
        # -------------------------------
        self.tree.destroy()
        self.v_scroll.destroy()
        self.h_scroll.destroy()
    
        self.tree = ttk.Treeview(self.left_frame, show="headings")
        self.tree.bind("<<TreeviewSelect>>", self.on_row_selected)
    
        self.v_scroll = ttk.Scrollbar(
            self.left_frame, orient=tk.VERTICAL, command=self.tree.yview
        )
        self.h_scroll = ttk.Scrollbar(
            self.left_frame, orient=tk.HORIZONTAL, command=self.tree.xview
        )
    
        self.tree.configure(
            yscrollcommand=self.v_scroll.set,
            xscrollcommand=self.h_scroll.set,
        )
    
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")
    
        # -------------------------------
        # 1) Build display dataframe
        # -------------------------------
        if columns is not None:
            self.display_df = df[columns].copy()
            self.current_cols = list(columns)
        else:
            self.display_df = df.copy()
            self.current_cols = list(self.display_df.columns)
    
        # -------------------------------
        # 2) Define columns
        # -------------------------------
        self.tree["columns"] = list(self.display_df.columns)
    
        for col in self.display_df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor="center", stretch=True)
    
        # -------------------------------
        # 1) Enforce consistency first
        # -------------------------------
        self._sync_indices_and_check()
        
        # -------------------------------
        # 2) Define columns
        # -------------------------------
        self.tree["columns"] = list(self.display_df.columns)
        
        for col in self.display_df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor="center", stretch=True)
        
        # -------------------------------
        # 3) Insert rows (POSitional indices only)
        # -------------------------------
        for i in range(len(self.display_df)):
            row = self.display_df.iloc[i]
            self.tree.insert(
                "",
                tk.END,
                iid=str(i),          
                values=list(row),
            )
        
        self.current_idx = None

        # -------------------------------
        # 4) Reset selection state
        # -------------------------------
        self.current_idx = None
    
    def unique_values_by_column(self, df, dropna=True, as_str=False):
        values = {}
        dtypes = {}
    
        for col in df.columns:
            vals = df[col]
            dtypes[col] = vals.dtype   # store original dtype
    
            if dropna:
                vals = vals.dropna()
    
            uniq = vals.unique().tolist()
    
            if as_str:
                uniq = [str(v) for v in uniq]
    
            values[col] = uniq
    
        return values, dtypes

    # ==========================================================
    # ===================== UI Callbacks =======================
    # ==========================================================
    def add_paws(self):
        win = tk.Toplevel(self.master)
        win.title("Add paws settings")
        win.geometry("600x600")
        win.grab_set()   # modal
    
        main = ttk.Frame(win)
        main.pack(fill="both", expand=True, padx=15, pady=15)
    
        # -------------------------------------------------
        # Variables
        # -------------------------------------------------
        model_path_var   = tk.StringVar(value = find_model())
        device_var       = tk.StringVar(value="cpu")
        metadata_path_var = tk.StringVar()
        isVideo_var      = tk.BooleanVar(value=False)
        image_path_var   = tk.StringVar()
        output_path_var  = tk.StringVar()
    
        # -------------------------------------------------
        # Helper browse functions
        # -------------------------------------------------
        def browse_model_path():
            path = filedialog.askopenfilename()
            if path:
                model_path_var.set(path)
    
        def browse_metadata_path():
            path = filedialog.askopenfilename()
            if path:
                metadata_path_var.set(path)
    
        def browse_image_path():
            if isVideo_var.get():
                path = filedialog.askopenfilename()
            else:
                path = filedialog.askdirectory()
            if path:
                image_path_var.set(path)
    
        def browse_output_path():
            path = filedialog.askdirectory()
            if path:
                output_path_var.set(path)
    
        # -------------------------------------------------
        # Layout helpers
        # -------------------------------------------------
        def labeled_entry(parent, label_text, var, browse_cmd=None):
            ttk.Label(parent, text=label_text).pack(anchor="w", pady=(10, 2))
    
            row = ttk.Frame(parent)
            row.pack(fill="x")
    
            entry = ttk.Entry(row, textvariable=var)
            entry.pack(side="left", fill="x", expand=True)
    
            if browse_cmd is not None:
                ttk.Button(row, text="Find", command=browse_cmd).pack(side="left", padx=5)
    
        # -------------------------------------------------
        # Fields
        # -------------------------------------------------
    
        # Model path
        labeled_entry(
            main,
            "Specify the path to the model",
            model_path_var,
            browse_model_path,
        )
    
        # Device
        ttk.Label(main, text="Select hardware for inference: CPU or GPU (cuda = GPU) ").pack(anchor="w", pady=(10, 2))
        device_cb = ttk.Combobox(
            main,
            values=["cpu", "cuda"],
            state="readonly",
            textvariable=device_var,
        )
        device_cb.pack(fill="x")
    
    
        # Metadata path
        labeled_entry(
            main,
            "Specify the path to the experimental meta data",
            metadata_path_var,
            browse_metadata_path,
        )
    
        # isVideo checkbox
        row_video = ttk.Frame(main)
        row_video.pack(anchor="w", pady=(15, 5))
    
        ttk.Checkbutton(
            row_video,
            variable=isVideo_var,
        ).pack(side="left")
    
        ttk.Label(
            row_video,
            text="Tick if input is a video",
        ).pack(side="left", padx=5)
    
        # Image / video path
        labeled_entry(
            main,
            "Specify the path to the image directory or video file",
            image_path_var,
            browse_image_path,
        )
    
        # Output path
        labeled_entry(
            main,
            "Specify the path for image and data export",
            output_path_var,
            browse_output_path,
        )
    
        # -------------------------------------------------
        # Confirm button
        # -------------------------------------------------
        def on_confirm():
    
            metadata = load_settings_json(metadata_path_var.get())
            self.detector_settings["model_path"] = model_path_var.get()
            self.detector_settings["device"] = device_var.get()
         
            win.destroy()
            
            app = ImageSequenceExporter(self.master,
                            image_path_var.get(),
                            metadata,
                            self.detector_settings,
                            width=500)

            # WAIT here until the exporter window is closed
            self.master.wait_window(app.root)
            
            # Now the UI is finished — safe to continue
            print("Exporter closed, continuing...")
            
            self.backend.merge_data(app.paw_stats)

            self._sync_indices_and_check()
            self.load_dataframe(self.backend.label_db, self.current_cols)
            
            
            
        ttk.Button(main, text="Confirm", command=on_confirm).pack(pady=25)

    
    def on_row_selected(self, event):
        selected = self.tree.selection()
        if selected:
            self.current_idx = int(selected[0])
            # print("Current idx:", self.current_idx)

    def _assert_full_consistency(self):
        n = len(self.backend.label_db)
    
        assert len(self.display_df) == n, "display_df length mismatch"
        assert self.backend.boxes.shape[0] == n, "boxes length mismatch"
        assert self.backend.pts.shape[0] == n, "pts length mismatch"

        # test one random row
        if n > 0:
            i = np.random.randint(0, n)
            _ = self.backend.label_db.iloc[i]
            _ = self.backend.boxes[i]
            _ = self.backend.pts[i]    


    def correct_entry(self):
        bbox = self.backend.boxes[self.current_idx]
        pts = self.backend.pts[self.current_idx]
        u_vals, col_dtypes = self.unique_values_by_column(self.backend.label_db,
                                                          as_str=True)
        
        selected_columns = self.column_selection_popup("correct",self.backend.label_db)
        image_name = self.backend.label_db["image_name"].iloc[self.current_idx]
        i_path = self.backend.label_db["image_dir"].iloc[self.current_idx]
        if isinstance(i_path,str):
            image_path = i_path + '/' + image_name
        else:
            image_path = str(i_path) + '/' + image_name
            
        out = self.open_correction_window(
            bbox,
            image_path,
            pts,
            self.current_idx,
            u_vals,
            col_dtypes,
            selected_columns,
        )
        
        if out[0] is not None:
            row = out[0]
            # enforce column dtypes explicitly before assignment
            for col in self.backend.label_db.columns:
                try:
                    row[col] = row[col]
                except:
                    pass
            self.backend.label_db.iloc[self.current_idx] = row
            self.backend.boxes[self.current_idx] = out[2]
            self.backend.pts[self.current_idx] = out[1]
            self.backend.label_db.iloc[self.current_idx]
            self.load_dataframe(self.backend.label_db, self.current_cols)
        
    def delete_entry(self):
        if self.current_idx is None:
            messagebox.showwarning("Warning", "No entry selected.")
            return
    
        confirm = messagebox.askyesno(
            "Confirm delete",
            f"Delete entry at index {self.current_idx}?"
        )
        if not confirm:
            return
    
        # ---- delete in backend (positional) ----
        self.backend.delete_index(self.current_idx)
    
        # ---- resync everything ----
        self._sync_indices_and_check()
    
        # ---- reload table ----
        self.load_dataframe(self.backend.label_db, self.current_cols)
    
        self.current_idx = None

    def load_dataframe_dialog(self):
        self.backend.load_data_zip()
        self.loaded_path = os.getcwd()

        self.column_selection_popup("load",self.backend.label_db)

    def save_dataframe_dialog(self):
        self.backend.save_data_zip()
    
    def merge_data(self):
        self.backend.merge_data(app.paw_stats)

        self._sync_indices_and_check()
        self.load_dataframe(self.backend.label_db, self.current_cols)

        
        

    def close(self):
        has_backend_data = (
            self.backend.label_db is not None and
            len(self.backend.label_db) > 0
        )
    
        has_display_data = (
            self.display_df is not None and
            len(self.display_df) > 0
        )
    
        # If no data at all → close directly
        if not has_backend_data and not has_display_data:
            self.master.quit()
            self.master.destroy()
            return
    
        # Otherwise ask the user
        answer = messagebox.askyesnocancel(
            "Unsaved data",
            "There is data in the session.\n\nDo you want to save before closing?"
        )
    
        # Cancel → abort closing
        if answer is None:
            return
    
        # Yes → save then close
        if answer is True:
            try:
                self.backend.save_data_zip()
            except Exception as e:
                messagebox.showerror("Save error", f"Could not save data:\n{e}")
                return   # abort closing if save failed

        # No or successful save → close
        self.master.quit()
        self.master.destroy()

    # ==========================================================
    # ================= Column Selection UI ====================
    # ==========================================================

    def column_selection_popup(self, phase, df):
        popup = tk.Toplevel(self.master)
        popup.title("Select columns for display")
        popup.geometry("300x400")
        popup.grab_set()   # <-- make it modal
    
        container = ttk.Frame(popup)
        container.pack(fill=tk.BOTH, expand=True)
    
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
    
        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
    
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
    
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
        col_vars = {}
        for col in df.columns:
            var = tk.BooleanVar(value=True)
            chk = ttk.Checkbutton(scroll_frame, text=col, variable=var)
            chk.pack(anchor="w", padx=10, pady=2)
            col_vars[col] = var
    
        selected_cols = []   # <-- buffer to return result
    
        def on_confirm():
            nonlocal selected_cols
            selected_cols = [c for c, v in col_vars.items() if v.get()]
    
            if not selected_cols:
                messagebox.showwarning("Warning", "Select at least one column.")
                return
    
            popup.destroy()
    
        btn_confirm = ttk.Button(popup, text="Confirm", command=on_confirm)
        btn_confirm.pack(pady=10)
    
        # ⏸️ block until popup is closed
        self.master.wait_window(popup)
    
        # phase-specific behavior
        if phase == "load" and selected_cols:
            self.load_dataframe(self.backend.label_db, selected_cols)
            self.current_cols = selected_cols
    
        return selected_cols
    
    def cast_to_dtype(self, val, dtype):
        """
        Cast string coming from Tk widget back to original pandas dtype.
        """
        if pd.isna(val) or val == "":
            return np.nan
    
        try:
            # integer columns
            if np.issubdtype(dtype, np.integer):
                return int(val)
    
            # float columns
            if np.issubdtype(dtype, np.floating):
                return float(val)
    
            # boolean columns
            if np.issubdtype(dtype, np.bool_):
                if isinstance(val, str):
                    return val.lower() in ("true", "1", "yes")
                return bool(val)
    
            # categorical columns
            if isinstance(dtype, pd.CategoricalDtype):
                return val
    
            # string / object columns
            return val
    
        except Exception:
            # fallback: return original string if casting fails
            return val
    
    def _sync_indices_and_check(self):
        """
        Enforce:
          - backend.label_db index = 0..N-1
          - display_df is strict column-subset of backend.label_db
          - sizes are consistent
        """
    
        # -------------------------
        # Reset backend index
        # -------------------------
        if self.backend.label_db is not None:
            self.backend.label_db = self.backend.label_db.reset_index(drop=True)
    
        # -------------------------
        # Safety check for backend arrays
        # -------------------------
        n = len(self.backend.label_db)
    
        if hasattr(self.backend, "boxes") and len(self.backend.boxes) != n:
            raise RuntimeError("Mismatch: boxes length != label_db length")
    
        if hasattr(self.backend, "pts") and len(self.backend.pts) != n:
            raise RuntimeError("Mismatch: pts length != label_db length")
    
        # -------------------------
        # Rebuild display_df strictly from backend
        # -------------------------
        if self.current_cols is None:
            self.display_df = self.backend.label_db.copy()
            self.current_cols = list(self.display_df.columns)
        else:
            # ensure only valid columns
            valid_cols = [c for c in self.current_cols if c in self.backend.label_db.columns]
            self.current_cols = valid_cols
            self.display_df = self.backend.label_db[valid_cols].copy()
    
        # -------------------------
        # Final consistency check
        # -------------------------
        if len(self.display_df) != len(self.backend.label_db):
            raise RuntimeError("display_df and backend.label_db length mismatch")
        
    
    def open_correction_window(
        self,
        bbox,
        image_path,
        points,
        row_index,
        unique_values_by_column,
        column_dtypes,          # 👈 NEW
        selected_columns,
    ):
        """
        Opens a correction window that allows:
          - viewing an image
          - editing selected DataFrame columns via dropdowns
          - correcting paw points + bounding box via interactive_plot_UI
    
        Returns:
            updated_row (pd.Series) or None
            updated_points (np.ndarray) or None
            updated_bbox (np.ndarray) or None
        """
        
                
        
        win = tk.Toplevel(self.master)
        win.title("Correction window")
        win.geometry("1000x600")
        win.grab_set()  # modal
    
        # ------------------------------------------------------------------
        # State buffers
        # ------------------------------------------------------------------
        result = {
            "row": None,
            "points": None,
            "bbox": None,
            "saved": False,
        }
    
        current_pts = points.copy() if points is not None else None
        current_bbox = np.array(bbox).copy() if bbox is not None else None
    
        # ------------------------------------------------------------------
        # Layout
        # ------------------------------------------------------------------
        main_frame = ttk.Frame(win)
        main_frame.pack(fill="both", expand=True)
    
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
    
        # ------------------------------------------------------------------
        # Scrollable RIGHT frame (metadata + buttons)
        # ------------------------------------------------------------------
        right_container = ttk.Frame(main_frame)
        right_container.pack(side="right", fill="y", padx=5, pady=5)
        
        canvas_right = tk.Canvas(right_container, width=300)
        scrollbar = ttk.Scrollbar(right_container, orient="vertical", command=canvas_right.yview)
        
        scrollable_frame = ttk.Frame(canvas_right)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_right.configure(
                scrollregion=canvas_right.bbox("all")
            )
        )
        
        canvas_right.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_right.configure(yscrollcommand=scrollbar.set)
        
        canvas_right.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # This is now your working right frame
        right_frame = scrollable_frame
        
        
        # ------------------------------------------------------------------
        # Image display (LEFT)
        # ------------------------------------------------------------------
        canvas = tk.Canvas(left_frame, bg="black")
        canvas.pack(fill="both", expand=True)
    
        try:
            img = Image.open(image_path).convert("RGB")
            orig_w, orig_h = img.width, img.height
            max_w, max_h = 480, 480
            scale = min(max_w / img.width, max_h / img.height, 1.0)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(img)
            canvas.config(width=new_size[0], height=new_size[1])
            canvas.create_image(0, 0, anchor="nw", image=imgtk)
            canvas.image = imgtk
            # ------------------------------------------------------------------
            # Overlay plotting: points + bbox
            # ------------------------------------------------------------------
            connect_logic = self.detector_settings["connect_logic"]
            colors_ui = self.detector_settings["colors_ui"]
            
            # scale factors: original image → displayed image
            #sx = new_size[0] / img.width
            #sy = new_size[1] / img.height
            sx = new_size[0] / orig_w
            sy = new_size[1] / orig_h
            
            
            def draw_points_and_bbox_on_canvas():
                canvas.delete("points")
                canvas.delete("bbox")
            
                if current_pts is None or current_bbox is None:
                    return
            
                # ----------------------------
                # Draw connectivity lines
                # ----------------------------
                for i, conn in enumerate(connect_logic):
                    p1 = current_pts[conn[0], :2]
                    p2 = current_pts[conn[1], :2]
            
                    x1, y1 = p1[0] * sx, p1[1] * sy
                    x2, y2 = p2[0] * sx, p2[1] * sy
            
                    canvas.create_line(
                        x1, y1, x2, y2,
                        fill=colors_ui[i],
                        width=3,
                        tags="points"
                    )
            
                # ----------------------------
                # Point colors (same logic as your UI)
                # ----------------------------
                searchMat = np.asarray(connect_logic)
                point_colors = []
            
                for i in range(len(current_pts)):
                    if i == 0:
                        point_colors.append(colors_ui[-1])
                    else:
                        a = np.where(searchMat == i)
                        point_colors.append(colors_ui[int(a[0][0])])
            
                # ----------------------------
                # Draw keypoints
                # ----------------------------
                for i, pt in enumerate(current_pts):
                    x, y = pt[0] * sx, pt[1] * sy
                    canvas.create_oval(
                        x - 3.5, y - 3.5,
                        x + 3.5, y + 3.5,
                        fill=point_colors[i],
                        outline="",
                        tags="points"
                    )
            
                # ----------------------------
                # Draw bounding box
                # ----------------------------
                bx, by, ex, ey = current_bbox[0]
                #ex = bx + ex
                #ey = by + ey
                
                x1 = bx * sx
                y1 = by * sy
                x2 = ex * sx
                y2 = ey * sy
                
                #x2 = (bx + bw) * sx
                #y2 = (by + bh) * sy
            
                canvas.create_rectangle(
                    x1, y1, x2, y2,
                    outline="lime",
                    width=2,
                    tags="bbox"
                )

            
            
        
        except Exception as e:
            messagebox.showerror("Image error", f"Could not load image:\n{e}")
    
        # ------------------------------------------------------------------
        # Dropdown fields (RIGHT)
        # ------------------------------------------------------------------
        draw_points_and_bbox_on_canvas() 
        ttk.Label(
            right_frame,
            text="Metadata correction",
            font=("TkDefaultFont", 10, "bold"),
        ).pack(pady=(0, 8))
    
        row = self.backend.label_db.iloc[row_index]
        field_vars = {}
    
        for col in selected_columns:
            ttk.Label(right_frame, text=col).pack(anchor="w")
    
            values = unique_values_by_column.get(col, [])
            cb = ttk.Combobox(
                right_frame,
                values=values,
                state="readonly",
            )
            current_val = row[col]
            if current_val in values:
                cb.set(current_val)
            elif len(values) > 0:
                cb.set(values[0])
            else:
                cb.set("")
    
            cb.pack(fill="x", pady=2)
            field_vars[col] = cb
    
        # ------------------------------------------------------------------
        # Button callbacks
        # ------------------------------------------------------------------
        def on_change_paw():
            nonlocal current_pts, current_bbox
           
            if current_pts is None or current_bbox is None:
                #messagebox.showwarning("No data", "No points/bounding box to correct.")
                return
    
            try:
                img = cv2.imread(image_path)    
                
                
                app = interactive_plot_UI(
                    win,
                    img,
                    current_pts[:, [0, 1]],
                    current_bbox[0],
                    self.detector_settings["connect_logic"],
                    self.detector_settings["colors_ui"],
                    title="Inspect and correct paw",
                    window_size=[900, 900],
                )
    
                pts_out, bbox_out = app.return_data()

                current_pts = np.ones((pts_out.shape[0],3))
                current_pts[:,[0,1]] = pts_out
                current_bbox = bbox_out.reshape((1,bbox_out.shape[0]))
                
                draw_points_and_bbox_on_canvas()
   
   
            except Exception as e:
                messagebox.showerror("Correction error", str(e))
    
        def on_save():
            updated_row = row.copy()
        
            # only update subselected columns
            for col, widget in field_vars.items():
                raw_val = widget.get()
                dtype = column_dtypes.get(col, object)
        
                cast_val = self.cast_to_dtype(raw_val, dtype)
                updated_row[col] = cast_val
        
            result["row"] = updated_row
            result["points"] = current_pts
            result["bbox"] = current_bbox
            result["saved"] = True
        
            win.destroy()
    
        def on_exit():
            win.destroy()
    
        # ------------------------------------------------------------------
        # Buttons
        # ------------------------------------------------------------------
        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(pady=10, fill="x")
    
        ttk.Button(
            btn_frame,
            text="Change paw",
            command=on_change_paw,
        ).pack(fill="x", pady=2)
    
        ttk.Button(
            btn_frame,
            text="Save changes",
            command=on_save,
        ).pack(fill="x", pady=2)
    
        ttk.Button(
            btn_frame,
            text="Exit without saving",
            command=on_exit,
        ).pack(fill="x", pady=2)
    
        # ------------------------------------------------------------------
        # Block until window closes
        # ------------------------------------------------------------------
        win.wait_window()
    
        if result["saved"]:
            return result["row"], result["points"], result["bbox"]
        else:
            return None, None, None

def find_model():
    pth,_= os.path.split(inspect.getfile(paw_statistics))
    parent_pth = os.path.dirname(pth)
    mdl_path = parent_pth + '/model/' + "model_torch.pt"
    return mdl_path


def load_settings_json(fpath):
    with open(fpath, "r", encoding="utf-8") as f:
        my_dict = json.load(f)
    return my_dict


# ==========================================================
# ======================= Run App ==========================
# ==========================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = DataFrameViewerUI(root)
    root.mainloop()