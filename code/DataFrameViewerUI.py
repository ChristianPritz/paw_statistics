#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 17:15:32 2026

@author: wormulon
"""

import tkinter as tk
import tkinter.font as tkfont
from .paw_statistics import paw_statistics
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import os,cv2,json,inspect
from PIL import Image, ImageTk
import numpy as np
import pandas as pd 
from .interactive_plot_UI import interactive_plot_UI
from .paw_UI import ImageSequenceExporter


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


class DataFrameViewerUI:
    def __init__(self, master=None):
        self._owns_master = False
        if master is None:
            master = tk.Tk()
            self._owns_master = True

        # DataFrameViewerUI.py lives in the ``code`` directory, so its parent
        # directory is the deployable project root used by settings paths.
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        pth,_= os.path.split(inspect.getfile(paw_statistics))
        settings_path = pth + '/' + "kpt_sttngs.json"
        

        if not os.path.isfile(settings_path):
            settings_path = filedialog.askopenfilename(
                parent=master,
                defaultextension=".json",
                filetypes=[("Json files", "*.json")],
            )
        self.detector_settings = load_settings_json(settings_path)    
        self.master = master
        self.master.title("DataFrame Viewer")
        self.master.geometry("1000x600")
        _show_in_foreground(self.master)


        self.label_db = None          # original loaded dataframe
        self.display_df = None        # dataframe currently displayed
        self.current_idx = None
        self.current_cols = None
        self.loaded_path = None
        self.filter_columns = [tk.StringVar(), tk.StringVar()]
        self.filter_terms = [tk.StringVar(), tk.StringVar()]
        
        
        #--------------- Backend --------------------------
        self.backend = paw_statistics()

        # ---------- Layout: two vertical halves ----------
        self.main_pane = ttk.Panedwindow(master, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True)

        self.left_frame = ttk.Frame(self.main_pane)
        self.right_frame = ttk.Frame(self.main_pane, width=200)

        self.main_pane.add(self.left_frame, weight=4)
        self.main_pane.add(self.right_frame, weight=1)

        # ---------- Filters ----------
        self.filter_frame = ttk.Frame(self.left_frame)
        self.filter_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=4)
        self.filter_frame.columnconfigure(1, weight=1)
        self.filter_frame.columnconfigure(3, weight=1)

        self.filter_boxes = []
        self.filter_entries = []
        for filter_number in range(2):
            offset = filter_number * 2
            box = ttk.Combobox(
                self.filter_frame,
                textvariable=self.filter_columns[filter_number],
                state="readonly",
                width=18,
            )
            box.grid(row=0, column=offset, padx=(0, 4), sticky="ew")
            entry = ttk.Entry(
                self.filter_frame,
                textvariable=self.filter_terms[filter_number],
            )
            entry.grid(row=0, column=offset + 1, padx=(0, 8), sticky="ew")
            box.bind("<<ComboboxSelected>>", self._on_filter_changed)
            entry.bind("<KeyRelease>", self._on_filter_changed)
            self.filter_boxes.append(box)
            self.filter_entries.append(entry)

        self.btn_clear_filters = ttk.Button(
            self.filter_frame, text="Clear filters", command=self.clear_filters
        )
        self.btn_clear_filters.grid(row=0, column=4, sticky="ew")

        # ---------- Table + scrollbars ----------
        self.tree = ttk.Treeview(self.left_frame, show="headings")
        self.tree.bind("<<TreeviewSelect>>", self.on_row_selected)

        self.v_scroll = ttk.Scrollbar(self.left_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.h_scroll = ttk.Scrollbar(self.left_frame, orient=tk.HORIZONTAL, command=self.tree.xview)

        self.tree.configure(yscrollcommand=self.v_scroll.set,
                            xscrollcommand=self.h_scroll.set)

        self.tree.grid(row=1, column=0, sticky="nsew")
        self.v_scroll.grid(row=1, column=1, sticky="ns")
        self.h_scroll.grid(row=2, column=0, sticky="ew")

        self.left_frame.rowconfigure(1, weight=1)
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

    def _resolve_settings_path(self, path):
        """Resolve a settings path relative to the project root."""
        if not path:
            return ""
        path = os.path.expanduser(path)
        if os.path.isabs(path):
            return os.path.normpath(path)
        return os.path.normpath(os.path.join(self.base_path, path))

    # ==========================================================
    # ===================== Core Methods =======================
    # ==========================================================

 
    def load_dataframe(self, df, columns=None, selected_idx=None, view_state=None):
        """Reload the table while retaining backend row identities and UI state."""
        if columns is not None:
            self.current_cols = [col for col in columns if col in df.columns]
        elif self.current_cols is None:
            self.current_cols = list(df.columns)

        self._sync_indices_and_check()
        self._update_filter_columns()
        self._apply_filters()
        self._populate_tree(selected_idx=selected_idx, view_state=view_state)

    def _update_filter_columns(self):
        columns = list(self.backend.label_db.columns)
        for box in self.filter_boxes:
            box["values"] = columns
        for variable in self.filter_columns:
            if variable.get() not in columns:
                variable.set(columns[0] if columns else "")

    def _apply_filters(self):
        """Build a view whose index remains the positional backend row index."""
        source = self.backend.label_db
        mask = pd.Series(True, index=source.index)
        for column_var, term_var in zip(self.filter_columns, self.filter_terms):
            column = column_var.get()
            term = term_var.get().strip()
            if term and column in source.columns:
                values = source[column].fillna("").astype(str)
                mask &= values.str.contains(term, case=False, regex=False, na=False)
        self.display_df = source.loc[mask, self.current_cols].copy()

    def _populate_tree(self, selected_idx=None, view_state=None):
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(self.display_df.columns)
        table_font = tkfont.nametofont("TkDefaultFont")
        for col in self.display_df.columns:
            self.tree.heading(col, text=col)
            # Size from representative content, but keep exceptionally long
            # values from making a column unwieldy. Disabling stretch ensures
            # that many columns use the horizontal scrollbar instead of being
            # compressed to fit the window.
            sample = self.display_df[col].head(200).fillna("").astype(str)
            content_width = max(
                [table_font.measure(str(col))] +
                [table_font.measure(value) for value in sample]
            )
            column_width = min(max(content_width + 24, 110), 300)
            self.tree.column(
                col,
                width=column_width,
                minwidth=110,
                anchor="center",
                stretch=False,
            )

        for backend_idx, row in self.display_df.iterrows():
            self.tree.insert("", tk.END, iid=str(backend_idx), values=list(row))

        self.current_idx = None
        if selected_idx is not None and str(selected_idx) in self.tree.get_children():
            iid = str(selected_idx)
            self.tree.selection_set(iid)
            self.tree.focus(iid)
            self.current_idx = selected_idx

        self.master.update_idletasks()
        if view_state is not None:
            self.tree.yview_moveto(view_state[0])
            self.tree.xview_moveto(view_state[1])
        elif self.current_idx is not None:
            self.tree.see(str(self.current_idx))

    def _capture_view_state(self):
        yview = self.tree.yview()
        xview = self.tree.xview()
        return (yview[0] if yview else 0.0, xview[0] if xview else 0.0)

    def _on_filter_changed(self, _event=None):
        selected_idx = self.current_idx
        view_state = self._capture_view_state()
        self._apply_filters()
        self._populate_tree(selected_idx=selected_idx, view_state=view_state)

    def clear_filters(self):
        for term in self.filter_terms:
            term.set("")
        self._apply_filters()
        self._populate_tree()
    
    def unique_values_by_column(self, df, dropna=True, as_str=False):
        values = {}
        dtypes = {}
    
        for col in df.columns:
            vals = df[col]
            dtypes[col] = vals.dtype   # store original dtype
            has_missing = vals.isna().any()
    
            if dropna:
                vals = vals.dropna()
    
            uniq = vals.unique().tolist()
    
            if as_str:
                uniq = [str(v) for v in uniq]

            # Comboboxes need an explicit string representation for missing
            # values; pandas' NaN itself cannot be selected reliably in Tk.
            if has_missing and "nan" not in uniq:
                uniq.append("nan")
    
            values[col] = uniq
    
        return values, dtypes

    # ==========================================================
    # ===================== UI Callbacks =======================
    # ==========================================================
    def add_paws(self):
        win = tk.Toplevel(self.master)
        win.title("Add paws settings")
        win.geometry("600x600")
        _show_in_foreground(win, self.master)
        win.grab_set()   # modal
    
        main = ttk.Frame(win)
        main.pack(fill="both", expand=True, padx=15, pady=15)
    
        # -------------------------------------------------
        # Variables
        # -------------------------------------------------
        object_model_path_var   = tk.StringVar(
            value=self._resolve_settings_path(
                self.detector_settings.get("detector_model_path", find_model())
            )
        )
        hind_model_path_var = tk.StringVar(
            value=self._resolve_settings_path(
                self.detector_settings.get("hind_model_path", "")
            )
        )
        front_model_path_var = tk.StringVar(
            value=self._resolve_settings_path(
                self.detector_settings.get("front_model_path", "")
            )
        )
        metadata_path_var = tk.StringVar(
            value=self._resolve_settings_path(
                self.detector_settings.get("metadata_path", "")
            )
        )
        
        device_var       = tk.StringVar(value="cpu")
        #metadata_path_var = tk.StringVar()
        isVideo_var      = tk.BooleanVar(value=False)
        image_path_var   = tk.StringVar()
        output_path_var  = tk.StringVar()
    
        # -------------------------------------------------
        # Helper browse functions
        # -------------------------------------------------
        def browse_model_path():
            path = filedialog.askopenfilename(parent=win)
            if path:
                object_model_path_var.set(path)

        def browse_hind_model_path():
            path = filedialog.askopenfilename(parent=win)
            if path:
                hind_model_path_var.set(path)

        def browse_front_model_path():
            path = filedialog.askopenfilename(parent=win)
            if path:
                front_model_path_var.set(path)
    
        def browse_metadata_path():
            path = filedialog.askopenfilename(parent=win)
            if path:
                metadata_path_var.set(path)
    
        def browse_image_path():
            if isVideo_var.get():
                path = filedialog.askopenfilename(parent=win)
            else:
                path = filedialog.askdirectory(parent=win)
            if path:
                image_path_var.set(path)
    
        def browse_output_path():
            path = filedialog.askdirectory(parent=win)
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
            "Specify the path to the object detection YOLO model",
            object_model_path_var,
            browse_model_path,
        )

        labeled_entry(
            main,
            "Specify the path to the hind paw specialist YOLO model",
            hind_model_path_var,
            browse_hind_model_path,
        )

        labeled_entry(
            main,
            "Specify the path to the front paw specialist YOLO model",
            front_model_path_var,
            browse_front_model_path,
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
            "Specify the path where cropped images are stored",
            output_path_var,
            browse_output_path,
        )
    
        # -------------------------------------------------
        # Confirm button
        # -------------------------------------------------
        def on_confirm():
            missing_models = [
                label for label, value in (
                    ("object detection model", object_model_path_var.get()),
                    ("hind paw specialist model", hind_model_path_var.get()),
                    ("front paw specialist model", front_model_path_var.get()),
                )
                if not value
            ]
            if missing_models:
                messagebox.showwarning(
                    "Missing model path",
                    "Please specify: " + ", ".join(missing_models),
                    parent=win,
                )
                return
    
            
            #self.detector_settings["model_path"] = model_path_var.get()
            self.detector_settings["detector_model_path"] = object_model_path_var.get()
            self.detector_settings["hind_model_path"] = hind_model_path_var.get()
            self.detector_settings["front_model_path"] = front_model_path_var.get()
            self.detector_settings["metadata_path"] = metadata_path_var.get()
            self.detector_settings["device"] = device_var.get()
            
            metadata = load_settings_json(metadata_path_var.get())
            win.destroy()
            
            app = ImageSequenceExporter(self.master,
                            image_path_var.get(),
                            metadata,
                            self.detector_settings,
                            width=500,
                            image_save_dir=output_path_var.get())

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
    
        assert len(self.display_df) <= n, "display_df cannot exceed label_db length"
        assert self.display_df.index.isin(self.backend.label_db.index).all()
        assert self.backend.boxes.shape[0] == n, "boxes length mismatch"
        assert self.backend.pts.shape[0] == n, "pts length mismatch"

        # test one random row
        if n > 0:
            i = np.random.randint(0, n)
            _ = self.backend.label_db.iloc[i]
            _ = self.backend.boxes[i]
            _ = self.backend.pts[i]    


    def correct_entry(self):
        if self.current_idx is None:
            messagebox.showwarning("Warning", "No entry selected.", parent=self.master)
            return

        selected_idx = self.current_idx
        view_state = self._capture_view_state()
        bbox = self.backend.boxes[self.current_idx]
        pts = self.backend.pts[self.current_idx]
        u_vals, col_dtypes = self.unique_values_by_column(self.backend.label_db,
                                                          as_str=True)
        selected_columns = list(self.backend.label_db.columns)
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
            self.load_dataframe(
                self.backend.label_db,
                self.current_cols,
                selected_idx=selected_idx,
                view_state=view_state,
            )
        
    def delete_entry(self):
        if self.current_idx is None:
            messagebox.showwarning("Warning", "No entry selected.", parent=self.master)
            return
    
        confirm = messagebox.askyesno(
            "Confirm delete",
            f"Delete entry at index {self.current_idx}?",
            parent=self.master,
        )
        if not confirm:
            return
    
        deleted_idx = self.current_idx
        view_state = self._capture_view_state()
        visible_indices = list(self.display_df.index)
        visible_position = visible_indices.index(deleted_idx)

        # Prefer the entry below; at the bottom of the view, select the one above.
        if visible_position + 1 < len(visible_indices):
            next_idx = visible_indices[visible_position + 1]
        elif visible_position > 0:
            next_idx = visible_indices[visible_position - 1]
        else:
            next_idx = None

        # ---- delete in backend (positional) ----
        self.backend.delete_index(deleted_idx)

        # Backend indices are reset after deletion, so later rows shift by one.
        if next_idx is not None and next_idx > deleted_idx:
            next_idx -= 1
    
        # ---- resync everything ----
        self._sync_indices_and_check()
    
        # ---- reload table ----
        self.load_dataframe(
            self.backend.label_db,
            self.current_cols,
            selected_idx=next_idx,
            view_state=view_state,
        )

    def load_dataframe_dialog(self):
        self.backend.load_data_zip()
        self.loaded_path = os.getcwd()
        self.load_dataframe(self.backend.label_db)

    def save_dataframe_dialog(self):
        self.backend.save_data_zip()
    
    def merge_data(self):
        
        self.backend.merge_data_zip()

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
            "There is data in the session.\n\nDo you want to save before closing?",
            parent=self.master,
        )
    
        # Cancel → abort closing
        if answer is None:
            return
    
        # Yes → save then close
        if answer is True:
            try:
                self.backend.save_data_zip()
            except Exception as e:
                messagebox.showerror("Save error", f"Could not save data:\n{e}", parent=self.master)
                return   # abort closing if save failed

        # No or successful save → close
        self.master.quit()
        self.master.destroy()

    def cast_to_dtype(self, val, dtype):
        """
        Cast string coming from Tk widget back to original pandas dtype.
        """
        if pd.isna(val) or val == "" or (
            isinstance(val, str) and val.strip().lower() == "nan"
        ):
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
        # Rebuild display_df strictly from backend, retaining active filters.
        # -------------------------
        if self.current_cols is None:
            self.current_cols = list(self.backend.label_db.columns)
        # ensure only valid columns
        valid_cols = [c for c in self.current_cols if c in self.backend.label_db.columns]
        self.current_cols = valid_cols
        self._apply_filters()
    
        # -------------------------
        # Final consistency check
        # -------------------------
        if not self.display_df.index.isin(self.backend.label_db.index).all():
            raise RuntimeError("display_df contains rows not present in backend.label_db")
        
    
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
        _show_in_foreground(win, self.master)
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

        image_available = False
        display_offset_x = 0
        display_offset_y = 0
        connect_logic = self.detector_settings["connect_logic"]
        colors_ui = self.detector_settings["colors_ui"]

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
            image_available = True
            sx = new_size[0] / orig_w
            sy = new_size[1] / orig_h

        except (FileNotFoundError, OSError):
            # Keep metadata correction usable even when its source image is gone.
            canvas.config(width=480, height=480)
            canvas.create_text(
                240,
                22,
                text="Image not found. Keypoint correction mode disabled.",
                fill="white",
                font=("TkDefaultFont", 10, "bold"),
            )

            # Fit the stored virtual skeleton into the otherwise empty canvas.
            coordinates = []
            if current_pts is not None and len(current_pts):
                coordinates.append(np.asarray(current_pts)[:, :2])
            if current_bbox is not None and np.asarray(current_bbox).size >= 4:
                box = np.asarray(current_bbox).reshape(-1, 4)[0]
                coordinates.append(box.reshape(2, 2))

            if coordinates:
                all_coordinates = np.vstack(coordinates)
                max_x = max(float(np.nanmax(all_coordinates[:, 0])), 1.0)
                max_y = max(float(np.nanmax(all_coordinates[:, 1])), 1.0)
                scale = min(460 / max_x, 410 / max_y, 1.0)
                sx = sy = scale
                display_offset_x = 10
                display_offset_y = 50
            else:
                sx = sy = 1.0

        def draw_points_and_bbox_on_canvas():
            canvas.delete("points")
            canvas.delete("bbox")

            if current_pts is None or current_bbox is None:
                return

            for i, conn in enumerate(connect_logic):
                p1 = current_pts[conn[0], :2]
                p2 = current_pts[conn[1], :2]
                x1 = p1[0] * sx + display_offset_x
                y1 = p1[1] * sy + display_offset_y
                x2 = p2[0] * sx + display_offset_x
                y2 = p2[1] * sy + display_offset_y
                canvas.create_line(
                    x1, y1, x2, y2,
                    fill=colors_ui[i], width=3, tags="points"
                )

            searchMat = np.asarray(connect_logic)
            point_colors = []
            for i in range(len(current_pts)):
                if i == 0:
                    point_colors.append(colors_ui[-1])
                else:
                    matches = np.where(searchMat == i)
                    point_colors.append(colors_ui[int(matches[0][0])])

            for i, pt in enumerate(current_pts):
                x = pt[0] * sx + display_offset_x
                y = pt[1] * sy + display_offset_y
                canvas.create_oval(
                    x - 3.5, y - 3.5, x + 3.5, y + 3.5,
                    fill=point_colors[i], outline="", tags="points"
                )

            bx, by, ex, ey = current_bbox[0]
            canvas.create_rectangle(
                bx * sx + display_offset_x,
                by * sy + display_offset_y,
                ex * sx + display_offset_x,
                ey * sy + display_offset_y,
                outline="lime", width=2, tags="bbox"
            )
    
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
            if pd.isna(current_val) and "nan" in values:
                cb.set("nan")
            elif str(current_val) in values:
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
                messagebox.showerror("Correction error", str(e), parent=win)
    
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
    
        change_paw_button = ttk.Button(
            btn_frame,
            text="Change paw",
            command=on_change_paw,
        )
        change_paw_button.pack(fill="x", pady=2)
        if not image_available:
            change_paw_button.state(["disabled"])
    
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
