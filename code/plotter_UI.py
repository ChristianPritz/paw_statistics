import tkinter as tk
import os, copy, ast
import numpy as np
from tkinter import colorchooser
from tkinter import ttk, filedialog
from itertools import combinations, product
from IPython import embed

class PlotterUI:
    """
    Fully self-contained Tkinter UI.
    """

    def __init__(self, backend_obj):

        # root (hidden)
        self._root = tk.Tk()
        self._root.withdraw()

        # visible window
        self.win = tk.Toplevel(self._root)
        self.win.title("Plotting Control Panel")
        self.win.protocol("WM_DELETE_WINDOW", self.close)

        # Backend
        self.backend = backend_obj
        df = backend_obj.label_db

        # Properties
        self.label_col = ""
        self.angles = "named angles"
        self.selected_pcs = ""
        self.xlabel = ""
        self.ylabel = ""
        self.xlim = "auto"
        self.ylim = "auto"
        self.xticks = "[]"
        self.yticks = "[]"
        self.figsize = "[6,6]"
        self.save_path = ""
        self.pc_combos = None
        self.color_axis = [0, 1]
        self.current_plot = 1  # NEW: store selected plot type
        temp = [{'figsize':None,'xlim':'AUTO','ylim':'AUTO',
                'xlabel':None,'ylabel':None,'xticks':None,
                'yticks':None}]
        self.plot_props = [copy.deepcopy(x) for _ in range(4) for x in temp]
        self.group_colors = np.array([[0.5,0.5,0.5],[0.3,0.3,0.3]])

        # these properties get reinitiated by reset conditions 
        self.current_unique = []
        self.unique_grps = []
        self.all_columns = []
        self.n_groups = 0

        # Main grid with column headers
        # --- Configure grid so columns expand down ---
        self.win.grid_rowconfigure(1, weight=1)
        for i in range(4):
            self.win.grid_columnconfigure(i, weight=1)
        
        # Main grid with centered column headers
        header_frame = tk.Frame(self.win)
        header_frame.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(5,0))
        
        # Make header columns expand equally
        for i in range(4):
            header_frame.grid_columnconfigure(i, weight=1)
        
        tk.Label(header_frame, text="Conditions",
                 font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="n")
        tk.Label(header_frame, text="Parameters",
                 font=("Arial", 12, "bold")).grid(row=0, column=1, sticky="s")
        tk.Label(header_frame, text="Plot Properties",
                 font=("Arial", 12, "bold")).grid(row=0, column=2, sticky="n")
        tk.Label(header_frame, text="", width=10).grid(row=0, column=3)  # spacer
        
        # Columns - now stretch vertically ("ns")
        self.col1 = tk.Frame(self.win, bd=1, relief="solid")
        self.col1.grid(row=1, column=0, padx=5, pady=10, sticky="ns")
        
        self.col2 = tk.Frame(self.win, bd=1, relief="solid")
        self.col2.grid(row=1, column=1, padx=5, pady=10, sticky="ns")
        
        self.col3 = tk.Frame(self.win, bd=1, relief="solid")
        self.col3.grid(row=1, column=2, padx=5, pady=10, sticky="ns")
        
        self.col4 = tk.Frame(self.win)
        self.col4.grid(row=1, column=3, padx=10, pady=10, sticky="ns")


        # ----- COLUMN 1 -----
        tk.Label(self.col1, text="Select Label Column:").pack(anchor="w")
        columns = list(df.columns)
        self.col_var = tk.StringVar(value=columns[0])
        self.label_col = columns[0]

        self.col_dropdown = ttk.Combobox(
            self.col1, textvariable=self.col_var, values=columns, state="readonly"
        )
        self.col_dropdown.pack(anchor="w")
        self.col_dropdown.bind("<<ComboboxSelected>>", self._update_label_col)

        # Scrollable list
        tk.Label(self.col1, text="Values in selected column:").pack(anchor="w", pady=(10, 0))
        self.unique_frame = tk.Frame(self.col1)
        self.unique_frame.pack(anchor="w", fill="both", expand=False)

        self.unique_scroll = tk.Scrollbar(self.unique_frame, orient="vertical")
        self.unique_list = tk.Listbox(
            self.unique_frame,
            height=6,
            width=25,
            yscrollcommand=self.unique_scroll.set,
            exportselection=False
        )

        self.unique_scroll.config(command=self.unique_list.yview)
        self.unique_scroll.pack(side="right", fill="y")
        self.unique_list.pack(side="left", fill="both", expand=True)

        tk.Button(self.col1, text="Add condition",
                  command=self._update_conditions).pack(anchor="w", pady=5)

        tk.Button(self.col1, text="Reset conditions",
                  command=self._reset_conditions).pack(anchor="w", pady=5)

        tk.Label(self.col1, text="Included columns:").pack(anchor="w", pady=(10, 0))
        self.all_groups_var = tk.StringVar(value="")
        self.all_groups_entry = tk.Entry(self.col1, textvariable=self.all_groups_var,
                                         state="readonly", width=30)
        self.all_groups_entry.pack(anchor="w")
        
        tk.Label(self.col1, text="groups:").pack(anchor="w", pady=(10, 0))
        self.exp_groups_var = tk.StringVar(value="")
        self.exp_groups_entry = tk.Entry(self.col1, textvariable=self.exp_groups_var,
                                         state="readonly", width=30)
        self.exp_groups_entry.pack(anchor="w")
        

        # ----- COLUMN 2 -----
        tk.Label(self.col2, text="Angles:").pack(anchor="w")
        self.angles_var = tk.StringVar(value="named angles")
        tk.Entry(self.col2, textvariable=self.angles_var).pack(anchor="w")
        self.angles_var.trace_add("write", lambda *args: self._update_angles())

        tk.Label(self.col2, text="PCs:").pack(anchor="w")
        self.pcs_var = tk.StringVar()
        tk.Entry(self.col2, textvariable=self.pcs_var).pack(anchor="w")
        self.pcs_var.trace_add("write", lambda *args: self._update_pcs())

        # ----- COLUMN 3 -----
        # Existing plot input fields
        self._make_plot_input(self.col3, "xlabel", "")
        self._make_plot_input(self.col3, "ylabel", "")
        self._make_plot_input(self.col3, "xlim", "AUTO")
        self._make_plot_input(self.col3, "ylim", "AUTO")
        self._make_plot_input(self.col3, "xticks", "")
        self._make_plot_input(self.col3, "yticks", "")
        self._make_plot_input(self.col3, "figsize", "[4,6]")

        tk.Label(self.col3, text="Color Axis [lo, hi]:").pack(anchor="w", pady=(15, 0))
        self.caxis_var = tk.StringVar(value="[0,1]")
       

        # ---- NEW: Plot selection dropdown ----
        tk.Label(self.col3, text="Choose Plot Type:").pack(anchor="w", pady=(10,0))

        self.plot_var = tk.StringVar(value="1 Volcano")

        choices = ["1 Volcano", "2 pV mat", "3 pV mat", "4 scatter plots"]

        dropdown = ttk.Combobox(
            self.col3, textvariable=self.plot_var, values=choices, state="readonly"
        )
        dropdown.pack(anchor="w")

        dropdown.bind("<<ComboboxSelected>>", self._update_plot_type)

        tk.Button(self.col3, text="Commit", command=self.commit_action).pack(anchor="w", pady=15)
        tk.Button(self.col3, text="Choose Group Colors",
          command=self.open_color_window).pack(anchor="w", pady=5)
        
        tk.Entry(self.col3, textvariable=self.caxis_var).pack(anchor="w")
        self.caxis_var.trace_add("write", lambda *args: self._update_caxis())

        # ----- COLUMN 4 -----
        tk.Button(self.col4, text="Analyze Angles", command=self.analyze_angles).pack(fill="x", pady=5)
        tk.Button(self.col4, text="Run PCA", command=self.run_pca).pack(fill="x", pady=5)
        tk.Button(self.col4, text="Save Path", command=self._select_save_path).pack(fill="x", pady=5)
        tk.Button(self.col4, text="Close", command=self.close).pack(fill="x", pady=5)

        self._root.mainloop()

    # ------------ NEW METHODS ------------
    def open_color_window(self):
        """
        Opens a window that lets the user pick n colors (self.n_groups)
        and stores them as a numpy array in self.group_colors (scaled 0–1).
        """
        if self.n_groups == 0:
            print("No groups defined yet.")
            return
    
        win = tk.Toplevel(self.win)
        win.title("Select Group Colors")
    
        tk.Label(win, text=f"Pick {self.n_groups} colors:",
                 font=("Arial", 11, "bold")).grid(row=0, column=0, columnspan=3, pady=10)
    
        self._color_vars = []     # store hex color strings
        self._color_buttons = []  # store button widgets
    
        def make_row(i):
            tk.Label(win, text=f"Group {i+1}:").grid(row=i+1, column=0, padx=10)
    
            color_var = tk.StringVar(value="#808080")  # default gray
            btn = tk.Button(win, text="Pick color",
                            command=lambda v=color_var: self._pick_color(v))
            btn.grid(row=i+1, column=1, padx=10)
    
            entry = tk.Entry(win, textvariable=color_var, width=10)
            entry.grid(row=i+1, column=2, padx=10)
    
            self._color_vars.append(color_var)
            self._color_buttons.append(btn)
    
        for i in range(self.n_groups):
            make_row(i)
    
        # Confirm button
        tk.Button(win, text="Confirm", bg="#88ff88",
                  command=lambda w=win: self._finalize_colors(w)).grid(
            row=self.n_groups + 1, column=0, columnspan=3, pady=15
        )
    
    
    def _pick_color(self, var):
        """
        Opens tkinter color picker and stores returned hex color in var.
        """
        color = colorchooser.askcolor(title="Choose color")
        if color and color[1]:
            var.set(color[1])  # hex color string
    
    
    def _finalize_colors(self, window):
        """
        Convert picked hex colors to a normalized float numpy array (0–1).
        Save in self.group_colors.
        """
        hex_list = [v.get() for v in self._color_vars]
    
        rgb = []
        for h in hex_list:
            h = h.lstrip("#")
            r = int(h[0:2], 16) / 255
            g = int(h[2:4], 16) / 255
            b = int(h[4:6], 16) / 255
            rgb.append([r, g, b])
    
        self.group_colors = np.array(rgb, dtype=float)
        print("Saved group colors:\n", self.group_colors)
    
        window.destroy()

    def _update_plot_type(self, *args):
        """Extract numeric prefix from selection."""
        txt = self.plot_var.get()
        try:
            num = int(txt.split()[0])
            self.current_plot = num - 1
        except:
            self.current_plot = None
        print("Selected plot type:", self.current_plot)

    def commit_action(self):
        idx = self.current_plot
        
        txt = self.ylim_var.get()
        if txt == 'AUTO':
            self.plot_props[idx]["ylim"] = txt
        else:
            self.plot_props[idx]["ylim"] = self._parse_string(txt)
        
        txt = self.xlim_var.get()
        if txt == 'AUTO':
            self.plot_props[idx]["xlim"] = txt
        else:
            self.plot_props[idx]["xlim"] = self._parse_string(txt)    

        txt = self.xlabel_var.get()
        self.plot_props[idx]["xlabel"] = txt
        
        txt = self.ylabel_var.get()
        self.plot_props[idx]["ylabel"] = txt
        
        txt = self.xticks_var.get()
        self.plot_props[idx]["xticks"] = self._parse_string(txt)
        
        txt = self.yticks_var.get()
        self.plot_props[idx]["yticks"] = self._parse_string(txt)
        
        txt = self.figsize_var.get()
        self.plot_props[idx]["figsize"] = self._parse_string(txt)
        
     
 

    def _parse_string(self,txt):
        try:
            value = self._parse_list_string(txt)
        except:
            value = None
        return value
    
    
    def _update_caxis(self):
        txt = self.caxis_var.get()
        try:
            value = self._parse_list_string(txt)
            self.color_axis = value
        except:
            self.color_axis = None


    def _parse_list_string(self, s):
        try:
            out = ast.literal_eval(s)
            if isinstance(out, (list, tuple)):
                return list(out)
            else:
                return None
        except:
            return None

    def _update_label_col(self, *args):
        self.label_col = self.col_var.get()
        col_values = self.backend.label_db[self.label_col]
        col_values_clean = col_values.dropna().astype(str)
        self.current_unique = np.unique(col_values_clean)
        self._refresh_unique_list()

    def _refresh_unique_list(self):
        self.unique_list.config(state="normal")
        self.unique_list.delete(0, tk.END)
        for item in self.current_unique:
            self.unique_list.insert(tk.END, str(item))
        self.unique_list.config(state="disabled")

    def _make_plot_input(self, parent, name, default):
        tk.Label(parent, text=name).pack(anchor="w")
        var = tk.StringVar(value=default)
        tk.Entry(parent, textvariable=var).pack(anchor="w")
        setattr(self, f"{name}_var", var)
        var.trace_add("write", lambda *args, n=name: self._update_plot_prop(n))

    def _update_conditions(self):
        self.all_columns.append(self.label_col)
        self.unique_grps.append(self.current_unique)
        self.all_groups_var.set(str(self.all_columns))
        grps,_ = self.build_groups()
        displ_grps = []
        for i in grps:
           
            displ_grps.append("-".join(str(j) for j in i.values()))
            
        self.exp_groups_var.set(str(displ_grps))
        self.n_groups = len(grps)

    def _reset_conditions(self):
        self.current_unique = []
        self.unique_grps = []
        self.all_columns = []
        self.n_colors = 0
        self.all_groups_var.set("")
        self.exp_groups_var.set("")

    def _update_angles(self):
        self.angles = self.angles_var.get()

    def _update_pcs(self):
        txt = self.pcs_var.get()
        try:
            value = self._parse_list_string(txt)
            self.pc_combos = value
        except:
            self.pc_combos = None

    def _update_plot_prop(self, name):
        setattr(self, name, getattr(self, f"{name}_var").get())

    # ... rest of your original code unchanged ...


    def build_groups(self):
        groups = []
 
        
        for combo in product(*self.unique_grps):
            items = {}
      
            for idx,i in enumerate(combo):
         

                items[self.all_columns[idx]] = i
            groups.append(items)
        

   
        design_matrix = list(combinations(np.arange(0,len(groups)),2))
 
        return groups,design_matrix
                
                
                
    # Actions
    def analyze_angles(self):
        if self.save_path == '':
            self._select_save_path()
        

        self.backend.plot_path = self.save_path
        self.backend.angle_range = '-pi' 
        self.backend.all_angles()
        folder = 'angle_plots'
        #defining some color variables # hard coded
        finger_colors= [[0.33333333, 0.33333333, 1.],
                 [0.54117647, 0.56470588, 0.88235294],
                 [0.93333333, 1.        , 0.66666667],
                 [0.94117647, 0.56470588, 0.54901961],
                 [0.95294118, 0.04313725, 0.40784314]]

        
        orig_xtick_format = copy.copy(self.backend.plt_prp['xtick_format'])

        # overall differences between injured and non-injured .........................
        
        #groups = np.unique(self.backend.label_db[self.label_col])
        
        data_groups,design_matrix = self.build_groups()

        colors = self.group_colors
        
        if len(data_groups) > colors.shape[0]:
            colors = np.random.random((len(data_groups),3))
        
        comparison_labels = []
        for i in design_matrix:
            name = data_groups[i[0]][self.label_col] + '-vs-' +  data_groups[i[1]][self.label_col]
            comparison_labels.append(name)

        num = int(len(self.backend.label_db)/len(data_groups)*1.5)
        for i in data_groups:
             a,p = self.backend.filter_data(i)
             self.backend.paw_plot(a,err_ang=14,offset=10,max_n=num,
                           headlines=finger_colors,folder=folder,tag='auto')



        # # pairwise hyptothesis tests FDR  
        if self.angles=='named angles':
            angles = self.backend.named_angles
        else:
            angles = int(self.angles)
        
        all_results,plot_data,plot_labels = self.backend.test_all_angles(data_groups,
                                                                  design_matrix
                                                                  ,CI_90=True,folder=folder,tag='-auto')  # opening angle

        # #do the volcano plot and further filtering  
        pps= self.plot_props[0]
        if pps["figsize"] is not None:
            fs = pps["figsize"]
        else: 
            fs = [4,6]
        
        thresholded_results,corrected_results = self.backend.volcano_plot(all_results, 'relative_delta',
                                                                   'qVal', 0.5, 0.05,figsize=fs,folder=folder,tag='-auto')

        # #plot the heatmap for the pValues... 
        self.backend.plt_prp['xtick_format'][0] = 90
        xl = comparison_labels        
        pps= self.plot_props[1]
        if pps["figsize"] is not None:
            fs = pps["figsize"]
        else: 
            fs = [3,6]
     
        if pps["xlabel"] is not None:
            if pps["xlabel"] == 'off':
                xl = []
        else: 
            xl = comparison_labels
       
        
        p_V_mat = self.backend.reshuffle_pvs(corrected_results) 
        self.backend.plot_pvalue_heatmap(p_V_mat,y_labels='auto',x_labels=xl,aspect_ratio=fs,folder=folder,tag='-auto')
        # # plotting only the intuitive (named) angles
        xl = comparison_labels
        pps= self.plot_props[2]
        if pps["figsize"] is not None:
            fs = pps["figsize"]
        else: 
            fs = [7,6]
        if pps["xlabel"] is not None:
            if pps["xlabel"] == 'off':
                xl = []
        else: 
            xl = comparison_labels
        
        p_V_mat = self.backend.reshuffle_pvs(corrected_results,angles=self.backend.named_angles) 
        self.backend.plot_pvalue_heatmap(p_V_mat, y_labels='auto',x_labels=xl,aspect_ratio=fs,folder=folder,tag='-auto')

        # # This plots the paw_mapping, caxis is in revolutions--------------------------
        # # the index, here 1, lets you choose the from your stored paws.
        self.backend.multi_group_paw_mapping(thresholded_results,1,caxis=self.color_axis,folder=folder,tag='-auto') 

        pps= self.plot_props[3]
        if pps["figsize"] is not None:
            fs = pps["figsize"]
        else: 
            fs = [2,6]
        
        if self.angles=='named angles':
            
            plot_props = self.backend.default_plot_props()
            plot_props['xlim'] = 'AUTO'
            plot_props['ylim'] = 'AUTO'
            plot_props['xlabel'] = ''
            plot_props['ylabel'] = 'angle [°]'
            
            plot_props['xtick_format'][0] = 90
            self.backend.plot_n_angles(self.backend.named_angles,plot_data,plot_labels,colors,
                                plot_props=plot_props,figsize=fs,folder=folder,tag='auto')
            
        else:
            plot_props = self.backend.default_plot_props()
            plot_props['xtick_format'][0] = 90
 
            if pps["xlim"] is not None:
                plot_props['xlim'] = pps["xlim"]
            else: 
                plot_props['xlim'] = 'AUTO'
            
            if pps["ylim"] is not None:
                plot_props['ylim'] = pps["ylim"]
            else: 
                plot_props['ylim'] = 'AUTO'
            if pps["ylabel"] is not None:
                plot_props['ylabel'] = pps["ylabel"]
            else: 
                plot_props['ylabel'] = 'angle [°]' 
                
            if pps["xlabel"] is not None:
                plot_props['xlabel'] = pps["xlabel"]
            else: 
                plot_props['xlabel'] = '' 
    
            self.backend.plot_n_angles([angles],plot_data,plot_labels,colors,
                                plot_props=plot_props,figsize=fs,folder=folder,tag='auto')
            
        self.backend.plt_prp['xtick_format'][0] = 0

    def run_pca(self):
        if self.save_path == '':
            self._select_save_path()
        self.backend.plot_path = self.save_path
        folder = 'PCA_plots'
        
        
        # Defining some colors for visualization 
        colors = self.group_colors
        
         #embed()
        labels = self.backend.label_db[self.all_columns].astype(str).agg('-'.join, axis=1)
            
          

        if len(np.unique(labels)) > colors.shape[0]:
            colors = np.random.random((len(np.unique(labels)),3))
            
        self.backend.default_plot_props()
        # normalizing, re-orienting, and flattening keypoint coordinates for PCA  
        pts = self.backend.pts_2_pca(self.backend.pts,nth_point=6,generic_right=True,
                             re_zero_type='mid_line',mirror_type='mid_line')


        if self.pc_combos is not None:
            categories = self.backend.run_pca(pts,
                                      labels,colors=colors,
                                      combos=self.pc_combos,
                                      folder=folder,tag='auto')
        else: 
            # running PCA using paw_statistics + plotting 
            categories = self.backend.run_pca(pts,
                                      labels,colors=colors,folder=folder,tag='auto')
    
    
            #plotting group specific postures that summarize the average group posture
            self.backend.compose_paws(range(30),names=categories,folder=folder,tag='auto')

           


    def _select_save_path(self):
        path = filedialog.askdirectory()
        if path:
            self.save_path = path
            print("Selected save path:", path)

    def close(self):
        try: self.win.destroy()
        except: pass
        try: self._root.destroy()
        except: pass
