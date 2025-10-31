#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:07:07 2024

@author: wormulon
"""




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 18:02:48 2024

@author: wormulon
"""
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import copy

class interactive_plot_UI:
    def __init__(self, parent, image, points, bbox, connectivity, colors, title="", window_size=None):
        # Create a blocking top-level window
        self.window = tk.Toplevel(parent)
        self.window.transient(parent)  # Keep on top of parent
        self.window.grab_set()         # Make modal (block interaction with parent)
        
        self.initialized = False
        self.window_size = window_size
        self.user_interaction = False
        self.window.title("Interactive Image")

           
        self.original_image = Image.fromarray(np.flip(image,axis=2))  # Store original image
        self.image = self.original_image.copy()
        self.original_points = points.copy()  # Store original points for scaling
        self.points = points
        # Store original bbox for scaling
 
        self.bbox = bbox
        self.bbox[2] = self.bbox[2]-self.bbox[0]
        self.bbox[3] = self.bbox[3]-self.bbox[1]
        self.original_bbox = self.bbox.copy() 
        self.selected_point = None
        self.selected_bbox = False
        self.selected_handle = None
        self.bbox_offset = (0, 0)
        self.connectivity = connectivity
        self.colors = colors
        self.handle_size = 5  # Size of the corner handles for resizing
        self.canvas = tk.Canvas(self.window,width=self.image.width, height=self.image.height)
               
        self.canvas.pack()
        # Add title label
        self.title_label = tk.Label(self.window, text=title, font=("Helvetica", 16))
        self.title_label.pack()
        
        
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.imgtk = ImageTk.PhotoImage(image=self.image)

        self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgtk)
        
        
        
        #properties of coordinate system
        
        #self.im_height_orig = self.image.height
        #self.im_width_orig = self.image.width
        self.im_height_orig = image.shape[0]
        self.im_width_orig = image.shape[1]
        
        self.AR = self.image.height/self.image.width
        self.scaling = 1 
        
        self.draw_points_and_bbox()

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.exit_button = tk.Button(self.window, text="Confirm", command=self.end_it)
        self.exit_button.pack()

       
        self.window.bind("<Configure>", self.on_resize)  # Bind the resize event
        self.window.after(500, self.on_ui_initialized)
        self.window.wait_window(self.window)

    
    
    def end_it(self):

        all_points_within = np.all([
            self.points[:, 0] >= self.bbox[0],
            self.points[:, 0] <= self.bbox[0] + self.bbox[2],
            self.points[:, 1] >= self.bbox[1],
            self.points[:, 1] <= self.bbox[1] + self.bbox[3]
        ], axis=0)

        if not all(all_points_within):
            min_x = np.min(self.points[:, 0])
            max_x = np.max(self.points[:, 0])
            min_y = np.min(self.points[:, 1])
            max_y = np.max(self.points[:, 1])
            self.bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
            
        #self.window.quit()
        self.window.destroy() # Stop the main loop    
    
    def on_ui_initialized(self):
        # This method is called after the UI is fully initialized and visible
        #print("UI has been fully initialized and is now visible.")
        window_width = self.window.winfo_width() + 100
        window_height = self.window.winfo_height() + 100
        screen_height = self.window.winfo_screenheight()
        screen_width = self.window.winfo_screenwidth()
        #print(window_height,window_width,screen_height,screen_width)
        self.initialized = True 
        if self.window_size != None:
           self.window.geometry(f"{self.window_size[0]}x{self.window_size[1]}")
           
        elif screen_height < window_height:
            #print('resizing the window')
            new_height = int(screen_height*0.9)
            new_width = int(window_width*(new_height/window_height))
        
        # # Rescale the window to the new dimensions
            self.window.geometry(f"{new_width}x{new_height}")
        
        
    def draw_points_and_bbox(self):
        self.canvas.delete("points")
        self.canvas.delete("bbox")
        self.canvas.delete("handles")
        #print("Connectivity",self.connectivity)
        #print("points", self.points)
        #self.report_properties()
        for idx, conn in enumerate(self.connectivity):
            sx, sy = self.points[conn[0]]
            ex, ey = self.points[conn[1]]
            self.canvas.create_line(sx, sy, ex, ey, fill=self.colors[idx], width=4, tags="points")
        
        point_colors = []
        searchMat = np.asarray(self.connectivity)
        for i in range(len(self.points)):
            if i == 0:
                point_colors.append(self.colors[len(self.colors)-1])
            else:
                a = np.where(searchMat == i)
                #print('a is',a[0][0])
                point_colors.append(self.colors[int(a[0][0])])
            
        
        for idx,point in enumerate(self.points):
            self.canvas.create_oval(point[0] - 3.5, point[1] - 3.5, point[0] + 3.5, point[1] + 3.5, fill=point_colors[idx], tags="points")
        
   
        x, y, w, h = self.bbox
        self.canvas.create_rectangle(x, y, x + w, y + h, outline="green", tags="bbox")

        # Draw handles for resizing
        self.draw_handles(x, y, w, h)
    
    def on_resize(self, event):

        if self.initialized :
            # Get the new size of the window
            new_height = int(event.height*0.95)
            scale = new_height/self.im_height_orig 
            new_width = int(self.im_width_orig*scale)
            
            if new_height > 200 : 
                self.scaling = scale
                # Resize the image to fit the new window size
                self.image = self.original_image.resize((new_width, new_height),resample=Image.NEAREST)
                # Rescale points and bbox to maintain relative positions
                self.points = (self.original_points * [self.scaling, self.scaling]).astype(int)
                self.bbox = [
                    int(self.original_bbox[0] * self.scaling),
                    int(self.original_bbox[1] * self.scaling),
                    int(self.original_bbox[2] * self.scaling),
                    int(self.original_bbox[3] * self.scaling)
                ]
                # Update the canvas and redraw elements
                self.canvas.config(width=new_width, height=new_height)
                self.imgtk = ImageTk.PhotoImage(image=self.image)
                self.canvas.itemconfig(self.image_id, image=self.imgtk)
                self.draw_points_and_bbox()        


    def draw_handles(self, x, y, w, h):
        handle_coords = [
            (x, y),  # Top-left
            (x + w, y),  # Top-right
            (x, y + h),  # Bottom-left
            (x + w, y + h)  # Bottom-right
        ]
        for hx, hy in handle_coords:
            self.canvas.create_rectangle(hx - self.handle_size, hy - self.handle_size,
                                         hx + self.handle_size, hy + self.handle_size,
                                         outline="blue", fill="blue", tags="handles")

    def on_button_press(self, event):
        
        
        x, y = event.x, event.y

        distances = np.linalg.norm(self.points - np.array([x, y]), axis=1)
        if np.min(distances) < 5:
            self.selected_point = np.argmin(distances)
            self.selected_bbox = False
            self.selected_handle = None
        else:
            self.selected_handle = self.get_selected_handle(x, y)
            if self.selected_handle is not None:
                self.selected_point = None
                self.selected_bbox = False
            else:
                bbox_x, bbox_y, bbox_w, bbox_h = self.bbox
                if (bbox_x <= x <= bbox_x + bbox_w) and (bbox_y <= y <= bbox_y + bbox_h):
                    self.selected_point = None
                    self.selected_bbox = True
                    self.bbox_offset = (x - bbox_x, y - bbox_y)
                else:
                    self.selected_point = None
                    self.selected_bbox = False
                    self.selected_handle = None

    def get_selected_handle(self, x, y):
        bbox_x, bbox_y, bbox_w, bbox_h = self.bbox
        handle_coords = [
            (bbox_x, bbox_y),  # Top-left
            (bbox_x + bbox_w, bbox_y),  # Top-right
            (bbox_x, bbox_y + bbox_h),  # Bottom-left
            (bbox_x + bbox_w, bbox_y + bbox_h)  # Bottom-right
        ]
        for i, (hx, hy) in enumerate(handle_coords):
            if hx - self.handle_size <= x <= hx + self.handle_size and hy - self.handle_size <= y <= hy + self.handle_size:
                return i
        return None

    def on_mouse_drag(self, event):
        if self.selected_point is not None:
            self.points[self.selected_point] = [event.x, event.y]
            self.draw_points_and_bbox()
            #print('saved point')
            self.update_original_data()
            
        elif self.selected_bbox:
            new_x = event.x - self.bbox_offset[0]
            new_y = event.y - self.bbox_offset[1]
            self.bbox = [new_x, new_y, self.bbox[2], self.bbox[3]]
            #print('saved bbox')
            self.draw_points_and_bbox()
            self.update_original_data()
            
        elif self.selected_handle is not None:
            self.resize_bbox(event.x, event.y)
            self.draw_points_and_bbox()
            self.update_original_data()

    def resize_bbox(self, x, y):
        bbox_x, bbox_y, bbox_w, bbox_h = self.bbox
        if self.selected_handle == 0:  # Top-left
            new_w = bbox_w + (bbox_x - x)
            new_h = bbox_h + (bbox_y - y)
            if new_w > 0 and new_h > 0:
                self.bbox = [x, y, new_w, new_h]
        elif self.selected_handle == 1:  # Top-right
            new_w = x - bbox_x
            new_h = bbox_h + (bbox_y - y)
            if new_w > 0 and new_h > 0:
                self.bbox = [bbox_x, y, new_w, new_h]
        elif self.selected_handle == 2:  # Bottom-left
            new_w = bbox_w + (bbox_x - x)
            new_h = y - bbox_y
            if new_w > 0 and new_h > 0:
                self.bbox = [x, bbox_y, new_w, new_h]
        elif self.selected_handle == 3:  # Bottom-right
            new_w = x - bbox_x
            new_h = y - bbox_y
            if new_w > 0 and new_h > 0:
                self.bbox = [bbox_x, bbox_y, new_w, new_h]
        #print('saved box size change')
        self.update_original_data()

    def on_button_release(self, event):
        self.user_interaction = True
        #print('USER INPUT ##################################################')
        if self.selected_point is not None:
            self.points[self.selected_point] = [event.x, event.y]
            self.selected_point = None
            self.draw_points_and_bbox()
        elif self.selected_bbox:
            self.selected_bbox = False
            self.draw_points_and_bbox()
        elif self.selected_handle is not None:
            self.selected_handle = None
            self.draw_points_and_bbox()

    def update_original_data(self):
        self.original_points = (self.points / self.scaling).astype(int)
        self.original_bbox = [
            int(self.bbox[0] / self.scaling),
            int(self.bbox[1] / self.scaling),
            int(self.bbox[2] / self.scaling),
            int(self.bbox[3] / self.scaling)
            ]        
    def report_properties(self):
        print('####################################################')
        print('scale',self.scaling)
        print('orig_im_height',self.im_height_orig)
        print('orig_im_width',self.im_width_orig)
        print('im_height',self.image.height)
        print('im_width',self.image.width)
        print('bbox',self.bbox)
        print('pts',self.points)
        print('orig_pts',self.original_points)
        


    def return_data(self):
        
        self.bbox[2] = self.bbox[2] + self.bbox[0]
        self.bbox[3] = self.bbox[3] + self.bbox[1]
        self.bbox = np.asarray(self.bbox)/self.scaling
        self.points = np.asarray(self.points)/self.scaling
        
        
        if self.user_interaction:
            return self.points, self.bbox
        else:
            return self.original_points, self.original_bbox
    
   