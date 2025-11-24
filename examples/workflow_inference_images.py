# -*- coding: utf-8 -*-
"""
Spyder Editor

Paw segmentation demo - Christian Pritz 2025

This is a demo script for inference from images. 
Please copy all images into a single folder. The sript will ask you for the 
image folder location. 

"""
#-----------------------------------------------------------------------------#
#                   I M P O R T S   &   F U N C T I O N S
#-----------------------------------------------------------------------------#

from ImageSequenceExporter import ImageSequenceExporter
import os
import tkinter as tk
from tkinter import filedialog

def ask_for_directory(destination):
    root = tk.Tk()
    root.withdraw() 
    directory = filedialog.askdirectory(title="Select an " + destination)
    root.destroy()
    return directory

def ask_for_pt_file(extension):
    root = tk.Tk()
    root.withdraw()

    filepath = filedialog.askopenfilename(
        title="Select the model file:  " + extension  +" file",
        filetypes=[("targetfile", "*"+extension), ("All files", "*.*")]
    )
    root.destroy()
    return filepath



# IMPORTANT NOTE:
# -----------------------------------------------------------------------------
# This UI tool allows you to collect corrected predictions and metadata from
# images and videos, and saves them into a database.
#
# IMPORTANT: THE UI DOES NOT REMEMBER PREVIOUSLY SAVED PREDICTIONS DURING RUNTIME.
# Navigating back to earlier predictions is not possible because the UI works
# strictly in a feed-forward manner.
#
# Workflow:
#     IMAGE --> prediction --> UI: correction + labels --> DATABASE ENTRY
#
# Note that every click on "Export cropped" creates a new database entry.
# Be careful to avoid creating duplicate entries.
# ----------------------------------------------------------------------------
 



# this is the path to the model_troch.pt file adjust it accordingly
# you can use the UI to locate the model or supply a static folder path as string.  

model_path = './model/model_torch.pt'
# UI support: uncomment the following two lines. 
model_path = '/home/wormulon/Documents/trained models/paw_model_reduced/model_torch.pt'
model_path = ask_for_pt_file('.pt')


device = 'cuda' #'cuda' is you have a NVIDIA graphics card otherwise otherwise 'cpu' 


# these are settings for the keypoints and the detector------------------------
detector_settings = {'threshold':0.95,'connect_logic': [[0,     1],
                 [0,     3],
                 [0,     6],
                 [0,     9],
                 [0,    12],
                 [1,     2],
                 [3,     4],
                 [4,     5],
                 [6,     7],
                 [7,     8],
                 [9,    10],
                 [10,   11],
                 [12,   13],
                 [13,   14],
                 [1,     3],
                 [3,     6],
                 [6,     9],
                 [9,    12]],
                     
            'colors_ui': ['#5555ff',
                 '#8a90e1',
                 '#eeffaa',
                 '#f0908c',
                 '#f30b68',
                 '#5555ff',
                 '#8a90e1',
                 '#8a90e1',
                 '#eeffaa',
                 '#eeffaa',
                '#f0908c',
                '#f0908c',
                '#f30b68',
                '#f30b68',
                '#676c53',
                '#676c53',
                '#676c53',
                '#676c53'], 
                     
            'keypoint_names': ["heel","base_i","tip_i","base_ii",
                  "phal_ii","tip_ii","base_iii","phal_iii",
                  "tip_iii","base_iv","phal_iv","tip_iv",
                  "base_v","phal_v","tip_v"],
            "model_path":model_path,
            "device":device} 




#SETTING UP THE META DATA FOR LABELLING----------------------------------------
# Here you can specify the labels for your experimental groups such as 
# age, sex, experimental treatment, etc... 

prefix = 'mouse_' # prefix for animals 
DOB = "01.01.26" # Day of birth default value
 
# this defines your labels, each keyword is a label category, 
#each list the possible labels
metadata = {"treatment": ["treatment 1","treatment 2","treatment 3","control"],
            "animal_id": prefix,
            "DOB": DOB,
            "gender":["MALE","FEMALE"],
            "genotype":["WT","KO"],
            "strain":"C57BL/6",
            "comment": "",
            "side": ["right", "left"],
            "pain_status":["pain","no_pain","accute_pain","recovered"],
            "paw_posture":["clenched","open","closed"]}

#-----------------------------------------------------------------------------#
#                     I M P O R T   F R O M   I M A G E S 
#-----------------------------------------------------------------------------#


# SET UP THE META DATA AS IN THE VIDEO EXAMPLE 

#image_path = filedialog.askdirectory(title="Select image Folder")
image_path = ask_for_directory("image directory")
exporter = ImageSequenceExporter(image_path, metadata,detector_settings,
                                 width=500,prefix=prefix)


## in case you want to continue working on an existing file uncomment the 
# following lines: 

#exporter = ImageSequenceExporter(image_path, metadata,detector_settings,
#                                  width=500,prefix=prefix,
#                                  paw_stats="")
    


