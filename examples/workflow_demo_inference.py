# -*- coding: utf-8 -*-
"""
Spyder Editor

Paw segmentation demo - Christian Pritz 2025
"""
#-----------------------------------------------------------------------------#
#                                I M P O R T S 
#-----------------------------------------------------------------------------#

from ImageSequenceExporter import ImageSequenceExporter
from tkinter import filedialog
import os
import numpy as np



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
#model_path = '/home/wormulon/Documents/trained models/paw_model_reduced/model_torch.pt'
model_path = './model/model_torch.pt'
device = 'cuda' #'cuda' for a GPU otherwise 'cpu' 


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
metadata = {"treatment": ["CAPSAICIN_25","vehicle_control","CAPSAICIN_INJECTED","NO_INJECTION"],
            "animal_id": prefix,
            "DOB": DOB,
            "gender":["MALE","FEMALE"],
            "genotype":["WT","KO"],
            "strain":"C57BL/6",
            "side": ["right", "left"],
            "pain_status":["pain","no_pain","accute_pain","recovered"],
            "paw_posture":["clenched","open","closed"]}

#-----------------------------------------------------------------------------#
#                   I M P O R T   F R O M   V I D E O S 
#-----------------------------------------------------------------------------#



# Specify the filepath of the video file
video_path = './examples/demo_video/example_movie.MOV'

# OPTIONAL: use an UI for finding the movie 
# video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("MOV Files", "*.MOV")])

#specify an output path for saving exported images and data
output_path = '/examples/demo_video/output'
if not os.path.isdir(output_path):
    os.mkdir(output_path)


exporter = ImageSequenceExporter(video_path, metadata,detector_settings,
                                 width=500,prefix=prefix,output_dir=output_path)

# if you want to continue adding paws to an existing database use the paw_stats
# argument: 
    
path_to_zip = './your/path/to/zipfile.zip'

exporter = ImageSequenceExporter(video_path,metadata,detector_settings,
                                 width=500,prefix=prefix,
                                 output_dir=output_path,
                                 paw_stats=path_to_zip)



#-----------------------------------------------------------------------------#
#                     I M P O R T   F R O M   I M A G E S 
#-----------------------------------------------------------------------------#


# SET UP THE META DATA AS IN THE VIDEO EXAMPLE 

#image_path = filedialog.askdirectory(title="Select image Folder")
image_path ='./examples/demo_images/'
output_path = './examples/demo_images/output'
if not os.path.isdir(output_path):
    os.mkdir(output_path)
exporter = ImageSequenceExporter(image_path, metadata,detector_settings,width=500,prefix=prefix,output_dir=output_path)

#Adding data to an existing file 
zip_path = './your/path/to/your_file.zip'
exporter = ImageSequenceExporter(image_path, metadata,detector_settings,width=500,prefix=prefix,output_dir=output_path,paw_stats=zip_path)



#-----------------------------------------------------------------------------#
#            P L O T T I NG   F R O M   S E G M E N T E D   P A W S
#-----------------------------------------------------------------------------#

exporter.paw_stats.label_db.predicted_side.iloc[0]
exporter.paw_stats.plot_single_prediction(1)
