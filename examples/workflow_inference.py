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

from DataFrameViewerUI import DataFrameViewerUI
import os,json


def dump_dict_2_json(dt,path=None):
    if path is None:
        path = str(os.getcwd()) + '/meta_data.json'
    with open(path, "w") as f:
        json.dump(dt, f,indent=4)


#Define your experimental labels that will be saved alongside the data. 
meta_data = {
    "treatment": [
        "SURGERY",
        "CONTROL"
    ],
    "animal_id": "",
    "DOB": "01.01.26",
    "gender": [
        "MALE",
        "FEMALE"
    ],
    "genotype": [
        "WT",
        "IMPA3KO"
    ],
    "strain": "C57BL/6",
    "side": [
        "right",
        "left"
    ],
    "pain_status": [
        "pain",
        "no_pain",
        "accute_pain",
        "recovered"
    ],
    "paw_posture": [
        "clenched",
        "open",
        "closed"
    ],
    "orientation": [
        "good",
        "ok",
        "useless",
        "impossible"
    ],
    "useful": [
        "yes",
        "no"
    ]
}    
dump_dict_2_json(meta_data)


# To run inference from images, first copy all images into a single directory.

# Launch DataFrameViewerUI and use the UI elements to infer paw keypoints.

# Click on "Add paws" to infer keypoints from images or videos:
#   - Provide the path to the meta_data.json file (created above)
#   - Provide the path to the image directory or video file
#   - Provide an output directory where cropped paw images and the exported
#     data (ZIP format) will be saved
#   - Select the hardware for inference:
#         * Choose "cuda" if you have an NVIDIA GPU
#         * Choose "cpu" otherwise
#   - Use the UI to run inference and correct keypoints if necessary

# "Correct entry" allows you to manually correct labeling or keypoint positions
# "Delete entry" removes faulty or unwanted entries
# "Merge with existing data" loads an existing ZIP file and merges it with the
#   data currently displayed in the UI
# "Save data" exports all data currently in the UI as a ZIP file, including
#   metadata, keypoints, and bounding boxes
# "Close" terminates the UI

DataFrameViewerUI()
    

