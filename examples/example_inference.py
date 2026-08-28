#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 21:44:08 2026

@author: C.Pritz
"""
# -------------------------------------------------------------------------
# imports
# -------------------------------------------------------------------------
import json
from pathlib import Path
from paw_UI import ImageSequenceExporter
from DataFrameViewerUI import DataFrameViewerUI

# -------------------------------------------------------------------------
# helper functions 
# -------------------------------------------------------------------------


def dump_dict_to_json(data, path):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return path


# -------------------------------------------------------------------------
# Define your experimental metadata
# -------------------------------------------------------------------------

METADATA = {
    "treatment": ["SURGERY", "CONTROL"],
    "animal_id": "",
    "DOB": "01.01.26",
    "gender": ["male", "female"],
    "genotype": ["wt", "IMPA3KO"],
    "strain": "C57BL/6",
    "side": ["right", "left"],
    "pain_status": ["pain", "no_pain", "accute_pain", "recovered"],
    "paw_posture": ["clenched", "open", "closed"],
    "orientation": ["good", "ok", "useless", "impossible"],
    "useful": ["yes", "no"],
    "ant_or_post": ["post", "ant"]
}
# Adding a list as the value for a key creates a dropdown menu:
# "treatment": ["SURGERY", "CONTROL"] allows the user to select either
# "SURGERY" or "CONTROL".
#
# Adding a string as the value creates a free-text input field, allowing
# the user to enter any value manually.


metadata_save_path = "PLEASE_SPECIFY_A_SAVEPATH/metadata.json"
dump_dict_to_json(METADATA, metadata_save_path) 
#When pressing "Add paws" button you will be prompted to specify the path to 
#your experimental metadata. 


# -------------------------------------------------------------------------
# Running the datamanagement and predition UI
# -------------------------------------------------------------------------

DataFrameViewerUI()








