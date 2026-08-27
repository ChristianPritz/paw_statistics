#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 21:44:08 2026

@author: C.Pritz
"""
# -------------------------------------------------------------------------
# imports
# -------------------------------------------------------------------------

import os 
os.chdir('/home/dominik/models/paw_bench/code')
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


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def require_existing_path(label, path, must_be_file=False):
    path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist:\n{path}")

    if must_be_file and not path.is_file():
        raise FileNotFoundError(f"{label} is not a file:\n{path}")

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


dump_dict_to_json(METADATA, "examples/metadata.json") 
#so it can be loaded by ImageSequenceImporter instance


# -------------------------------------------------------------------------
# Running the datamanagement and predition UI
# -------------------------------------------------------------------------

DataFrameViewerUI()








