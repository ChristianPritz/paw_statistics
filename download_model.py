#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import requests

def fetch_from_osf(token: str, osf_resource_id: str, destination: str):
    """
    Recursively download all files from an OSF project or folder.
    
    Args:
        token (str): Your OSF personal access token.
        osf_resource_id (str): The OSF resource ID (e.g., 'dc745').
        destination (str): Local directory to save the files.
    """
    headers = {"Authorization": f"Bearer {token}"}
    base_url = f"https://files.osf.io/v1/resources/{osf_resource_id}/providers/osfstorage/"

    def download_folder(folder_url, local_path):
        os.makedirs(local_path, exist_ok=True)
        response = requests.get(folder_url, headers=headers)
        response.raise_for_status()
        data = response.json()

        for entry in data["data"]:
            attrs = entry["attributes"]
            links = entry["links"]
            name = attrs["name"]
            kind = attrs["kind"]

            if kind == "folder":
                # Recurse into subfolder
                subfolder_url = links["new_folder"].split("?")[0]
                download_folder(subfolder_url, os.path.join(local_path, name))
            elif kind == "file":
                # Download file
                download_url = links.get("download") or links["self"] + "?action=download"
                file_path = os.path.join(local_path, name)
                print(f"⬇️ Downloading: {file_path}")
                with requests.get(download_url, headers=headers, stream=True) as r:
                    r.raise_for_status()
                    with open(file_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            else:
                print(f"⚠️ Unknown entry type: {kind} ({name})")

    download_folder(base_url, destination)
    print("✅ Download complete!")

if __name__ == "__main__":
    # Hardcoded arguments
    OSF_TOKEN = "BPQqLG9Cz30LE2qJsFRrujRmgn1hYrKJ0ndGaqq9BrwZibxefiPIaP1ZsOugnCcT7kKlIF"
    OSF_RESOURCE_ID = "dc745"  # replace with your actual project or component ID
    DESTINATION =  os.getcwd()

    # Call the download function
    fetch_from_osf(OSF_TOKEN, OSF_RESOURCE_ID, DESTINATION)
