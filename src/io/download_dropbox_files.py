# src/io/download_dropbox_files.py

import dropbox
import os
import sys
# Changed import to reflect relocated directory
from src.io.dropbox_utils import refresh_access_token 

ALLOWED_EXTENSIONS = [".step", ".stp", ".json", ".zip"]

def download_files_from_dropbox(dropbox_folder, local_folder, refresh_token, client_id, client_secret, log_file_path):
    access_token = refresh_access_token(refresh_token, client_id, client_secret)
    dbx = dropbox.Dropbox(access_token)

    with open(log_file_path, "a") as log_file:
        log_file.write("🚀 Starting download process...\n")
        try:
            os.makedirs(local_folder, exist_ok=True)
            has_more = True
            cursor = None
            while has_more:
                result = (
                    dbx.files_list_folder_continue(cursor)
                    if cursor else
                    dbx.files_list_folder(dropbox_folder)
                )
                
                for entry in result.entries:
                    if isinstance(entry, dropbox.files.FileMetadata):
                        ext = os.path.splitext(entry.name)[1].lower()
                        if ext in ALLOWED_EXTENSIONS:
                            local_path = os.path.join(local_folder, entry.name)
                            with open(local_path, "wb") as f:
                                _, res = dbx.files_download(path=entry.path_lower)
                                f.write(res.content)
                            log_file.write(f"✅ Downloaded {entry.name}\n")
                            print(f"✅ Downloaded: {entry.name}")

                has_more = result.has_more
                cursor = result.cursor
            log_file.write("🎉 Download completed.\n")
        except Exception as e:
            log_file.write(f"❌ Error: {e}\n")
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Standard arg parsing (as provided in your snippet)
    download_files_from_dropbox(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])