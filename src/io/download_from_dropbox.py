# src/io/download_from_dropbox.py

import dropbox
import os
import sys
from src.io.dropbox_utils import refresh_access_token 

# Contract: Only ingest files that the Navier-Stokes solver can process
ALLOWED_EXTENSIONS = [".step", ".stp", ".json", ".zip"]

def download_files_from_dropbox(dropbox_folder, local_folder, refresh_token, client_id, client_secret, log_file_path):
    """Syncs allowed files from cloud to local data/testing-input-output."""
    access_token = refresh_access_token(refresh_token, client_id, client_secret)
    dbx = dropbox.Dropbox(access_token)

    with open(log_file_path, "a") as log_file:
        log_file.write(f"🚀 Syncing from Dropbox: {dropbox_folder}\n")
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
                            log_file.write(f"✅ Downloaded {entry.name} to local storage.\n")
                            print(f"✅ Downloaded: {entry.name}")
                        else:
                            log_file.write(f"⏭ Skipped: {entry.name} (Unsupported Extension)\n")

                has_more = result.has_more
                cursor = result.cursor
            log_file.write("🎉 Ingestion phase completed successfully.\n")
        except Exception as e:
            log_file.write(f"❌ Ingestion Error: {e}\n")
            print(f"❌ Ingestion Error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python download_from_dropbox.py <dbx_f> <local_f> <token> <id> <secret> <log>")
        sys.exit(1)
        
    download_files_from_dropbox(
        sys.argv[1], sys.argv[2], sys.argv[3], 
        sys.argv[4], sys.argv[5], sys.argv[6]
    )