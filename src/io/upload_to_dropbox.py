# src/io/upload_to_dropbox.py

import dropbox
import os
import sys
from src.io.dropbox_utils import refresh_access_token 

def upload_file_to_dropbox(local_file_path, dropbox_folder, refresh_token, client_id, client_secret):
    """Uploads a local file to a specified path on Dropbox."""
    try:
        access_token = refresh_access_token(refresh_token, client_id, client_secret)
        dbx = dropbox.Dropbox(access_token)
        
        output_file_name = os.path.basename(local_file_path)
        dropbox_file_path = f"{dropbox_folder}/{output_file_name}"

        with open(local_file_path, "rb") as f:
            dbx.files_upload(f.read(), dropbox_file_path, mode=dropbox.files.WriteMode.overwrite)
        
        print(f"✅ Successfully uploaded to Dropbox: {dropbox_file_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to upload '{local_file_path}': {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python upload_to_dropbox.py <local_path> <dbx_folder> <token> <id> <secret>")
        sys.exit(1)

    local_path, dbx_folder, token, client_id, secret = sys.argv[1:]
    
    if not os.path.exists(local_path):
        print(f"❌ Error: Local file '{local_path}' not found.")
        sys.exit(1)

    success = upload_file_to_dropbox(local_path, dbx_folder, token, client_id, secret)
    sys.exit(0 if success else 1)