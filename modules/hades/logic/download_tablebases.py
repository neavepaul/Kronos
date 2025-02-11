import requests
import re
import os
from pathlib import Path

# Syzygy Tablebase URL
SYZYGY_URL = "http://tablebase.sesse.net/syzygy/3-4-5/"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# Set directory to store Syzygy DTZ tablebases
ROOT_PATH = Path(__file__).resolve().parents[3]  # Moves up to "Kronos/"
SYZYGY_DIR = ROOT_PATH / "modules/hades/logic/syzygy/345"
os.makedirs(SYZYGY_DIR, exist_ok=True)

def get_syzygy_file_list():
    """Fetches Syzygy 6-DTZ file list."""
    print("🔍 Fetching file list from Syzygy server...")
    headers = {"User-Agent": USER_AGENT}
    
    response = requests.get(SYZYGY_URL, headers=headers)
    if response.status_code != 200:
        print(f"❌ Error: Unable to access {SYZYGY_URL} (Status {response.status_code})")
        return []
    
    file_names = re.findall(r'href="([^"]+\.(?:rtbw|rtbz))"', response.text)

    if not file_names:
        print("⚠️ No tablebase files found. Server structure may have changed.")
        return []

    print(f"✅ Found {len(file_names)} tablebase files.")
    return file_names

def download_file(file_name):
    """Downloads a single file with a browser-mimicking User-Agent."""
    file_url = f"{SYZYGY_URL}{file_name}"
    file_path = SYZYGY_DIR / file_name

    if file_path.exists():
        print(f"✅ File already exists: {file_name}")
        return

    print(f"⬇️ Downloading {file_name} ...")
    headers = {"User-Agent": USER_AGENT}
    
    with requests.get(file_url, headers=headers, stream=True) as r:
        if r.status_code == 403:
            print(f"🚫 403 Forbidden: Server is blocking this request for {file_name}")
            return
        
        r.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    print(f"✅ Download complete: {file_name}")

def main():
    """Fetches Syzygy file list and downloads them sequentially."""
    files = get_syzygy_file_list()
    
    if not files:
        print("⚠️ No files to download. Exiting.")
        return

    for file_name in files:
        download_file(file_name)

if __name__ == "__main__":
    main()
