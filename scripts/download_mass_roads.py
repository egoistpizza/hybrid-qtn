import os
import shutil
import sys
import zipfile
from pathlib import Path

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from tqdm import tqdm
except ImportError:
    sys.exit("Error: 'kaggle' and 'tqdm' packages are required.")

def main():
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        sys.exit("Error: KAGGLE_USERNAME and KAGGLE_KEY environment variables must be set.")

    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "mass_roads"
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    tmp_dir = data_dir / "tmp"

    for directory in [images_dir, masks_dir, tmp_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(
        "balraj98/massachusetts-roads-dataset",
        path=tmp_dir,
        unzip=False,
        quiet=False
    )

    zip_path = tmp_dir / "massachusetts-roads-dataset.zip"
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.infolist()
        for member in tqdm(members, desc="Extracting", unit="file"):
            zip_ref.extract(member, tmp_dir)

    zip_path.unlink()

    files_to_move = [f for f in tmp_dir.rglob("*.*") if f.is_file()]
    
    for file_path in tqdm(files_to_move, desc="Organizing", unit="file"):
        ext = file_path.suffix.lower()
        name = file_path.name
        dest = None
        
        if ext == ".tiff":
            dest = images_dir / name
        elif ext == ".tif":
            dest = masks_dir / name
        elif ext == ".png":
            if "mask" in [p.lower() for p in file_path.parts] or "map" in name.lower():
                dest = masks_dir / name
            else:
                dest = images_dir / name

        if dest and not dest.exists():
            shutil.move(str(file_path), str(dest))

    shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
