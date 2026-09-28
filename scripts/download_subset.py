"""Downloads the selected 400-image SIIM-ACR subset from the re-hosted
Kaggle dataset (jesperdramsch/siim-acr-pneumothorax-segmentation-data),
which preserves original SOPInstanceUID filenames matching stage_2_train.csv
ImageIds -- unlike the official competition's stage_2_images/ files, which
were re-anonymized and don't match the label CSV.
"""
import json
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parent.parent
SUBSET_DIR = ROOT / "data" / "siim_subset"
DICOM_DIR = SUBSET_DIR / "dicom"
DICOM_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "jesperdramsch/siim-acr-pneumothorax-segmentation-data"


def main():
    with open(SUBSET_DIR / "selected_uids.json") as f:
        selection = json.load(f)
    path_by_uid = selection["path_by_uid"]

    api = KaggleApi()
    api.authenticate()

    total = len(path_by_uid)
    done = 0
    failed = []
    for uid, remote_path in path_by_uid.items():
        out_file = DICOM_DIR / f"{uid}.dcm"
        if out_file.exists():
            done += 1
            continue
        for attempt in range(5):
            try:
                api.dataset_download_file(
                    DATASET, remote_path, path=str(DICOM_DIR), force=True, quiet=True
                )
                downloaded = DICOM_DIR / Path(remote_path).name
                if downloaded != out_file and downloaded.exists():
                    downloaded.rename(out_file)
                break
            except Exception as e:
                wait = 3 * (attempt + 1)
                print(f"[{uid}] attempt {attempt+1} failed ({e}); retrying in {wait}s")
                time.sleep(wait)
        else:
            failed.append(uid)
            continue
        done += 1
        if done % 25 == 0 or done == total:
            print(f"{done}/{total} downloaded")

    print(f"DONE. {done}/{total} succeeded. {len(failed)} failed: {failed}")


if __name__ == "__main__":
    main()
