"""
Builds notebooks/train_colab.ipynb by embedding the current contents of
dicom_seg/*.py as %%writefile cells, so the Colab notebook always runs the
exact same, already-tested code as this repo -- not a hand-copied,
potentially-drifted version.

Run locally whenever dicom_seg/ changes and the notebook needs updating:
    .venv/Scripts/python.exe scripts/build_colab_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = ROOT / "dicom_seg"
NOTEBOOK_PATH = ROOT / "notebooks" / "train_colab.ipynb"

MODULE_FILES = [
    "__init__.py",
    "preprocessing.py",
    "rle.py",
    "dataset.py",
    "unet.py",
    "losses.py",
    "metrics.py",
    "baseline.py",
    "gradcam.py",
]


def md(*lines):
    # Each argument already ends with "\n" (except conventionally the last),
    # matching nbformat's source-list convention -- don't add another.
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": list(lines)}


def writefile_cell(rel_path: str, content: str):
    header = f"%%writefile {rel_path}\n"
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [header, content]}


def build():
    cells = []

    cells.append(md(
        "# Medical Image Segmentation - Colab Training\n",
        "\n",
        "Trains the U-Net from the `Medical Imaging` project on the SIIM-ACR "
        "pneumothorax subset (the same 400 images used for the local naive-baseline "
        "benchmark, so results are directly comparable), evaluates Dice/IoU against "
        "that baseline, and generates Seg-Grad-CAM explainability examples.\n",
        "\n",
        "**Before running:** `Runtime > Change runtime type > T4 GPU`.\n",
        "\n",
        "**You will need:** your `kaggle.json` API token (Kaggle > Settings > API > "
        "Create New Token), and to have joined/accepted the rules for the "
        "`siim-acr-pneumothorax-segmentation` competition at kaggle.com (even though "
        "images come from a re-hosted dataset, the label CSV comes from the "
        "competition itself)."
    ))

    cells.append(code("!nvidia-smi"))

    cells.append(md("## Setup"))
    cells.append(code("!pip install -q pydicom kaggle"))

    cells.append(code(
        "import os\n",
        "\n",
        "os.makedirs(os.path.expanduser(\"~/.kaggle\"), exist_ok=True)\n",
        "kaggle_path = os.path.expanduser(\"~/.kaggle/kaggle.json\")\n",
        "if not os.path.exists(kaggle_path):\n",
        "    from google.colab import files\n",
        "    print(\"Upload your kaggle.json (Kaggle > Settings > Create New Token):\")\n",
        "    uploaded = files.upload()\n",
        "    fname = list(uploaded.keys())[0]\n",
        "    os.rename(fname, kaggle_path)\n",
        "os.chmod(kaggle_path, 0o600)\n",
        "print(\"Kaggle credentials ready.\")"
    ))

    cells.append(md(
        "## Recreate the `dicom_seg` package\n",
        "\n",
        "These cells write out the exact same, already-tested modules from the local "
        "repo (`dicom_seg/`), via `%%writefile`, so training runs identical code to "
        "what `pytest` verified locally."
    ))
    cells.append(code("!mkdir -p dicom_seg"))

    for filename in MODULE_FILES:
        content = (PKG_DIR / filename).read_text(encoding="utf-8")
        cells.append(writefile_cell(f"dicom_seg/{filename}", content))

    cells.append(md(
        "## Download the same 400-image subset used locally\n",
        "\n",
        "Same UID selection (seed 42, 200 positive / 200 negative) as "
        "`data/siim_subset/selected_uids.json` in the local repo -- so the "
        "train/val split, and therefore the baseline comparison in "
        "`BENCHMARKS.md`, line up exactly."
    ))

    cells.append(code("!mkdir -p data/siim_subset"))

    compact_uids = json.loads((ROOT / "data" / "siim_subset" / "selected_uids_compact.json").read_text())
    cells.append(writefile_cell(
        "data/selected_uids_compact.json",
        json.dumps(compact_uids)
    ))

    cells.append(code(
        "!kaggle competitions download -c siim-acr-pneumothorax-segmentation "
        "-f stage_2_train.csv -p data/siim_subset\n",
        "!cd data/siim_subset && unzip -o -q stage_2_train.csv.zip && rm -f stage_2_train.csv.zip\n",
        "print(\"Labels CSV ready.\")"
    ))

    cells.append(code(
        "import json\n",
        "import time\n",
        "from pathlib import Path\n",
        "\n",
        "from kaggle.api.kaggle_api_extended import KaggleApi\n",
        "\n",
        "DATASET = \"jesperdramsch/siim-acr-pneumothorax-segmentation-data\"\n",
        "\n",
        "with open(\"data/selected_uids_compact.json\") as f:\n",
        "    sel = json.load(f)\n",
        "selected_uids = set(sel[\"positive\"] + sel[\"negative\"])\n",
        "print(f\"Selected {len(selected_uids)} images to download.\")\n",
        "\n",
        "api = KaggleApi()\n",
        "api.authenticate()\n",
        "\n",
        "token = None\n",
        "path_by_uid = {}\n",
        "while True:\n",
        "    for attempt in range(6):\n",
        "        try:\n",
        "            resp = api.dataset_list_files(DATASET, page_token=token, page_size=200)\n",
        "            break\n",
        "        except Exception as e:\n",
        "            wait = 5 * (attempt + 1)\n",
        "            print(f\"retry after {wait}s: {e}\")\n",
        "            time.sleep(wait)\n",
        "    else:\n",
        "        raise RuntimeError(\"failed to list dataset files after retries\")\n",
        "    for f in resp.files:\n",
        "        if f.name.startswith(\"dicom-images-train/\"):\n",
        "            uid = f.name.rsplit(\"/\", 1)[-1][:-4]\n",
        "            if uid in selected_uids:\n",
        "                path_by_uid[uid] = f.name\n",
        "    token = resp.next_page_token\n",
        "    if not token or len(path_by_uid) >= len(selected_uids):\n",
        "        break\n",
        "    time.sleep(1.0)\n",
        "\n",
        "print(f\"Matched {len(path_by_uid)}/{len(selected_uids)} selected files in the dataset listing.\")"
    ))

    cells.append(code(
        "DICOM_DIR = Path(\"data/siim_subset/dicom\")\n",
        "DICOM_DIR.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "done = 0\n",
        "for uid, remote_path in path_by_uid.items():\n",
        "    out_file = DICOM_DIR / f\"{uid}.dcm\"\n",
        "    if out_file.exists():\n",
        "        done += 1\n",
        "        continue\n",
        "    for attempt in range(5):\n",
        "        try:\n",
        "            api.dataset_download_file(DATASET, remote_path, path=str(DICOM_DIR), force=True, quiet=True)\n",
        "            downloaded = DICOM_DIR / Path(remote_path).name\n",
        "            if downloaded != out_file and downloaded.exists():\n",
        "                downloaded.rename(out_file)\n",
        "            break\n",
        "        except Exception as e:\n",
        "            time.sleep(3 * (attempt + 1))\n",
        "    done += 1\n",
        "    if done % 50 == 0:\n",
        "        print(f\"{done}/{len(path_by_uid)}\")\n",
        "\n",
        "print(\"Downloaded:\", len(list(DICOM_DIR.glob(\"*.dcm\"))), \"files\")"
    ))

    cells.append(md("## Build the dataset and dataloaders"))
    cells.append(code(
        "import sys\n",
        "sys.path.insert(0, \".\")\n",
        "\n",
        "from torch.utils.data import DataLoader\n",
        "from dicom_seg.dataset import build_manifest, train_val_split, SIIMPneumothoraxDataset\n",
        "\n",
        "CSV_PATH = Path(\"data/siim_subset/stage_2_train.csv\")\n",
        "\n",
        "entries = build_manifest(CSV_PATH, DICOM_DIR)\n",
        "train_entries, val_entries = train_val_split(entries, val_fraction=0.2, seed=42)\n",
        "print(f\"{len(entries)} images -> {len(train_entries)} train / {len(val_entries)} val\")\n",
        "print(f\"Train positive: {sum(e.has_mask for e in train_entries)}, \"\n",
        "      f\"Val positive: {sum(e.has_mask for e in val_entries)}\")\n",
        "\n",
        "train_ds = SIIMPneumothoraxDataset(train_entries, target_size=256)\n",
        "val_ds = SIIMPneumothoraxDataset(val_entries, target_size=256)\n",
        "train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)\n",
        "val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)"
    ))

    cells.append(md(
        "## Mount Google Drive (for checkpoints)\n",
        "\n",
        "Checkpoints are saved periodically so a disconnected Colab runtime doesn't "
        "lose training progress."
    ))
    cells.append(code(
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "CHECKPOINT_DIR = Path(\"/content/drive/MyDrive/medical_imaging_checkpoints\")\n",
        "CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)\n",
        "print(\"Checkpoints will be saved to:\", CHECKPOINT_DIR)"
    ))

    cells.append(md(
        "## Train\n",
        "\n",
        "Combined Dice+BCE loss (Milestone 5), Adam at lr=1e-4. Tracks train/val loss "
        "and val Dice/IoU every epoch to watch for overfitting."
    ))
    cells.append(code(
        "import torch\n",
        "from torch.optim import Adam\n",
        "\n",
        "from dicom_seg.unet import UNet\n",
        "from dicom_seg.losses import DiceBCELoss\n",
        "from dicom_seg.metrics import dice_coefficient, iou_score\n",
        "\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "print(\"device:\", device)\n",
        "\n",
        "model = UNet(in_channels=1, out_channels=1).to(device)\n",
        "criterion = DiceBCELoss()\n",
        "optimizer = Adam(model.parameters(), lr=1e-4)\n",
        "\n",
        "EPOCHS = 30\n",
        "history = {\"train_loss\": [], \"val_loss\": [], \"val_dice\": [], \"val_iou\": []}\n",
        "\n",
        "for epoch in range(EPOCHS):\n",
        "    model.train()\n",
        "    train_loss = 0.0\n",
        "    for images, masks in train_loader:\n",
        "        images, masks = images.to(device), masks.to(device)\n",
        "        optimizer.zero_grad()\n",
        "        preds = model(images)\n",
        "        loss = criterion(preds, masks)\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "        train_loss += loss.item() * images.size(0)\n",
        "    train_loss /= len(train_ds)\n",
        "\n",
        "    model.eval()\n",
        "    val_loss = 0.0\n",
        "    dice_scores, iou_scores = [], []\n",
        "    with torch.no_grad():\n",
        "        for images, masks in val_loader:\n",
        "            images, masks = images.to(device), masks.to(device)\n",
        "            preds = model(images)\n",
        "            loss = criterion(preds, masks)\n",
        "            val_loss += loss.item() * images.size(0)\n",
        "            for p, m in zip(preds, masks):\n",
        "                dice_scores.append(dice_coefficient(p, m))\n",
        "                iou_scores.append(iou_score(p, m))\n",
        "    val_loss /= len(val_ds)\n",
        "    val_dice = sum(dice_scores) / len(dice_scores)\n",
        "    val_iou = sum(iou_scores) / len(iou_scores)\n",
        "\n",
        "    history[\"train_loss\"].append(train_loss)\n",
        "    history[\"val_loss\"].append(val_loss)\n",
        "    history[\"val_dice\"].append(val_dice)\n",
        "    history[\"val_iou\"].append(val_iou)\n",
        "    print(f\"Epoch {epoch+1}/{EPOCHS}  train_loss={train_loss:.4f}  \"\n",
        "          f\"val_loss={val_loss:.4f}  val_dice={val_dice:.4f}  val_iou={val_iou:.4f}\")\n",
        "\n",
        "    if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:\n",
        "        ckpt_path = CHECKPOINT_DIR / f\"unet_epoch{epoch+1}.pth\"\n",
        "        torch.save(model.state_dict(), ckpt_path)\n",
        "        print(\"  saved checkpoint:\", ckpt_path)"
    ))

    cells.append(md("## Loss / Dice curves"))
    cells.append(code(
        "import matplotlib.pyplot as plt\n",
        "\n",
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
        "axes[0].plot(history[\"train_loss\"], label=\"train\")\n",
        "axes[0].plot(history[\"val_loss\"], label=\"val\")\n",
        "axes[0].set_title(\"Dice+BCE loss\")\n",
        "axes[0].set_xlabel(\"epoch\")\n",
        "axes[0].legend()\n",
        "\n",
        "axes[1].plot(history[\"val_dice\"], label=\"val Dice\")\n",
        "axes[1].plot(history[\"val_iou\"], label=\"val IoU\")\n",
        "axes[1].set_title(\"Validation metrics\")\n",
        "axes[1].set_xlabel(\"epoch\")\n",
        "axes[1].legend()\n",
        "fig.tight_layout()\n",
        "fig.savefig(\"training_curves.png\", dpi=150)\n",
        "plt.show()"
    ))

    cells.append(md(
        "## Final evaluation vs. the naive baseline\n",
        "\n",
        "Same validation split, same metrics as the local "
        "`scripts/evaluate_baseline.py` run recorded in `BENCHMARKS.md` "
        "(Otsu-threshold baseline: Dice 0.0190 overall / 0.0379 positive-only) "
        "-- so this is an apples-to-apples comparison, not just a training-run number."
    ))
    cells.append(code(
        "import cv2\n",
        "import numpy as np\n",
        "\n",
        "from dicom_seg.baseline import naive_threshold_baseline\n",
        "from dicom_seg.preprocessing import dicom_to_image\n",
        "from dicom_seg.rle import rle_decode_multi\n",
        "\n",
        "model.eval()\n",
        "unet_dice, unet_iou = [], []\n",
        "unet_dice_pos, unet_iou_pos = [], []\n",
        "baseline_dice_pos, baseline_iou_pos = [], []\n",
        "\n",
        "with torch.no_grad():\n",
        "    for entry in val_entries:\n",
        "        image = dicom_to_image(entry.dicom_path, output_range=\"unit\")\n",
        "        target = rle_decode_multi(entry.rle_strings, height=image.shape[0], width=image.shape[1])\n",
        "\n",
        "        image_256 = cv2.resize(image, (256, 256))\n",
        "        image_t = torch.from_numpy(image_256).float().unsqueeze(0).unsqueeze(0).to(device)\n",
        "        pred = model(image_t)[0, 0].cpu().numpy()\n",
        "        pred_resized = cv2.resize(pred, (image.shape[1], image.shape[0]))\n",
        "\n",
        "        d, i = dice_coefficient(pred_resized, target), iou_score(pred_resized, target)\n",
        "        unet_dice.append(d)\n",
        "        unet_iou.append(i)\n",
        "        if entry.has_mask:\n",
        "            unet_dice_pos.append(d)\n",
        "            unet_iou_pos.append(i)\n",
        "            baseline_pred = naive_threshold_baseline(image)\n",
        "            baseline_dice_pos.append(dice_coefficient(baseline_pred, target))\n",
        "            baseline_iou_pos.append(iou_score(baseline_pred, target))\n",
        "\n",
        "print(f\"U-Net      -- Dice (all): {np.mean(unet_dice):.4f}  IoU (all): {np.mean(unet_iou):.4f}\")\n",
        "print(f\"U-Net      -- Dice (pos-only): {np.mean(unet_dice_pos):.4f}  IoU (pos-only): {np.mean(unet_iou_pos):.4f}\")\n",
        "print(f\"Baseline   -- Dice (pos-only): {np.mean(baseline_dice_pos):.4f}  IoU (pos-only): {np.mean(baseline_iou_pos):.4f}\")\n",
        "print(\"\\nCopy these numbers into BENCHMARKS.md's U-Net row.\")"
    ))

    cells.append(md(
        "## Seg-Grad-CAM examples\n",
        "\n",
        "Adapted Grad-CAM (Milestone 7): backpropagates from the summed predicted-mask "
        "logits through the U-Net bottleneck, producing a heatmap of which regions "
        "drove the prediction."
    ))
    cells.append(code(
        "from dicom_seg.gradcam import SegGradCAM\n",
        "\n",
        "cam_tool = SegGradCAM(model, target_layer=model.bottleneck)\n",
        "\n",
        "positive_val = [e for e in val_entries if e.has_mask][:4]\n",
        "fig, axes = plt.subplots(3, len(positive_val), figsize=(4 * len(positive_val), 9))\n",
        "\n",
        "for col, entry in enumerate(positive_val):\n",
        "    image = dicom_to_image(entry.dicom_path, output_range=\"unit\")\n",
        "    target = rle_decode_multi(entry.rle_strings, height=image.shape[0], width=image.shape[1])\n",
        "    image_256 = cv2.resize(image, (256, 256))\n",
        "    image_t = torch.from_numpy(image_256).float().unsqueeze(0).unsqueeze(0).to(device)\n",
        "\n",
        "    cam, probs = cam_tool(image_t)\n",
        "    pred_mask = (probs[0, 0].cpu().numpy() > 0.5).astype(np.uint8)\n",
        "\n",
        "    axes[0, col].imshow(image_256, cmap=\"gray\")\n",
        "    axes[0, col].set_title(\"input\")\n",
        "    axes[0, col].axis(\"off\")\n",
        "\n",
        "    axes[1, col].imshow(image_256, cmap=\"gray\")\n",
        "    axes[1, col].imshow(np.ma.masked_where(pred_mask == 0, pred_mask), cmap=\"autumn\", alpha=0.5)\n",
        "    axes[1, col].set_title(\"predicted mask\")\n",
        "    axes[1, col].axis(\"off\")\n",
        "\n",
        "    axes[2, col].imshow(image_256, cmap=\"gray\")\n",
        "    axes[2, col].imshow(cam, cmap=\"jet\", alpha=0.4)\n",
        "    axes[2, col].set_title(\"Seg-Grad-CAM\")\n",
        "    axes[2, col].axis(\"off\")\n",
        "\n",
        "fig.tight_layout()\n",
        "fig.savefig(\"gradcam_examples.png\", dpi=150)\n",
        "plt.show()"
    ))

    cells.append(md(
        "## Bring results back to the local project\n",
        "\n",
        "Download `training_curves.png` and `gradcam_examples.png` (left sidebar > "
        "Files, right-click > Download) into the local `outputs/` folder, and copy "
        "the printed Dice/IoU numbers into `BENCHMARKS.md`. The final checkpoint in "
        "`CHECKPOINT_DIR` (on Drive) is the trained model."
    ))

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    NOTEBOOK_PATH.parent.mkdir(exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH} ({len(cells)} cells, "
          f"{NOTEBOOK_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    build()
