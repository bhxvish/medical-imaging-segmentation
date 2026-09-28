# Medical Image Segmentation with Explainability

Lesion segmentation on real DICOM files (chest X-ray, SIIM-ACR pneumothorax
dataset), with a from-scratch U-Net and Seg-Grad-CAM explainability.

Status: **Milestones 1-8 done**, including a trained U-Net evaluated against
a naive baseline (~3.2x Dice improvement, see [BENCHMARKS.md](BENCHMARKS.md))
and Seg-Grad-CAM explainability. Optional FastAPI viewer and final polish
remain.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate   # or source .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
pytest tests/ -v
```

## Milestone 1 - What's actually inside a DICOM file

A DICOM file is not "an image with some extra tags" - the pixel data on its
own is not display-ready. [`dicom_seg/preprocessing.py`](dicom_seg/preprocessing.py)
implements the correct pipeline:

1. **Read the pixel data.** `ds.pixel_array` gives you the *raw stored
   integers* - not calibrated to any physical unit yet.
2. **Apply the Modality LUT** (`RescaleSlope` / `RescaleIntercept`, or a LUT
   Sequence). For CT this converts stored values into Hounsfield Units (HU);
   for most X-ray this is close to identity, but it's still the correct step
   to apply before anything else.
3. **Apply the VOI LUT / default window** (`WindowCenter` / `WindowWidth`, or
   a VOI LUT Sequence), if present, to get display-ready contrast.
4. **Correct for `PhotometricInterpretation`.** Most images are
   `MONOCHROME2` (low value = dark, as expected). Some are `MONOCHROME1`
   (low value = **bright**) - get this backwards and every image renders as
   a photographic negative.
5. **Normalize** to `[0, 1]` or `[0, 255]`.

Key metadata fields and what they mean are documented in
[`METADATA_FIELDS`](dicom_seg/preprocessing.py) in the same file.

Run `python scripts/milestone1_2_demo.py` to print metadata for sample CT/MR
files and regenerate the images below (uses pydicom's bundled sample DICOMs,
no external dataset required).

![Milestone 1 examples](outputs/milestone1_examples.png)

Both render with correct orientation and contrast - not inverted, not
blown out.

## Milestone 2 - Windowing

CT stores intensities in Hounsfield Units across a huge range (air ~ -1000
HU, dense bone up to +3000 HU). No display can show that whole range with
good contrast at once. **Windowing** picks a window **center** and **width**
- a narrow slice of that range - and stretches just that slice across the
full display range, so the tissue you care about shows maximum contrast.
Everything outside the window clips to pure black or pure white.

`apply_window(image_hu, center, width)` in
[`dicom_seg/preprocessing.py`](dicom_seg/preprocessing.py) implements this.
The same CT slice, windowed two different ways:

![Windowing comparison](outputs/milestone2_windowing.png)

- **Lung window** (C=-600, W=1500): bone saturates to solid white, but soft
  tissue/lung detail is maximally distinguishable.
- **Bone window** (C=400, W=1800): trabecular bone detail becomes visible,
  at the cost of soft tissue contrast.

Chest X-ray (this project's primary dataset) doesn't have HU-calibrated
values, so windowing is less critical there than for CT - but the concept
carries over conceptually (and would matter immediately if this pipeline
were fed CT input instead).

## Tests

```bash
pytest tests/ -v
```

12 tests covering: metadata extraction, Modality LUT correction against a
hand-computed HU conversion, windowing against hand-computed clip/scale
values, and `MONOCHROME1` inversion verified against a synthetic in-memory
DICOM dataset (constructed directly with `pydicom.Dataset`, not a file on
disk, so the inversion logic is checked exactly rather than hoping a sample
file happens to be `MONOCHROME1`).

## Milestone 3 - Segmentation dataset pipeline

[`dicom_seg/rle.py`](dicom_seg/rle.py) decodes SIIM's column-major, delta-encoded
RLE mask format (ported from Kaggle's official `mask_functions.py`, with the
width/height transpose applied so masks line up with `dicom_to_image`'s
(H, W) output -- forgetting that transpose is the classic bug for this
dataset). [`dicom_seg/dataset.py`](dicom_seg/dataset.py) builds a manifest
pairing each downloaded DICOM with its mask, does a stratified train/val
split (so both splits keep a similar positive/negative ratio), and wraps it
in a `torch.utils.data.Dataset`.

Data subset: 400 images from SIIM-ACR (200 pneumothorax-positive, 200
negative), pulled via the Kaggle API from a re-hosted copy of the dataset
that preserves original DICOM filenames matching the label CSV (the
official competition's own downloadable files turn out to be the
anonymized *test* set, which doesn't match the labels at all).

Run `python scripts/milestone3_demo.py` to rebuild the manifest and the
verification image below -- overlaying decoded masks on their source
images is the cheapest way to catch a misaligned mask, the single most
common silent bug in segmentation projects.

![Mask overlay verification](outputs/milestone3_mask_overlay.png)

Red regions sit at the lung apex / pleural line, which is exactly where a
pneumothorax (partially collapsed lung) shows up -- confirming the RLE
decode and axis handling are correct, not transposed or offset.

## Milestone 4 - U-Net from scratch

[`dicom_seg/unet.py`](dicom_seg/unet.py) implements the encoder-decoder
with skip connections directly in PyTorch (no `segmentation_models_pytorch`
or similar). Skip connections concatenate each encoder stage's feature map
onto the decoder at the same resolution, before that decoder block's convs
-- this is what lets the network recover fine spatial detail lost during
downsampling; without them the output is a blurry, low-resolution guess
upsampled back to full size.

~31M parameters at the default `features=[64,128,256,512]` config. Verified
with parametrized input/output shape tests (64/128/256, single- and
multi-channel input) plus an autograd smoke test, and an end-to-end run of
a real `DataLoader` batch through the model.

## Milestone 5 - Loss function

[`dicom_seg/losses.py`](dicom_seg/losses.py): `DiceBCELoss` combines soft
Dice loss (direct overlap measure, stays informative when the foreground
region is small) with BCE (stable per-pixel gradient). The actual training
run happens on Colab (see `notebooks/train_colab.ipynb`), not this laptop.

## Milestone 6 - Evaluation vs. a naive baseline

[`dicom_seg/metrics.py`](dicom_seg/metrics.py): Dice coefficient and IoU,
hand-verified (identical masks -> 1.0, disjoint masks -> 0.0).
[`dicom_seg/baseline.py`](dicom_seg/baseline.py): a naive Otsu-thresholding
segmenter with no learning, as the "did the complexity earn its keep"
comparison point -- Dice 0.019 overall / 0.038 on pneumothorax-positive
images, confirming intensity alone can't separate pneumothorax from the
rest of the X-ray. The trained U-Net scores Dice 0.121 / IoU 0.073 on the
same positive-only images -- a ~3.2x Dice / ~3.6x IoU improvement. Full
numbers and honest notes on training-run stability in
[BENCHMARKS.md](BENCHMARKS.md).

## Training on Colab

`notebooks/train_colab.ipynb` is self-contained: it writes out the exact
`dicom_seg/*.py` files from this repo (kept in sync via
`scripts/build_colab_notebook.py`), downloads the same 400-image subset
(same seed, same UIDs) used for the local baseline benchmark, trains the
U-Net, plots loss/Dice curves, evaluates against the naive baseline on the
same validation split, and generates Seg-Grad-CAM examples. Open it in
Colab, set the runtime to a T4 GPU, and run top to bottom -- it'll prompt
for your `kaggle.json` the first time it needs it.

## Milestone 7 - Seg-Grad-CAM explainability

[`dicom_seg/gradcam.py`](dicom_seg/gradcam.py) adapts Grad-CAM to
segmentation: instead of backpropagating a classification score, it sums
the predicted mask's *logits* (pre-sigmoid, so gradients don't vanish at
saturated confidence) over a region of interest, then weights the target
layer's activations by the averaged gradient per channel, ReLUs (only
positive influence counts), and upsamples to the input size -- the
standard Grad-CAM recipe, applied to whichever conv layer you hook (the
bottleneck, or an encoder stage). Verified mechanically against an
untrained model (hooks fire, gradients flow, output shape/range correct)
before training, and now has real examples from the trained model in
`outputs/gradcam_examples.png` -- heatmaps concentrate around the lung
apex / upper chest, which is anatomically where pneumothorax presents.

## Roadmap

- [x] Milestone 1: DICOM metadata + pixel preprocessing
- [x] Milestone 2: CT windowing
- [x] Milestone 3: SIIM-ACR segmentation dataset pipeline (RLE mask decode)
- [x] Milestone 4: U-Net from scratch
- [x] Milestone 5: Combined Dice + BCE loss, trained on Colab (30 epochs, T4 GPU)
- [x] Milestone 6: Dice/IoU metrics + naive baseline -- U-Net beats it ~3.2x Dice / ~3.6x IoU (positive-only)
- [x] Milestone 7: Seg-Grad-CAM, with real heatmaps from the trained model (`outputs/gradcam_examples.png`)
- [x] Milestone 8: Test suite (57 tests across preprocessing, RLE, dataset, U-Net, losses, metrics, baseline, Grad-CAM)
- [ ] Milestone 9 (optional): FastAPI + minimal viewer
- [ ] Milestone 10: Polish, CI
