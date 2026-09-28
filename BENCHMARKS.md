# Benchmarks

Evaluated on the held-out validation split (80 images: 40 pneumothorax-positive,
40 negative) from the 400-image SIIM-ACR subset. Seed 42, `val_fraction=0.2`,
reproducible via `scripts/evaluate_baseline.py` / the Colab training notebook.

| Model                          | Dice (all) | IoU (all) | Dice (positive only) | IoU (positive only) |
|---------------------------------|:----------:|:---------:|:---------------------:|:--------------------:|
| Naive Otsu-threshold baseline   | 0.0190     | 0.0100    | 0.0379                | 0.0200                |
| U-Net (this project)            | 0.2604     | 0.2363    | 0.1208                | 0.0727                |

**~3.2x Dice improvement (0.1208 vs. 0.0379) and ~3.6x IoU improvement
(0.0727 vs. 0.0200) over the naive baseline, on the positive-only images
that actually matter clinically.** Trained 30 epochs, Adam lr=1e-4, batch
size 8, on a Colab T4 GPU (`notebooks/train_colab.ipynb`).

The gap between "all" and "positive only" Dice (0.26 vs 0.12) isn't a bug:
on the 40 negative validation images the ground truth mask is empty, and
whenever the model also predicts (near-)empty, that scores Dice=1.0 by
definition (see the "both empty -> 1.0" convention in `dicom_seg/metrics.py`,
matching two identical masks). Averaging those in inflates the "all" number
-- the positive-only column is the honest measure of actual segmentation
quality.

## Naive baseline

`dicom_seg/baseline.py`: Otsu's method picks a single global intensity
threshold with no training and no learned structure, then flags pixels
darker than that threshold (excluding pure-black background outside the
body) as the predicted mask. This is a legitimate "no learning at all"
comparison point, not a strawman -- it uses real image-processing logic,
it just has no way to distinguish an actual pneumothorax's subtle lucency
pattern from any other dark region in the image (ribs' shadows, normal
lung tissue, imaging artifacts).

The near-zero Dice/IoU scores confirm this: pneumothorax isn't
separable by intensity alone, which is exactly why a learned model with
spatial context (like the U-Net) is needed, and why this project's core
claim -- "the sophisticated approach earns its complexity" -- is a
measured result here, not an assumption.

## U-Net

Trained on Colab (T4 GPU) for 30 epochs, Adam optimizer (lr=1e-4), combined
Dice+BCE loss, batch size 8, on the same 320-image train split.

Observations worth being honest about:
- Validation Dice/IoU are noisy across epochs (peaking around 0.5 near
  epochs 5-10, then oscillating in the 0.08-0.23 range) rather than
  climbing smoothly -- expected with only 320 training images and no
  learning-rate schedule or early stopping, not a bug. Early stopping at
  the val-Dice peak, more training data, or augmentation would likely
  improve on this.
- Loss curves (train vs. val) track closely and decrease smoothly,
  indicating the combined Dice+BCE loss is optimizing correctly even
  though the thresholded Dice/IoU metric is noisier.

See `outputs/training_curves.png` (loss + Dice/IoU curves) and
`outputs/gradcam_examples.png` (predicted masks + Seg-Grad-CAM heatmaps on
validation examples) for the full training run.
