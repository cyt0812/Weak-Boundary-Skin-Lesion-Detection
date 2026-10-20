# MB-DCNN ISIC2017 reproduction

This directory records an adapted reproduction of the official
[YtongXie/MB-DCNN](https://github.com/YtongXie/MB-DCNN) Mask-CN pipeline on the
official ISIC2017 train/validation/test splits.

## Result

The selected classifier was evaluated on all 600 official test images. Nine
crop probabilities were averaged for each image.

| Metric | Value |
| --- | ---: |
| Accuracy | 0.7967 |
| Macro-F1 | 0.7236 |
| Nevus AUC | 0.8917 |
| Melanoma AUC | 0.8532 |
| Seborrheic keratosis AUC | 0.9600 |
| Mean melanoma/SK AUC | 0.9066 |

Confusion matrix (rows are true classes; columns are predicted classes):

```text
[[348, 23, 22],
 [ 46, 57, 14],
 [ 13,  4, 73]]
```

The complete metrics are in `results/test_results.json`.

## Experiment protocol

- Coarse segmentation: official DeepLabv3+ architecture, 500 epochs, selected
  by validation mean IoU. Best validation IoU: 0.7942.
- Classification: four-channel Xception_dilation (RGB + predicted mask), 50,001
  optimizer updates, selected by mean validation melanoma/SK AUC.
- Classification crops: 21,768 train, 1,350 validation and 5,400 test crops.
- Ground-truth masks are used to train/validate coarse segmentation only. The
  classifier receives predicted masks.
- The extra 1,320 training images used by the original work are omitted because
  their complete classification label table was unavailable.
- Python 3.10.21, PyTorch 2.6.0, CUDA 12.4 and native AMP were used.
- Smaller GPU microbatches with gradient accumulation were used, so this is an
  adapted reproduction rather than a numerically exact recreation.

## Run

Install dependencies in a Python 3.10 environment:

```bash
pip install -r requirements.txt
```

Place these upstream pretrained files in `models/` before running:

- `deeplabv3plus_xception_VOC2012_epoch46_all.pth`
- `xception-43020ad28.pth`

Then run from this directory:

```bash
python mbdcnn_pipeline.py
```

The script downloads and extracts the official ISIC2017 images and segmentation
annotations. Training outputs are written to `outputs/isic2017_no_extra/`.
Training data, generated crops, predicted masks and model weights are ignored by
Git and are not included here.

## Attribution

Architecture, crop functions and loss definitions are derived from
[YtongXie/MB-DCNN](https://github.com/YtongXie/MB-DCNN). The upstream MIT
license is included in `LICENSE`.

ISIC data source: <https://challenge.isic-archive.com/data/>
