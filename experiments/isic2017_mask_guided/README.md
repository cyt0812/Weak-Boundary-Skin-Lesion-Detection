# ISIC2017 MALUNet and Mask-guided classification experiments

This directory records Wu Xintong's ISIC2017 segmentation and classification experiments. It extends, but does not replace, the earlier [MALUNet ISIC2018 PR](https://github.com/cyt0812/Weak-Boundary-Skin-Lesion-Detection/pull/2) and [MB-DCNN ISIC2017 reproduction PR](https://github.com/cyt0812/Weak-Boundary-Skin-Lesion-Detection/pull/3). Those PRs are separate and may not yet be merged into `main`.

## What was run

1. MALUNet was trained **from random initialization** on the official ISIC2017 segmentation training images (2,000 images), with the 150-image validation split used for checkpoint selection. The 300-epoch run selected epoch 105. The final evaluation in the original MALUNet script reuses the same validation images; it is **not an independent segmentation test**.
2. The epoch-105 MALUNet checkpoint predicted lesion masks for the corresponding ISIC2017 classification crops. Ground-truth segmentation masks were **not** passed to the classifiers at inference.
3. A controlled MB-DCNN-based comparison trained one RGB-only classifier and one RGB-plus-MALUNet-mask classifier. Both used the same official 2,000/150/600 classification image split, crop manifests, architecture family, generic Xception pretraining, seed, and 50,001 optimizer steps. The 600-image official test set was evaluated at image level after validation checkpoint selection.
4. Validation-only analyses explored crop quality, mask dependence, feature pooling, learned feature fusion, and five-fold out-of-fold (OOF) training masks. These analyses reuse the 150-image validation split and are **exploratory**, not independent final-test evidence.

## Main results

| Experiment | Split | Accuracy | Macro-F1 | Other |
| --- | --- | ---: | ---: | --- |
| MALUNet segmentation | ISIC2017 validation, 150 images | 94.59% pixel accuracy | — | Dice 84.24%, IoU 72.76% |
| RGB-only MB-DCNN-based classifier | ISIC2017 official test, 600 images | 79.67% | 72.41% | Melanoma recall 60/117 |
| RGB + MALUNet-mask classifier | Same 600-image test | 80.17% | 71.53% | Melanoma recall 49/117 |

The mask-guided classifier had three more correct images overall but lower Macro-F1 and melanoma recall. **These results do not establish that the Mask improves classification.** The RGB-only arm is an internal no-Mask control, **not** teammate CYT's independently trained Scheme A. Scheme A has a different classifier and must be evaluated on the same task and split before a team-level A/B score comparison.

## Files

- `scripts/`: the local training, mask generation, audit, and validation-analysis scripts used for this work.
- `results/`: small metric files and per-image predictions. The two official test result files are the classifier outcomes above; files containing `validation` or `oof` are exploratory checks.
- `figures/`: presentation diagrams for the project-level Scheme A/B distinction, the four-channel implementation, the internal Mask ablation, actual ISIC2017/ISIC2018 status, and the planned weak-boundary study. The RGB-only branch of the internal ablation is not Scheme A. The clinical image and masks used for P5 are kept locally and are not uploaded here.

## Reproducing or reviewing

The scripts are archived as run and contain the author's `D:\Desktop\...` paths. Set those paths to the local locations of the ISIC2017 data, the MALUNet repository, and the MB-DCNN reproduction before rerunning. The original network implementations are in the two earlier PRs linked above; this directory does not duplicate their upstream code.

Python 3.10.21 and PyTorch 2.6.0 with CUDA 12.4 were used for the original MB-DCNN reproduction. See the earlier reproduction README for documented deviations from the paper, including omission of the extra 1,320 training images and smaller accumulated microbatches. The later Mask comparison used the same official classification split for both arms.

Raw ISIC images, original annotation files, crop images, tens of thousands of predicted Mask PNGs, pretrained third-party weights, and large classifier optimizer checkpoints are deliberately **not** committed here. They remain local. The MALUNet ISIC2017 best checkpoint and other model weights should be transferred separately with an explicit provenance/size policy if needed for deployment.

The official 600-image test results were already inspected during the project. Do not repeatedly select new variants against that set and then describe the selected score as an untouched test result.

