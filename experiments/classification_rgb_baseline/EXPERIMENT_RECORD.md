# RGB Skin Lesion Classification Baseline

## 1. Experiment Overview

- **Experiment name:** RGB Classification Baseline
- **Project:** Weak-Boundary-Guided Skin Lesion Detection System
- **Dataset:** ISIC 2018
- **Task:** Seven-class skin lesion classification
- **Input:** RGB images
- **Segmentation mask:** Not used
- **Status:** Training and independent test evaluation completed

## 2. Dataset and Splits

The dataset was divided into training, validation, and test sets using stratified sampling.

| Split | Number of images | Proportion |
|---|---:|---:|
| Training | 7,010 | 70% |
| Validation | 1,502 | 15% |
| Test | 1,503 | 15% |
| Total | 10,015 | 100% |

- Random seed: 42
- Split strategy: Stratified sampling by disease class
- The three subsets were checked for overlap.
- Image names were checked against the available image files.

## 3. Model and Training Configuration

- Framework: PyTorch
- Execution device: CPU
- Input size: 256 × 256
- Image format: RGB
- Normalization: Pixel values divided by 255
- Batch size: 8
- Maximum epochs: 20
- Optimizer: Adam
- Initial learning rate: 0.001
- Learning-rate scheduler: ReduceLROnPlateau
- Class weighting: Computed from the training split
- Model selection: Best validation Macro-F1
- Best checkpoint epoch: 18

## 4. Independent Test Results

The selected checkpoint was evaluated on the independent test set containing 1,503 images.

| Metric | Result |
|---|---:|
| Test Accuracy | 61.88% |
| Test Macro-F1 | 45.14% |

### Per-class results

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| MEL | 0.3050 | 0.5150 | 0.3831 | 167 |
| NV | 0.9194 | 0.6799 | 0.7817 | 1,006 |
| BCC | 0.3036 | 0.6623 | 0.4163 | 77 |
| AKIEC | 0.3684 | 0.2857 | 0.3218 | 49 |
| BKL | 0.3160 | 0.4424 | 0.3687 | 165 |
| DF | 0.3333 | 0.1176 | 0.1739 | 17 |
| VASC | 0.5882 | 0.9091 | 0.7143 | 22 |

## 5. Preliminary Analysis

The baseline achieved 61.88% test accuracy and a Macro-F1 score of 45.14%.

The results show a clear difference between classes. NV achieved an F1-score of 0.7817, while DF achieved an F1-score of 0.1739. The dataset is imbalanced, with NV accounting for 1,006 of the 1,503 test images.

These results suggest that overall accuracy alone is insufficient to assess performance across all seven classes. Macro-F1 and per-class recall should also be considered in subsequent experiments.

The RGB baseline provides a reference for evaluating future improvements, including segmentation-assisted classification and other model modifications.

## 6. Reproducibility and File Locations

Project root:

`D:\GraduationProject\Skin-Lesion-Detection`

Source scripts:

- `src/classification_dataset.py`
- `src/generate_classification_labels.py`
- `src/generate_classification_split.py`
- `src/train_classification_cpu.py`
- `src/test_classification_batch.py`
- `src/evaluate_classification_test.py`

Dataset and split files:

`datasets\ISIC2018\classification\`

Training and evaluation outputs:

`outputs\classification\`

### Model Checkpoint

`outputs\classification\classification_rgb_best_macro_f1.pth`

### Training Log

`outputs\classification\classification_training_log.csv`

### Test Evaluation Files

All test evaluation files are stored in:

`outputs\classification\test_evaluation\`

- `classification_report.csv` — per-class precision, recall, F1-score, and support.
- `confusion_matrix.csv` — confusion matrix for the seven classes.
- `test_metrics.txt` — summary of test metrics.
- `test_predictions.csv` — test predictions for individual images.

### Experiment Record

`experiments\classification_rgb_baseline\EXPERIMENT_RECORD.md`

The model checkpoint, training log, and evaluation files remain in the original output directory. They are not moved by this experiment-recording process.

## 7. Limitations and Next Steps

1. Use the RGB baseline as the reference experiment.
2. Investigate the performance of minority classes, particularly DF and AKIEC.
3. Evaluate whether segmentation masks can improve classification performance.
4. Keep the data splits and evaluation metrics consistent when comparing models.
5. Record each subsequent experiment separately, including its configuration, checkpoint, and test results.

**Note:** These results describe this particular experimental setup and should not be interpreted as clinical diagnostic performance.