# Crop and Plant Disease Classifier

This repository contains PyTorch training and inference scripts for plant leaf
classification. The main workflow is hierarchical:

1. classify the crop (`corn`, `cotton`, `potato`, `rice`)
2. route the image to that crop's disease classifier

The repository can also train a single rice disease model with MobileNetV2 or
MobileNetV3-Small.

## Current Project State

The saved hierarchical model artifacts are expected at:

```text
outputs_hierarchical/
  crop_classifier/
    best_model.pth
    class_names.json
    results.json
  disease_classifiers/
    corn/
    cotton/
    potato/
    rice/
```

Current saved test results:

| Model | Classes | Test accuracy |
| --- | ---: | ---: |
| Crop classifier | 4 | 99.96% |
| Corn disease classifier | 4 | 95.54% |
| Cotton disease classifier | 7 | 95.36% |
| Potato disease classifier | 3 | 99.59% |
| Rice disease classifier | 4 | 100.00% |

If you want a clone to run inference immediately, commit the
`outputs_hierarchical/**/best_model.pth`, `class_names.json`, and `results.json`
files. The model files are about 14 MB each.

## Requirements

- Python 3.10 to 3.12 recommended. The project has also been tested locally with
  Python 3.14.
- Enough disk space for datasets. The prepared datasets in this workspace are
  several hundred MB.
- Optional GPU support through CUDA, Apple MPS, or CPU fallback.

## Setup After Cloning

```bash
git clone <your-repo-url>
cd <repo-folder>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

For PyTorch with a specific CUDA version, install the matching `torch` and
`torchvision` wheels from the official PyTorch selector first, then install the
remaining requirements:

```bash
pip install -r requirements.txt
```

To silence non-critical online version checks from Albumentations in offline
environments:

```bash
export NO_ALBUMENTATIONS_UPDATE=1
```

On locked-down systems, Matplotlib may need a writable cache directory:

```bash
mkdir -p .cache/matplotlib
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
```

## Run Inference

### Hierarchical crop + disease prediction

```bash
python predict_hierarchical.py \
  --image path/to/leaf.jpg \
  --crop_model ./outputs_hierarchical/crop_classifier/best_model.pth \
  --disease_models_root ./outputs_hierarchical/disease_classifiers
```

Folder prediction:

```bash
python predict_hierarchical.py \
  --folder path/to/images \
  --crop_model ./outputs_hierarchical/crop_classifier/best_model.pth \
  --disease_models_root ./outputs_hierarchical/disease_classifiers
```

Folder prediction writes `hierarchical_predictions.json` into the image folder.

### Single model prediction

Use this for a single crop model such as the rice MobileNetV3-Small output:

```bash
python predict.py \
  --model ./outputs_mobilenetv3_small/best_model.pth \
  --image path/to/leaf.jpg
```

With Grad-CAM:

```bash
python predict.py \
  --model ./outputs_mobilenetv3_small/best_model.pth \
  --image path/to/leaf.jpg \
  --gradcam
```

## Dataset Layouts

Raw multi-crop data should be arranged as one folder per crop, with one folder
per disease inside each crop:

```text
dataset/
  corn/
    Blight/
    Common_Rust/
    Gray_Leaf_Spot/
    Healthy/
  cotton/
    Bacterial Blight/
    Curl Virus/
    Healthy Leaf/
  potato/
    Early_Blight/
    Healthy/
    Late_Blight/
  rice/
    Bacterialblight/
    Blast/
    Brownspot/
    Tungro/
```

Prepared hierarchical data is generated as:

```text
hierarchical_data/
  crop_classifier/
    train/<crop>/
    val/<crop>/
    test/<crop>/
  disease_classifiers/
    <crop>/train/<disease>/
    <crop>/val/<disease>/
    <crop>/test/<disease>/
```

Datasets are intentionally ignored by `.gitignore` because they are large.
After cloning, download or copy the raw dataset into `dataset/` or pass explicit
paths to the preparation scripts.

## Prepare Data

### Prepare hierarchical multi-crop data

If your raw data is in the `dataset/` layout shown above:

```bash
python prepare_hierarchical_datasets.py \
  --raw_root ./dataset \
  --out_dir ./hierarchical_data
```

You can also pass explicit crop mappings:

```bash
python prepare_hierarchical_datasets.py \
  --crop rice="./dataset/rice" \
  --crop cotton="./dataset/cotton" \
  --crop potato="./dataset/potato" \
  --crop corn="./dataset/corn" \
  --out_dir ./hierarchical_data
```

Use `--link_mode copy` on systems that do not support hardlinks:

```bash
python prepare_hierarchical_datasets.py --raw_root ./dataset --link_mode copy
```

### Prepare a generic single-crop dataset

For a class-folder dataset:

```text
raw_data/rice_leaf_disease_images/
  Bacterialblight/
  Blast/
  Brownspot/
  Tungro/
```

Create leakage-resistant grouped splits:

```bash
python prepare_rice_leaf_grouped_split.py \
  --data_dir ./raw_data/rice_leaf_disease_images \
  --out_dir ./dataset_rice_leaf_grouped
```

For Paddy Doctor:

```bash
pip install kaggle
export KAGGLE_USERNAME=<your_username>
export KAGGLE_KEY=<your_api_key>
kaggle competitions download -c paddy-disease-classification
unzip paddy-disease-classification.zip -d ./raw_data/paddy

python prepare_dataset.py \
  --raw_dir ./raw_data/paddy \
  --out_dir ./dataset_paddy \
  --format paddy
```

## Train Models

### Train the full hierarchical pipeline

```bash
python train_hierarchical.py \
  --data_root ./hierarchical_data \
  --output_root ./outputs_hierarchical \
  --epochs 30 \
  --batch_size 32
```

On CPU-only machines, reduce workers and batch size:

```bash
python train_hierarchical.py \
  --data_root ./hierarchical_data \
  --output_root ./outputs_hierarchical \
  --epochs 30 \
  --batch_size 16 \
  --num_workers 0
```

Train only crop-specific disease classifiers:

```bash
python train_hierarchical.py \
  --data_root ./hierarchical_data \
  --output_root ./outputs_hierarchical \
  --skip_crop_classifier
```

Train one disease classifier:

```bash
python train_crop_disease_classifier.py \
  --crop rice \
  --data_root ./hierarchical_data/disease_classifiers \
  --output_root ./outputs_hierarchical/disease_classifiers
```

### Train the single rice MobileNetV3-Small model

```bash
python train_mobilenetv3_small.py \
  --data_dir ./dataset_rice_leaf_grouped \
  --output_dir ./outputs_mobilenetv3_small \
  --epochs 30 \
  --batch_size 32
```

### Train the older MobileNetV2 pipeline

```bash
python train.py \
  --data_dir ./dataset_paddy \
  --output_dir ./outputs \
  --epochs 50 \
  --batch_size 32
```

## Export Test Predictions

`model_test.py` exports one CSV row per test image, including the true label,
predicted label, and probability for every class:

```bash
python model_test.py \
  --model_dir ./outputs_hierarchical/disease_classifiers/corn \
  --data_dir ./hierarchical_data/disease_classifiers/corn
```

The default output is:

```text
outputs_hierarchical/disease_classifiers/corn/test_predictions_with_probs.csv
```

## Important Files

```text
requirements.txt                  Python dependencies
prepare_hierarchical_datasets.py   Build crop and disease train/val/test splits
prepare_rice_leaf_grouped_split.py Build grouped rice train/val/test splits
train_hierarchical.py              Train crop classifier and disease classifiers
train_crop_classifier.py           Train only the crop classifier
train_crop_disease_classifier.py   Train one crop-specific classifier
train_mobilenetv3_small.py         MobileNetV3-Small training entrypoint
train.py                           Older MobileNetV2 training entrypoint
predict_hierarchical.py            Crop + disease inference
predict.py                         Single-model inference with optional Grad-CAM
model_test.py                      Export test predictions to CSV
```

## Git Notes

Commit source code, `requirements.txt`, this README, and the trained model
artifacts if you want immediate inference after cloning.

Do not commit:

- `.venv/`
- `__pycache__/`
- `.DS_Store`
- raw or prepared datasets
- ad hoc training scratch outputs

The included `.gitignore` follows that split.
