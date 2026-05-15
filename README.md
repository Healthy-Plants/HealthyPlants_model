# HealthyPlants Model Training WIP

Initial in-progress branch for the crop and disease model training pipeline.

This branch currently includes:

- hierarchical crop -> disease training orchestration
- MobileNetV3-Small training entrypoint
- grouped train/val/test split preparation
- hierarchical inference script
- dependency and ignore-file setup

Large datasets and trained model artifacts are intentionally not included in this
WIP branch.

## Main Commands

Prepare hierarchical data:

```bash
python prepare_hierarchical_datasets.py --raw_root ./dataset --out_dir ./hierarchical_data
```

Train the pipeline:

```bash
python train_hierarchical.py --data_root ./hierarchical_data --output_root ./outputs_hierarchical
```

Run hierarchical inference once model artifacts are available:

```bash
python predict_hierarchical.py --image path/to/leaf.jpg
```
