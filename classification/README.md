# Prerequisites

- https://github.com/lukemelas/EfficientNet-PyTorch


# Training

```bash
sh train.sh <experiment_name>
```

will create `class_ckpts/<experiment_name>` with
- `log.txt`: training log
- `train.sh`: copy of itself
- 'i_ckpt.pth': model's checkpoints with best accuracy

Add augmentations to `augmentations.py`.
Add losses to `losses.py`.
Add models to `model.py`.
Add auxiliary functions to `utils.py`.


# Notebooks

- make_classification.ipynb -- crop bboxes from the given images
- make_2stage_video.ipynb -- get detection output and classify proposals
- check_augmentations.ipynb -- see how augmentations distort images
- analyze_cls.ipynb -- see errors and metrics
