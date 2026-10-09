# Guided Image Classifier

A desktop PySide6 app for assigning custom labels to example images, training a supervised classifier, and copying all images into class folders.

## Install

Use Python 3.11–3.13 and run:

```bash
python -m pip install -r requirements.txt
python image_classifier.py
```

If PyTorch/torchvision need a specific CPU or CUDA build, follow https://pytorch.org/get-started/locally/.

## Use

1. Select a folder of images. This version scans the selected directory only, not subfolders.
2. Add at least two classes (e.g. Pass, Fail, Scratch). Give each class multiple representative examples.
3. Use keyboard digits 1–9 to assign labels and automatically advance; Left/Right arrows navigate. You can also click a class and select Assign Selected Class.
4. Press **Train and Classify All Images** and select an output folder separate from the input.
5. The application uses pretrained ResNet18 feature embeddings and an SVM to predict classes. Low-confidence images are placed into `_Needs_Review`.
6. Outputs include original image copies, `classification_results.csv` and `unreadable_images.csv`.

All images in the selected input folder, including the labeled training images, are copied. Labeled training images keep their manual labels, and are marked with confidence 1.0 (a manual assignment, not model-calibrated confidence).

On first run, torchvision downloads pretrained weights (~45 MB). Model accuracy depends on how many *varied, representative* examples you label. A minimum of one image per class is technically accepted, but dozens or hundreds are better; assess predictions in the output folders before using them operationally. The SVM confidence value is an estimate and should not be interpreted as a validated failure probability.
