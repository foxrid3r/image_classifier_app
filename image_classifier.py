"""Interactive image classifier: PySide6 + pretrained ResNet18 embeddings + SVM.

Install: pip install PySide6 Pillow numpy scikit-learn torch torchvision
Run: python image_classifier.py

On first use, torchvision may download pretrained ResNet18 weights (~45 MB).
"""
from __future__ import annotations

import csv
import io
import hashlib
import tempfile
import zipfile
import joblib
import json
import os
import shutil
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QPixmap, QImage, QKeySequence, QShortcut, QIcon
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QSpinBox, QDoubleSpinBox, QSplitter,
    QStackedWidget, QVBoxLayout, QWidget, QPlainTextEdit, QCheckBox,
    QDialog, QListWidgetItem, QAbstractItemView
)

MODEL_FORMAT_VERSION = 2
EMBEDDING_ID = 'torchvision/resnet18/ResNet18_Weights.DEFAULT/avgpool-512'

EXTENSIONS = {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp'}


def scan_images(directory: Path) -> list[Path]:
    return sorted((p for p in directory.iterdir()
                   if p.is_file() and p.suffix.lower() in EXTENSIONS),
                  key=lambda p: p.name.lower())


def safe_folder(label: str) -> str:
    invalid = '<>:"/\\|?*'
    clean = ''.join('_' if c in invalid or ord(c) < 32 else c for c in label)
    clean = clean.rstrip(' .')
    if not clean or clean in ('.', '..'):
        raise ValueError('Class name cannot be used as a folder name.')
    return clean


def unique_target(folder: Path, source_name: str) -> Path:
    target = folder / source_name
    if not target.exists():
        return target
    stem, suffix = Path(source_name).stem, Path(source_name).suffix
    n = 2
    while (folder / f'{stem}_{n}{suffix}').exists():
        n += 1
    return folder / f'{stem}_{n}{suffix}'


class ClassificationWorker(QThread):
    """Predict only; never writes any files. Cache features between training rounds."""
    progress = Signal(int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, files, labels, cached_features=None, existing_model=None, training_bank=None, retrain=True, parent=None):
        super().__init__(parent)
        self.files = list(files)
        self.labels = dict(labels)
        self.cached_features = cached_features
        self.existing_model = existing_model
        self.training_bank = dict(training_bank or {})
        self.retrain = retrain
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            from sklearn.svm import SVC
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            if self.cached_features is None:
                import torch
                from torchvision import models
                self.progress.emit(0, len(self.files), 'Loading pretrained vision model...')
                weights = models.ResNet18_Weights.DEFAULT
                network = models.resnet18(weights=weights)
                network.fc = torch.nn.Identity()
                network.eval()
                preprocess = weights.transforms()
                features, valid_files, errors = [], [], []
                with torch.inference_mode():
                    for i, file in enumerate(self.files):
                        if self._stop:
                            self.completed.emit(None)
                            return
                        try:
                            with Image.open(file) as im:
                                rgb = ImageOps.exif_transpose(im).convert('RGB')
                                x = preprocess(rgb).unsqueeze(0)
                            features.append(network(x).squeeze(0).numpy())
                            valid_files.append(file)
                        except Exception as exc:
                            errors.append((file.name, str(exc)))
                        self.progress.emit(i + 1, len(self.files), f'Extracting features: {file.name}')
                if not features:
                    raise RuntimeError('No readable images found.')
                cache = (valid_files, np.stack(features), errors)
            else:
                cache = self.cached_features
                self.progress.emit(0, len(self.files), 'Reusing image features; retraining classifier...')

            if self._stop:
                self.completed.emit(None)
                return
            valid_files, X, errors = cache
            if self.retrain:
                bank = dict(self.training_bank)
                # File-specific overrides ensure corrected labels replace previously saved examples.
                for i, file in enumerate(valid_files):
                    key = str(file.resolve())
                    if file.name in self.labels:
                        bank[key] = (np.asarray(X[i], dtype=np.float32), self.labels[file.name])
                    else:
                        # Clearing a label in the current folder removes the old example.
                        bank.pop(key, None)
                # Never train using classes the user has removed from the UI.
                active_classes = set(self.labels.values()) | {v[1] for v in bank.values()}
                bank = {k: v for k, v in bank.items() if v[1] in active_classes}
                if len({v[1] for v in bank.values()}) < 2:
                    raise RuntimeError('At least two labeled classes are required to train.')
                classifier = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True,
                                                                  class_weight='balanced', random_state=42))
                train_X = np.stack([v[0] for v in bank.values()])
                train_y = [v[1] for v in bank.values()]
                self.progress.emit(len(self.files), len(self.files), 'Training and predicting...')
                classifier.fit(train_X, train_y)
            else:
                if self.existing_model is None:
                    raise RuntimeError('No imported trained model is available.')
                classifier = self.existing_model
                bank = dict(self.training_bank)
                self.progress.emit(len(self.files), len(self.files), 'Predicting with saved model...')
            probabilities = classifier.predict_proba(X)
            classes = list(classifier.classes_)
            predictions = {}
            for i, file in enumerate(valid_files):
                j = int(np.argmax(probabilities[i]))
                predictions[file.name] = (classes[j], float(probabilities[i, j]))
            if self._stop:
                self.completed.emit(None)
                return
            self.completed.emit({'predictions': predictions, 'cache': cache, 'errors': errors,
                                 'model': classifier, 'training_bank': bank})
        except Exception:
            self.failed.emit(traceback.format_exc())


class ExportWorker(QThread):
    progress = Signal(int, int, str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, files, labels, predictions, errors, output, threshold, parent=None):
        super().__init__(parent)
        self.files, self.labels = list(files), dict(labels)
        self.predictions, self.errors = dict(predictions), list(errors)
        self.output, self.threshold = Path(output), float(threshold)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self.output.mkdir(parents=True, exist_ok=True)
            counts, rows = Counter(), []
            for i, file in enumerate(self.files):
                if self._stop:
                    break
                if file.name not in self.predictions:
                    continue
                prediction, confidence = self.predictions[file.name]
                manually_labeled = file.name in self.labels
                if manually_labeled:
                    prediction = self.labels[file.name]
                dest_class = prediction if (manually_labeled or confidence >= self.threshold) else '_Needs_Review'
                folder = self.output / safe_folder(dest_class)
                folder.mkdir(parents=True, exist_ok=True)
                target = unique_target(folder, file.name)
                shutil.copy2(file, target)
                counts[dest_class] += 1
                rows.append((file.name, prediction, f'{confidence:.4f}',
                             'manual_label' if manually_labeled else 'automatic', dest_class, str(target)))
                self.progress.emit(i + 1, len(self.files), f'Copying: {file.name}')
            with (self.output / 'classification_results.csv').open('w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow(['filename', 'predicted_class', 'model_confidence', 'source', 'output_folder', 'output_path'])
                w.writerows(rows)
            with (self.output / 'unreadable_images.csv').open('w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow(['filename', 'error'])
                w.writerows(self.errors)
            msg = ('Export stopped. ' if self._stop else 'Export complete. ')
            msg += f'Copied {len(rows)} images. ' + ', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))
            if self.errors:
                msg += f' | {len(self.errors)} unreadable images.'
            self.completed.emit(msg)
        except Exception:
            self.failed.emit(traceback.format_exc())


class TrainingSetDialog(QDialog):
    """Inspect/edit both current-folder labels and historical saved feature vectors."""

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.setWindowTitle('Training Set Browser — correct labels or remove examples')
        self.resize(1100, 750)
        self.setModal(True)
        root = QVBoxLayout(self)
        filters = QHBoxLayout()
        filters.addWidget(QLabel('Show class:'))
        self.class_filter = QComboBox()
        self.class_filter.currentIndexChanged.connect(self.populate)
        filters.addWidget(self.class_filter, 1)
        self.tally = QLabel('')
        filters.addWidget(self.tally)
        root.addLayout(filters)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, 1)
        self.gallery = QListWidget()
        self.gallery.setViewMode(QListWidget.IconMode)
        self.gallery.setIconSize(QSize(145, 105))
        self.gallery.setGridSize(QSize(180, 156))
        self.gallery.setResizeMode(QListWidget.Adjust)
        self.gallery.setMovement(QListWidget.Static)
        self.gallery.setWordWrap(True)
        self.gallery.setSelectionMode(QAbstractItemView.SingleSelection)
        self.gallery.currentItemChanged.connect(self.show_selected)
        split.addWidget(self.gallery)
        details = QWidget()
        details_layout = QVBoxLayout(details)
        self.big_preview = QLabel('Select a training example')
        self.big_preview.setAlignment(Qt.AlignCenter)
        self.big_preview.setMinimumWidth(250)
        self.big_preview.setStyleSheet('background:#252525;color:#ddd')
        details_layout.addWidget(self.big_preview, 1)
        self.details = QLabel('')
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        details_layout.addWidget(self.details)
        details_layout.addWidget(QLabel('Correct class:'))
        self.correct_class = QComboBox()
        details_layout.addWidget(self.correct_class)
        change = QPushButton('Apply Corrected Label')
        change.clicked.connect(self.relabel)
        details_layout.addWidget(change)
        remove = QPushButton('Remove from Training Set')
        remove.clicked.connect(self.remove_example)
        details_layout.addWidget(remove)
        self.info = QLabel('Changes require retraining before export or saving the model.')
        self.info.setWordWrap(True)
        details_layout.addWidget(self.info)
        split.addWidget(details)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        done = QPushButton('Close')
        done.clicked.connect(self.accept)
        root.addWidget(done)
        self.class_filter.addItems(['All classes'] + self.owner.class_names())
        self.populate()

    def entries(self):
        # Bank entries are saved features; current-folder labels not yet trained
        # must also appear. Same file is shown once, keyed by resolved path.
        entries = {}
        for key, (_vector, label) in self.owner.training_bank.items():
            entries[key] = (Path(key), label, True)
        for file in self.owner.images:
            key = str(file.resolve())
            if file.name in self.owner.labels:
                entries[key] = (file, self.owner.labels[file.name], key in self.owner.training_bank)
            else:
                # Removed labels override any retained historical entry.
                entries.pop(key, None)
        return sorted(entries.items(), key=lambda kv: (kv[1][1].casefold(), kv[1][0].name.casefold(), kv[0]))

    def populate(self, *_):
        selected_key = self.gallery.currentItem().data(Qt.UserRole) if self.gallery.currentItem() else None
        self.gallery.clear()
        chosen = self.class_filter.currentText()
        rows = [(key, path, label, saved) for key, (path, label, saved) in self.entries()
                if chosen == 'All classes' or label == chosen]
        for key, path, label, saved in rows:
            item = QListWidgetItem(f'{path.name}\n[{label}]')
            item.setData(Qt.UserRole, key)
            item.setToolTip(str(path) + ('\nImage not available' if not path.is_file() else ''))
            if path.is_file():
                try:
                    pm = QPixmap(str(path))
                    if not pm.isNull():
                        item.setIcon(QIcon(pm.scaled(145, 105, Qt.KeepAspectRatio, Qt.FastTransformation)))
                except Exception:
                    pass
            self.gallery.addItem(item)
            if key == selected_key:
                self.gallery.setCurrentItem(item)
        self.tally.setText(f'{len(rows)} examples shown / {len(self.entries())} total')
        if self.gallery.currentRow() < 0 and self.gallery.count():
            self.gallery.setCurrentRow(0)
        if not self.gallery.count():
            self.show_selected(None)

    def selected_record(self):
        item = self.gallery.currentItem()
        if item is None:
            return None
        key = item.data(Qt.UserRole)
        for k, entry in self.entries():
            if k == key:
                return key, *entry
        return None

    def show_selected(self, item, *_):
        record = self.selected_record() if item else None
        self.correct_class.clear()
        self.correct_class.addItems(self.owner.class_names())
        if not record:
            self.big_preview.clear()
            self.big_preview.setText('No training example selected')
            self.details.clear()
            return
        key, path, label, saved = record
        self.correct_class.setCurrentText(label)
        source = 'Saved feature vector' if saved else 'New label (not trained yet)'
        self.details.setText(f'Class: {label}\nFile: {path}\n{source}')
        pm = QPixmap(str(path)) if path.is_file() else QPixmap()
        if not pm.isNull():
            self._large_pixmap = pm
            self.render_big_preview()
        else:
            self._large_pixmap = QPixmap()
            self.big_preview.setPixmap(QPixmap())
            self.big_preview.setText('Original image unavailable\nSaved feature vector can still be relabeled or removed')

    def render_big_preview(self):
        if not self._large_pixmap.isNull():
            self.big_preview.setPixmap(self._large_pixmap.scaled(
                self.big_preview.size(), Qt.KeepAspectRatio, Qt.FastTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_large_pixmap'):
            self.render_big_preview()

    def relabel(self):
        record = self.selected_record()
        if not record:
            return
        key, path, label, saved = record
        new_label = self.correct_class.currentText()
        if not new_label or new_label == label:
            return
        if saved:
            vector, _ = self.owner.training_bank[key]
            self.owner.training_bank[key] = (vector, new_label)
        if self.owner.directory and path.parent == self.owner.directory and path.name in {p.name for p in self.owner.images}:
            self.owner.labels[path.name] = new_label
        self.owner.mark_stale()
        self.owner.refresh()
        self.populate()
        self.info.setText(f'Corrected {path.name}: {label} → {new_label}. Retrain before export.')

    def remove_example(self):
        record = self.selected_record()
        if not record:
            return
        key, path, label, saved = record
        if QMessageBox.question(self, 'Remove training example',
                                f'Remove {path.name} from training class "{label}"?\nThe original image will not be deleted.',
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.owner.training_bank.pop(key, None)
        if self.owner.directory and path.parent == self.owner.directory:
            self.owner.labels.pop(path.name, None)
        self.owner.mark_stale()
        self.owner.refresh()
        self.populate()
        self.info.setText(f'Removed {path.name} from training. Retrain before export.')


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Guided Image Classifier')
        self.resize(1200, 780)
        self.directory = None
        self.images = []
        self.labels = {}
        self.worker = None
        self.feature_cache = None
        self.predictions = {}
        self.errors = []
        self.predictions_stale = False
        self.model = None
        self.training_bank = {}
        self.loaded_model = False
        self.model_image_temp = None
        self.index = 0
        self.image_pixmap = QPixmap()
        self._build_ui()
        for n in range(1, 10):
            QShortcut(QKeySequence(str(n)), self, activated=lambda n=n: self.quick_label(n - 1))
        QShortcut(QKeySequence('Right'), self, activated=self.next_image)
        QShortcut(QKeySequence('Left'), self, activated=self.prev_image)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        top = QVBoxLayout(root)
        source = QHBoxLayout()
        self.source_text = QLineEdit()
        self.source_text.setReadOnly(True)
        source.addWidget(QLabel('Images:'))
        source.addWidget(self.source_text, 1)
        browse = QPushButton('Select Folder...')
        browse.clicked.connect(self.choose_folder)
        source.addWidget(browse)
        top.addLayout(source)

        splitter = QSplitter(Qt.Horizontal)
        top.addWidget(splitter, 1)
        side = QWidget()
        side.setMinimumWidth(270)
        side.setMaximumWidth(420)
        side_layout = QVBoxLayout(side)

        classes_box = QGroupBox('Classes (1–9 to assign)')
        cv = QVBoxLayout(classes_box)
        self.class_list = QListWidget()
        cv.addWidget(self.class_list)
        class_add = QHBoxLayout()
        self.class_name = QLineEdit()
        self.class_name.setPlaceholderText('e.g. Defect, Pass, Scratch')
        self.class_name.returnPressed.connect(self.add_class)
        class_add.addWidget(self.class_name)
        add = QPushButton('Add')
        add.clicked.connect(self.add_class)
        class_add.addWidget(add)
        cv.addLayout(class_add)
        remove = QPushButton('Remove Selected Class')
        remove.clicked.connect(self.remove_class)
        cv.addWidget(remove)
        self.assign = QPushButton('Assign Selected Class')
        self.assign.clicked.connect(self.assign_current)
        cv.addWidget(self.assign)
        side_layout.addWidget(classes_box, 2)

        training = QGroupBox('Training / Classification')
        tv = QVBoxLayout(training)
        self.stats = QLabel('Choose a folder to begin.')
        self.stats.setWordWrap(True)
        tv.addWidget(self.stats)
        self.browse_training = QPushButton('Browse / Correct Training Set...')
        self.browse_training.clicked.connect(self.browse_training_set)
        tv.addWidget(self.browse_training)
        tv.addWidget(QLabel('Unlabeled images are skipped during training.\nLabel examples from every class.'))
        controls = QFormLayout()
        self.min_conf = QDoubleSpinBox()
        self.min_conf.setRange(0.0, 1.0)
        self.min_conf.setDecimals(2)
        self.min_conf.setSingleStep(0.05)
        self.min_conf.setValue(0.65)
        controls.addRow('Min confidence:', self.min_conf)
        tv.addLayout(controls)
        model_actions = QHBoxLayout()
        self.save_model_btn = QPushButton('Save Model...')
        self.save_model_btn.setEnabled(False)
        self.save_model_btn.clicked.connect(self.save_model)
        self.load_model_btn = QPushButton('Load Model...')
        self.load_model_btn.clicked.connect(self.load_model)
        model_actions.addWidget(self.save_model_btn)
        model_actions.addWidget(self.load_model_btn)
        tv.addLayout(model_actions)
        self.start = QPushButton('Train / Retrain and Preview')
        self.start.clicked.connect(self.start_classify)
        tv.addWidget(self.start)
        self.review_only = QCheckBox('Review low-confidence images only')
        self.review_only.setEnabled(False)
        self.review_only.toggled.connect(self.on_review_toggled)
        tv.addWidget(self.review_only)
        self.review_stats = QLabel('No predictions yet.')
        self.review_stats.setWordWrap(True)
        tv.addWidget(self.review_stats)
        self.export = QPushButton('Export Classified Images...')
        self.export.setEnabled(False)
        self.export.clicked.connect(self.export_results)
        tv.addWidget(self.export)
        self.min_conf.valueChanged.connect(self.on_threshold_changed)
        self.stop = QPushButton('Stop')
        self.stop.setEnabled(False)
        self.stop.clicked.connect(self.stop_classify)
        tv.addWidget(self.stop)
        self.progress = QProgressBar()
        tv.addWidget(self.progress)
        self.status = QLabel('')
        self.status.setWordWrap(True)
        tv.addWidget(self.status)
        side_layout.addWidget(training, 2)
        splitter.addWidget(side)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        self.preview = QLabel('Select an image directory')
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(250, 200)
        self.preview.setStyleSheet('QLabel { background-color: #252525; color: #ddd; border-radius: 5px; }')
        center_layout.addWidget(self.preview, 1)
        self.image_info = QLabel('')
        self.image_info.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(self.image_info)
        nav = QHBoxLayout()
        prev = QPushButton('◀ Previous')
        prev.clicked.connect(self.prev_image)
        nav.addWidget(prev)
        self.clear = QPushButton('Clear Label')
        self.clear.clicked.connect(self.clear_label)
        nav.addWidget(self.clear)
        next_btn = QPushButton('Skip / Next ▶')
        next_btn.clicked.connect(self.next_image)
        nav.addWidget(next_btn)
        center_layout.addLayout(nav)
        self.jump = QPushButton('Jump to Next Unlabeled Image')
        self.jump.clicked.connect(self.next_unlabeled)
        center_layout.addWidget(self.jump)
        splitter.addWidget(center)
        splitter.setStretchFactor(1, 1)
        self.class_list.itemDoubleClicked.connect(lambda _: self.assign_current())

    def choose_folder(self):
        directory = QFileDialog.getExistingDirectory(self, 'Select image folder')
        if not directory:
            return
        path = Path(directory)
        files = scan_images(path)
        if not files:
            QMessageBox.warning(self, 'No images', 'No supported images found in this folder (non-recursive).')
            return
        self.directory = path
        self.images = files
        self.labels = {}
        self.feature_cache = None
        self.predictions = {}
        self.errors = []
        self.predictions_stale = False
        self.model = None
        self.training_bank = {}
        self.loaded_model = False
        self.save_model_btn.setEnabled(False)
        self.start.setText('Train / Retrain and Preview')
        self.class_list.clear()
        self.review_only.setChecked(False)
        self.review_only.setEnabled(False)
        self.export.setEnabled(False)
        self.review_stats.setText('No predictions yet.')
        self.index = 0
        self.source_text.setText(str(path))
        self.refresh()

    def browse_training_set(self):
        if self.worker:
            QMessageBox.information(self, 'Busy', 'Finish the current operation before editing labels.')
            return
        TrainingSetDialog(self).exec()

    def class_names(self):
        return [self.class_list.item(i).text() for i in range(self.class_list.count())]

    def add_class(self):
        name = self.class_name.text().strip()
        if not name:
            return
        try:
            folder = safe_folder(name)
        except ValueError as exc:
            QMessageBox.warning(self, 'Invalid class', str(exc))
            return
        if name.casefold() in (x.casefold() for x in self.class_names()) or folder.casefold() in (safe_folder(x).casefold() for x in self.class_names()) or folder.casefold() == '_needs_review':
            QMessageBox.warning(self, 'Duplicate class', 'Choose another unique class name.')
            return
        self.mark_stale()
        self.class_list.addItem(name)
        self.class_list.setCurrentRow(self.class_list.count() - 1)
        self.class_name.clear()
        self.refresh()

    def remove_class(self):
        idx = self.class_list.currentRow()
        if idx < 0:
            return
        name = self.class_list.takeItem(idx).text()
        self.labels = {key: value for key, value in self.labels.items() if value != name}
        self.training_bank = {key: value for key, value in self.training_bank.items() if value[1] != name}
        self.mark_stale()
        self.refresh()

    def quick_label(self, idx):
        if idx < self.class_list.count():
            self.class_list.setCurrentRow(idx)
            self.assign_current()

    def assign_current(self):
        if not self.images:
            return
        item = self.class_list.currentItem()
        if item is None:
            QMessageBox.information(self, 'Select a class', 'Select or create a class first.')
            return
        self.labels[self.images[self.index].name] = item.text()
        self.mark_stale()
        self.advance(1)
        self.refresh()

    def clear_label(self):
        if self.images:
            self.labels.pop(self.images[self.index].name, None)
            self.mark_stale()
            self.refresh()

    def review_indices(self):
        if not (self.review_only.isChecked() and self.predictions):
            return list(range(len(self.images)))
        threshold = self.min_conf.value()
        return [i for i, file in enumerate(self.images)
                if file.name in self.predictions and file.name not in self.labels
                and self.predictions[file.name][1] < threshold]

    def advance(self, step):
        indices = self.review_indices()
        if not indices:
            self.refresh()
            return
        if self.index not in indices:
            self.index = indices[0]
        else:
            at = indices.index(self.index)
            self.index = indices[max(0, min(len(indices) - 1, at + step))]
        self.refresh()

    def prev_image(self):
        self.advance(-1)

    def next_image(self):
        self.advance(1)

    def next_unlabeled(self):
        if not self.images:
            return
        candidates = self.review_indices()
        for i in [k for k in candidates if k > self.index] + [k for k in candidates if k <= self.index]:
            if self.images[i].name not in self.labels:
                self.index = i
                break
        self.refresh()

    def mark_stale(self):
        if self.predictions or self.model is not None:
            self.predictions_stale = True
            self.export.setEnabled(False)
            self.save_model_btn.setEnabled(False)
            self.review_stats.setText('Training labels/classes changed. Retrain to update all predictions before export.')

    def on_threshold_changed(self):
        self.update_review_stats()
        self.refresh()

    def on_review_toggled(self):
        ids = self.review_indices()
        if ids and self.index not in ids:
            self.index = ids[0]
        self.refresh()

    def update_review_stats(self):
        if not self.predictions:
            return
        count = sum(f.name not in self.labels and confidence < self.min_conf.value()
                    for f in self.images
                    for prediction, confidence in [self.predictions.get(f.name, ('', 1.0))])
        self.review_stats.setText(
            f'{count} low-confidence, unlabeled images to review. '
            + ('Retrain required before export.' if self.predictions_stale
               else 'Review them, add training labels if needed, or export now.'))

    def refresh(self):
        counts = Counter(self.labels.values())
        self.stats.setText(f'{len(self.labels)}/{len(self.images)} images labeled\n' +
                           ', '.join(f'{name}: {counts[name]}' for name in self.class_names()))
        if not self.images:
            return
        file = self.images[self.index]
        tag = self.labels.get(file.name, 'Unlabeled')
        prediction = self.predictions.get(file.name)
        pred_text = (f' | Model: {prediction[0]} ({prediction[1]:.1%})' if prediction else '')
        self.image_info.setText(f'{self.index + 1} / {len(self.images)}  •  {file.name}\nLabel: {tag}{pred_text}')
        try:
            with Image.open(file) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                im.thumbnail((1600, 1200), Image.Resampling.NEAREST)
                data = im.tobytes('raw', 'RGB')
                qimg = QImage(data, im.width, im.height, im.width * 3, QImage.Format_RGB888).copy()
            self.image_pixmap = QPixmap.fromImage(qimg)
            self.update_preview()
        except Exception as exc:
            self.image_pixmap = QPixmap()
            self.preview.setText(f'Cannot preview image:\n{exc}')

    def update_preview(self):
        if not self.image_pixmap.isNull():
            self.preview.setPixmap(self.image_pixmap.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.FastTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_preview()

    def save_model(self):
        """Save a portable ZIP-based .gicmodel with ORIGINAL image bytes."""
        if self.model is None or self.predictions_stale or self.worker:
            QMessageBox.warning(self, 'Model not ready', 'Train/retrain the model before saving.')
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Save trained model', 'classifier.gicmodel', 'Guided Image Classifier (*.gicmodel)')
        if not filename:
            return
        path = Path(filename)
        if path.suffix.lower() != '.gicmodel':
            path = path.with_suffix('.gicmodel')
        # Never produce an incomplete 'portable' model without the source pixels.
        missing = [key for key in self.training_bank if not Path(key).is_file()]
        if missing:
            QMessageBox.critical(
                self, 'Original training images unavailable',
                f'{len(missing)} training image(s) are unavailable.\n\n'
                'This version requires every training image to save a portable model. '
                'Restore the source images or remove the unavailable examples in the Training Set Browser.\n\n'
                + '\n'.join(missing[:10]))
            return
        payload = {
            'format_version': MODEL_FORMAT_VERSION,
            'embedding_id': EMBEDDING_ID,
            'classifier': self.model,
            'classes': self.class_names(),
            # Original keys are restored from the archive manifest on import.
            'training_bank': self.training_bank,
            'source_directory': str(self.directory.resolve()) if self.directory else None,
            'labels': dict(self.labels),
            'confidence_threshold': self.min_conf.value(),
        }
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=str(path.parent), prefix='.gicmodel_',
                                             suffix='.tmp', delete=False) as f:
                tmp_path = Path(f.name)
            manifest = {}
            with zipfile.ZipFile(tmp_path, 'w', allowZip64=True) as z:
                for i, key in enumerate(sorted(self.training_bank)):
                    source = Path(key)
                    # SHA includes key, preventing collisions between identical filenames.
                    digest = hashlib.sha256(key.encode('utf-8')).hexdigest()
                    member = f'images/{i:06d}_{digest[:16]}{source.suffix.lower()}'
                    # ZIP_STORED preserves the source image byte-for-byte, with no recompression.
                    z.write(source, arcname=member, compress_type=zipfile.ZIP_STORED)
                    manifest[key] = member
                z.writestr('manifest.json', json.dumps({'image_members': manifest}, indent=2))
                buffer = io.BytesIO()
                joblib.dump(payload, buffer, compress=3)
                z.writestr('classifier.joblib', buffer.getvalue(), compress_type=zipfile.ZIP_STORED)
            os.replace(tmp_path, path)
            self.status.setText(f'Model saved with {len(manifest)} full-resolution training images: {path}')
        except Exception as exc:
            QMessageBox.critical(self, 'Save failed', str(exc))
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()

    def load_model(self):
        if self.worker:
            return
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Load trained model', '', 'Guided Image Classifier (*.gicmodel)')
        if not filename:
            return
        if not self.images:
            QMessageBox.information(self, 'Select images first',
                                    'Select an image folder before loading the saved model.')
            return
        reply = QMessageBox.question(
            self, 'Trust this model?',
            'Saved models contain Python pickle data, which can execute code when loaded. '
            'Only open models from trusted sources. Continue?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        temp_images = None
        try:
            if zipfile.is_zipfile(filename):
                temp_images = tempfile.TemporaryDirectory(prefix='guided_classifier_training_')
                with zipfile.ZipFile(filename) as z:
                    payload = joblib.load(io.BytesIO(z.read('classifier.joblib')))
                    manifest = json.loads(z.read('manifest.json'))['image_members']
                    bank = payload.get('training_bank', {})
                    if set(manifest) != set(bank):
                        raise ValueError('Saved image manifest does not match training features.')
                    remapped = {}
                    for i, (old_key, (feature, label)) in enumerate(bank.items()):
                        member = manifest[old_key]
                        if member not in z.namelist() or not member.startswith('images/'):
                            raise ValueError('Invalid training image archive member.')
                        # Keep the current original path if present, otherwise use
                        # an extracted full-resolution copy owned by the application.
                        original = Path(old_key)
                        if original.is_file():
                            new_key = str(original.resolve())
                        else:
                            ext = Path(member).suffix
                            target = Path(temp_images.name) / (f'{i:06d}_{Path(old_key).stem}' + ext)
                            with z.open(member) as src, target.open('wb') as dest:
                                shutil.copyfileobj(src, dest)
                            new_key = str(target.resolve())
                        remapped[new_key] = (np.asarray(feature, dtype=np.float32), label)
                    bank = remapped
            else:
                # Version 3/4 legacy files still load, though they do not contain pixels.
                payload = joblib.load(filename)
                bank = payload.get('training_bank', {})
            if not isinstance(payload, dict) or payload.get('format_version') not in (1, 2):
                raise ValueError('Unsupported model file version.')
            if payload.get('embedding_id') != EMBEDDING_ID:
                raise ValueError('This model uses a different image feature extractor.')
            model = payload['classifier']
            if not hasattr(model, 'predict_proba') or not hasattr(model, 'classes_'):
                raise ValueError('Invalid classifier object.')
            classes = list(payload['classes'])
            if len(classes) < 2 or set(classes) != set(model.classes_):
                raise ValueError('Class metadata does not match the trained classifier.')
            if not isinstance(bank, dict):
                raise ValueError('Invalid training bank in the saved model.')
        except Exception as exc:
            if temp_images is not None:
                temp_images.cleanup()
            QMessageBox.critical(self, 'Load failed', str(exc))
            return
        if self.model_image_temp is not None:
            self.model_image_temp.cleanup()
        self.model_image_temp = temp_images
        self.model = model
        self.training_bank = bank
        self.loaded_model = True
        self.class_list.clear()
        self.class_list.addItems(classes)
        same_folder = payload.get('source_directory') == str(self.directory.resolve())
        self.labels = ({k: v for k, v in payload.get('labels', {}).items()
                        if k in {p.name for p in self.images} and v in classes}
                       if same_folder else {})
        self.feature_cache = None
        self.predictions = {}
        self.errors = []
        self.predictions_stale = False
        self.review_only.setChecked(False)
        self.review_only.setEnabled(False)
        self.export.setEnabled(False)
        self.save_model_btn.setEnabled(True)
        self.min_conf.setValue(float(payload.get('confidence_threshold', 0.65)))
        self.start.setText('Predict / Retrain and Preview')
        self.review_stats.setText('Loaded model. Run prediction before reviewing or exporting.')
        self.status.setText(f'Loaded {len(classes)} classes and {len(bank)} saved training examples.')
        self.refresh()

    def start_classify(self):
        if not self.images:
            QMessageBox.warning(self, 'Missing input', 'Choose an image folder first.')
            return
        names = self.class_names()
        if len(names) < 2:
            QMessageBox.warning(self, 'Missing classes', 'Create at least two classes.')
            return
        # A freshly imported model can be applied without any local training labels.
        # If labels have changed, rebuild using saved training features + current labels.
        available_labels = set(self.labels.values()) | {label for _, label in self.training_bank.values()}
        retrain = self.model is None or self.predictions_stale or (not self.loaded_model)
        if retrain:
            missing = [n for n in names if n not in available_labels]
            if missing:
                QMessageBox.warning(self, 'Missing training examples',
                                    'Label at least one image in each class: ' + ', '.join(missing))
                return
            counts = Counter(self.labels.values()) + Counter(v[1] for v in self.training_bank.values())
            if min(counts.values()) < 5:
                reply = QMessageBox.question(self, 'Small training set',
                                             'One or more classes has fewer than 5 labeled examples. Predictions may be unreliable. Continue?',
                                             QMessageBox.Yes | QMessageBox.No)
                if reply != QMessageBox.Yes:
                    return
        self.worker = ClassificationWorker(self.images, self.labels, self.feature_cache,
                                           self.model, self.training_bank, retrain, self)
        self.worker.progress.connect(self.on_progress)
        self.worker.completed.connect(self.on_predictions)
        self.worker.failed.connect(self.on_failed)
        self.start.setEnabled(False)
        self.export.setEnabled(False)
        self.save_model_btn.setEnabled(False)
        self.load_model_btn.setEnabled(False)
        self.stop.setEnabled(True)
        self.progress.setRange(0, len(self.images))
        self.progress.setValue(0)
        self.worker.start()

    def stop_classify(self):
        if self.worker:
            self.worker.stop()
            self.status.setText('Stop requested...')

    def on_progress(self, current, total, message):
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(current)
        self.status.setText(message)

    def on_predictions(self, result):
        self.worker = None
        self.start.setEnabled(True)
        self.load_model_btn.setEnabled(True)
        self.stop.setEnabled(False)
        if result is None:
            self.status.setText('Stopped. No results changed.')
            self.export.setEnabled(bool(self.predictions) and not self.predictions_stale)
            self.save_model_btn.setEnabled(self.model is not None and not self.predictions_stale)
            return
        self.feature_cache = result['cache']
        self.model = result['model']
        self.training_bank = result['training_bank']
        self.loaded_model = True
        self.start.setText('Predict / Retrain and Preview')
        self.save_model_btn.setEnabled(True)
        self.predictions = result['predictions']
        self.errors = result['errors']
        self.predictions_stale = False
        self.review_only.setEnabled(True)
        self.export.setEnabled(True)
        self.update_review_stats()
        self.review_only.setChecked(True)
        self.on_review_toggled()
        self.status.setText('Predictions ready. Review uncertain images before exporting.')

    def export_results(self):
        if not self.predictions or self.predictions_stale or self.worker:
            QMessageBox.warning(self, 'Retrain required', 'Train or retrain before exporting.')
            return
        directory = QFileDialog.getExistingDirectory(self, 'Choose export folder (separate from image source)')
        if not directory:
            return
        output = Path(directory).resolve()
        if output == self.directory.resolve():
            QMessageBox.warning(self, 'Invalid output', 'Choose a folder other than the input directory.')
            return
        self.worker = ExportWorker(self.images, self.labels, self.predictions, self.errors,
                                   output, self.min_conf.value(), self)
        self.worker.progress.connect(self.on_progress)
        self.worker.completed.connect(self.on_exported)
        self.worker.failed.connect(self.on_failed)
        self.start.setEnabled(False)
        self.export.setEnabled(False)
        self.stop.setEnabled(True)
        self.load_model_btn.setEnabled(False)
        self.save_model_btn.setEnabled(False)
        self.progress.setValue(0)
        self.worker.start()

    def on_exported(self, message):
        self.worker = None
        self.start.setEnabled(True)
        self.load_model_btn.setEnabled(True)
        self.save_model_btn.setEnabled(self.model is not None and not self.predictions_stale)
        self.export.setEnabled(True)
        self.stop.setEnabled(False)
        self.status.setText(message)
        QMessageBox.information(self, 'Export results', message)

    def on_failed(self, tb):
        self.start.setEnabled(True)
        self.load_model_btn.setEnabled(True)
        self.save_model_btn.setEnabled(self.model is not None and not self.predictions_stale)
        self.export.setEnabled(bool(self.predictions) and not self.predictions_stale)
        self.stop.setEnabled(False)
        self.status.setText('Error: ' + tb.splitlines()[-1])
        QMessageBox.critical(self, 'Operation failed', tb)
        self.worker = None

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, 'Classification in progress', 'Stop classification before closing the application.')
            event.ignore()
            return
        super().closeEvent(event)
        if self.model_image_temp is not None:
            self.model_image_temp.cleanup()
            self.model_image_temp = None


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
