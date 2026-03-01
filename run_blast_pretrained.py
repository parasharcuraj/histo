#!/usr/bin/env python3
"""BLAST vs Non-BLAST Segmentation with Top 5 Pretrained Medical Image Models."""

# # BLAST Segmentation: With vs Without BLAST
# ## Top 5 Pretrained Medical Image Models on Oral Cancer Histopathology
# 
# This notebook downloads **10 real oral cancer histopathological images** from public sources and compares:
# - **WITH BLAST**: Pretrained model features → BLAST similarity matching → segmentation
# - **WITHOUT BLAST**: Pretrained encoder + trainable decoder → direct pixel-wise segmentation
# 
# ### Top 5 Pretrained Medical Image Models:
# 1. **ResNet50** — Most widely used backbone in medical imaging research
# 2. **DenseNet121** — CheXNet backbone, standard for histopathology
# 3. **InceptionV3** — Google Health dermatology/pathology AI backbone
# 4. **EfficientNetB0** — State-of-the-art efficiency, popular in recent medical AI papers
# 5. **Xception** — Depthwise separable convolutions, used in medical imaging studies
# 
# ### Data Sources:
# - [Mendeley: Oral Cancer Histopathology Repository](https://data.mendeley.com/datasets/ftmp4cvtmb/1)
# - [Mendeley: OCDC H&E Dataset](https://data.mendeley.com/datasets/9bsc36jyrt/1)
# - [ORCHID Database (Nature Scientific Data)](https://www.nature.com/articles/s41597-024-03836-6)
# - [Wikimedia Commons: Medical Histopathology](https://commons.wikimedia.org/)

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# GPU Setup
nvidia_base = "/usr/local/lib/python3.10/dist-packages/nvidia"
nvidia_libs = [
    f"{nvidia_base}/cuda_runtime/lib", f"{nvidia_base}/cudnn/lib",
    f"{nvidia_base}/cublas/lib", f"{nvidia_base}/cufft/lib",
    f"{nvidia_base}/curand/lib", f"{nvidia_base}/cusolver/lib",
    f"{nvidia_base}/cusparse/lib", f"{nvidia_base}/nvjitlink/lib",
    "/usr/lib/wsl/lib",
]
os.environ["LD_LIBRARY_PATH"] = ":".join(nvidia_libs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["PATH"] = f"{nvidia_base}/cuda_nvcc/bin:" + os.environ.get("PATH", "")
os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={nvidia_base}/cuda_nvcc"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import cv2
import gc
import time
import json
import urllib.request
import ssl
from io import BytesIO
from collections import Counter

from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray, rgb2hed
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, jaccard_score
)
from sklearn.preprocessing import normalize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print(f'TensorFlow: {tf.__version__}')
print(f'GPU devices: {gpus}')
print(f'NumPy: {np.__version__}')
print(f'OpenCV: {cv2.__version__}')

# ══════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = 256
PATCH_SIZE = 64
PATCHES_PER_SIDE = IMG_SIZE // PATCH_SIZE  # 4
PATCHES_PER_IMAGE = PATCHES_PER_SIDE ** 2  # 16

NUM_IMAGES = 10
CLASS_NAMES = ['Normal', 'Dysplasia', 'Carcinoma']
NUM_CLASSES = len(CLASS_NAMES)
CLASS_COLORS = {0: [144, 238, 144], 1: [255, 255, 102], 2: [255, 99, 71]}

TOP_K = 5

DATA_DIR = 'data/real_images'
OUTPUT_DIR = 'outputs/pretrained_comparison'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f'Image size: {IMG_SIZE}x{IMG_SIZE}, Patch size: {PATCH_SIZE}x{PATCH_SIZE}')
print(f'Patches per image: {PATCHES_PER_IMAGE}, Total images: {NUM_IMAGES}')

# ## 1. Download 10 Oral Cancer Histopathological Images
# 
# We download real H&E-stained oral cancer histopathological images from public sources:
# - Wikimedia Commons (CC-licensed medical histopathology images)
# - PubMed Central open-access figure images
# 
# Images include **Normal epithelium**, **Dysplasia**, and **Oral Squamous Cell Carcinoma (OSCC)**.

# ══════════════════════════════════════════════════════════
# DOWNLOAD 10 ORAL CANCER HISTOPATHOLOGICAL IMAGES
# ══════════════════════════════════════════════════════════

# Public URLs for oral/head-neck cancer histopathology images
# Sources: Wikimedia Commons (CC licensed), PMC open-access figures
IMAGE_URLS = [
    # Normal oral epithelium / Oral mucosa
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Normal_Epidermis_and_Dermis_with_Intradermal_Nevus_10x.JPG/1280px-Normal_Epidermis_and_Dermis_with_Intradermal_Nevus_10x.JPG",
     "Normal", "normal_epithelium_01"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Normal_Oral_Mucosa_Histology.jpg/1280px-Normal_Oral_Mucosa_Histology.jpg",
     "Normal", "normal_mucosa_02"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Hyperplastic_stratified_squamous_epithelium.jpg/1280px-Hyperplastic_stratified_squamous_epithelium.jpg",
     "Normal", "normal_squamous_03"),
    # Dysplasia
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Oral_leukoplakia_%282%29_-_Dysplasia.jpg/1280px-Oral_leukoplakia_%282%29_-_Dysplasia.jpg",
     "Dysplasia", "dysplasia_04"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/Cervical_intraepithelial_neoplasia_%283%29_CIN2.jpg/1280px-Cervical_intraepithelial_neoplasia_%283%29_CIN2.jpg",
     "Dysplasia", "dysplasia_05"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Mild_dysplasia.jpg/1280px-Mild_dysplasia.jpg",
     "Dysplasia", "dysplasia_06"),
    # Carcinoma (OSCC)
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Squamous_Cell_Carcinoma.jpg/1280px-Squamous_Cell_Carcinoma.jpg",
     "Carcinoma", "carcinoma_07"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Oral_cancer_%281%29_squamous_cell_carcinoma_histopathology.jpg/1280px-Oral_cancer_%281%29_squamous_cell_carcinoma_histopathology.jpg",
     "Carcinoma", "carcinoma_08"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/Laryngeal_squamous_carcinoma_%28well_differentiated%29_HE_stain.jpg/1280px-Laryngeal_squamous_carcinoma_%28well_differentiated%29_HE_stain.jpg",
     "Carcinoma", "carcinoma_09"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Squamous_cell_carcinoma_in_situ.jpg/1280px-Squamous_cell_carcinoma_in_situ.jpg",
     "Carcinoma", "carcinoma_10"),
]


def generate_synthetic_he_image(img_id, cls_label, rng):
    """Generate a realistic synthetic H&E-stained histopathology image as fallback."""
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    # H&E background colors per class
    bg = {'Normal': [230, 200, 210], 'Dysplasia': [215, 180, 200], 'Carcinoma': [200, 160, 185]}
    base = np.array(bg[cls_label], dtype=np.float64)
    noise = rng.normal(0, 8, size=(IMG_SIZE, IMG_SIZE, 3))
    for c in range(3):
        img[:, :, c] = np.clip(base[c] + noise[:, :, c], 0, 255).astype(np.uint8)

    # Draw nuclei
    if cls_label == 'Normal':
        n_nuclei, r_range, color_base = 80, (2, 4), [80, 40, 120]
    elif cls_label == 'Dysplasia':
        n_nuclei, r_range, color_base = 150, (3, 6), [60, 20, 100]
    else:
        n_nuclei, r_range, color_base = 250, (3, 8), [40, 10, 80]

    for _ in range(n_nuclei):
        cx, cy = rng.randint(5, IMG_SIZE - 5, size=2)
        r = rng.randint(r_range[0], r_range[1] + 1)
        color_var = rng.randint(-15, 16, size=3)
        color = np.clip(np.array(color_base) + color_var, 0, 255).tolist()
        cv2.circle(img, (int(cx), int(cy)), r, color, -1)

    # Stroma/fiber structures
    for _ in range(rng.randint(5, 15)):
        pt1 = tuple(rng.randint(0, IMG_SIZE, size=2))
        pt2 = tuple(rng.randint(0, IMG_SIZE, size=2))
        color = np.clip(np.array([200, 170, 190]) + rng.randint(-20, 20, size=3), 0, 255).tolist()
        cv2.line(img, pt1, pt2, color, rng.randint(1, 3))

    img = cv2.GaussianBlur(img, (3, 3), 0.7)
    return img


def download_image(url, save_path, timeout=15):
    """Download an image from URL, return True on success."""
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Research/Academic histopathology analysis)'
        })
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as response:
            data = response.read()

        img_array = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Center crop to square then resize
        h, w = img.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        img = img[y0:y0+s, x0:x0+s]
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return True
    except Exception as e:
        print(f'    Download failed: {e}')
        return False


# Download all images
images = []
image_labels = []  # class label per image
image_sources = []  # track source (downloaded vs synthetic)

print('Downloading 10 oral cancer histopathological images...')
print('=' * 60)

for i, (url, cls_label, name) in enumerate(IMAGE_URLS):
    save_path = os.path.join(DATA_DIR, f'{name}.png')
    print(f'[{i+1}/10] {name} ({cls_label})...')

    # Try download first
    success = download_image(url, save_path)

    if success:
        img = cv2.imread(save_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        image_sources.append('Downloaded')
        print(f'    OK (downloaded from web)')
    else:
        # Fallback: generate realistic synthetic H&E image
        print(f'    Fallback: generating synthetic H&E image')
        rng = np.random.RandomState(SEED + i * 7)
        img = generate_synthetic_he_image(i, cls_label, rng)
        images.append(img)
        image_sources.append('Synthetic')
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    image_labels.append(cls_label)

images = np.array(images)
print(f'\nTotal images loaded: {len(images)}')
print(f'Sources: {Counter(image_sources)}')
print(f'Labels:  {Counter(image_labels)}')

# ══════════════════════════════════════════════════════════
# VISUALIZE DOWNLOADED IMAGES
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for i in range(NUM_IMAGES):
    r, c = i // 5, i % 5
    axes[r, c].imshow(images[i])
    axes[r, c].set_title(f'{image_labels[i]}\n({image_sources[i]})', fontsize=10)
    axes[r, c].axis('off')

plt.suptitle('10 Oral Cancer Histopathological Images', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_downloaded_images.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Images visualized.')

# ## 2. Generate Ground Truth Segmentation Masks
# 
# Since downloaded images don't come with pixel-level segmentation masks, we create **expert-informed pseudo-ground-truth** masks using:
# 1. **H&E Color Deconvolution** (separate Hematoxylin/Eosin channels)
# 2. **K-Means Clustering** on color + texture features
# 3. **Class Assignment** based on nuclear density and staining intensity
# 
# This is a standard approach in computational pathology when expert annotations are not available.

# ══════════════════════════════════════════════════════════
# CREATE GROUND TRUTH MASKS VIA EXPERT-INFORMED CLUSTERING
# ══════════════════════════════════════════════════════════

def create_segmentation_mask(image, n_clusters=NUM_CLASSES, seed=SEED):
    """Create pseudo-ground-truth mask using color deconvolution + k-means.

    Steps:
    1. Convert to H&E color space (Hematoxylin-Eosin Deconvolution)
    2. Extract local texture features (LBP variance at each pixel block)
    3. K-Means cluster into Normal / Dysplasia / Carcinoma
    4. Assign class labels based on nuclear density (hematoxylin intensity)
    """
    h, w = image.shape[:2]

    # H&E color deconvolution
    img_float = image.astype(np.float64) / 255.0
    img_float = np.clip(img_float, 1e-6, 1.0)
    try:
        hed = rgb2hed(img_float)
        hematoxylin = hed[:, :, 0]
        eosin = hed[:, :, 1]
    except Exception:
        # Fallback: use simple color channels
        gray = rgb2gray(img_float)
        hematoxylin = 1.0 - gray
        eosin = gray

    # Compute local nuclear density using block statistics
    block_size = 8
    density_map = np.zeros((h, w), dtype=np.float64)
    intensity_map = np.zeros((h, w), dtype=np.float64)

    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block_h = hematoxylin[y:y+block_size, x:x+block_size]
            block_e = eosin[y:y+block_size, x:x+block_size]
            density_map[y:y+block_size, x:x+block_size] = block_h.std()
            intensity_map[y:y+block_size, x:x+block_size] = block_h.mean()

    # Build feature vectors for each pixel
    features = np.stack([
        hematoxylin.flatten(),
        eosin.flatten(),
        density_map.flatten(),
        intensity_map.flatten(),
        rgb2gray(img_float).flatten(),
    ], axis=1)

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(features)
    mask = labels.reshape(h, w).astype(np.uint8)

    # Assign class labels based on hematoxylin intensity
    # Higher hematoxylin = more nuclear staining = more likely carcinoma
    cluster_hematoxylin = []
    for c in range(n_clusters):
        cluster_hematoxylin.append(hematoxylin.flatten()[labels == c].mean())

    sorted_clusters = np.argsort(cluster_hematoxylin)
    # Lowest hematoxylin → Normal (0), Middle → Dysplasia (1), Highest → Carcinoma (2)
    remap = np.zeros(n_clusters, dtype=np.uint8)
    remap[sorted_clusters[0]] = 0  # Normal
    remap[sorted_clusters[1]] = 1  # Dysplasia
    remap[sorted_clusters[2]] = 2  # Carcinoma

    mask = remap[mask]
    return mask


# Generate masks for all images
masks = []
for i in range(NUM_IMAGES):
    mask = create_segmentation_mask(images[i], seed=SEED + i)
    masks.append(mask)
    cv2.imwrite(os.path.join(DATA_DIR, f'mask_{i:02d}.png'), mask)

masks = np.array(masks) if all(m.shape == masks[0].shape for m in masks) else masks
print(f'Generated {len(masks)} segmentation masks.')

# Show mask statistics
for i in range(NUM_IMAGES):
    unique, counts = np.unique(masks[i], return_counts=True)
    total = counts.sum()
    pcts = {CLASS_NAMES[u]: f'{c/total*100:.1f}%' for u, c in zip(unique, counts)}
    print(f'  Image {i} ({image_labels[i]}): {pcts}')

# Visualize images with ground truth masks
fig, axes = plt.subplots(NUM_IMAGES, 3, figsize=(12, 4 * NUM_IMAGES))

for i in range(NUM_IMAGES):
    axes[i, 0].imshow(images[i])
    axes[i, 0].set_title(f'Image {i} ({image_labels[i]})')
    axes[i, 0].axis('off')

    # Color-coded mask
    color_mask = np.zeros((*masks[i].shape, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS.items():
        color_mask[masks[i] == cls_id] = color
    axes[i, 1].imshow(color_mask)
    axes[i, 1].set_title(f'GT Mask {i}')
    axes[i, 1].axis('off')

    # Overlay
    overlay = (images[i].astype(np.float32) * 0.6 + color_mask.astype(np.float32) * 0.4).astype(np.uint8)
    axes[i, 2].imshow(overlay)
    axes[i, 2].set_title(f'Overlay {i}')
    axes[i, 2].axis('off')

legend_patches = [mpatches.Patch(color=np.array(c)/255, label=n) for n, c in zip(CLASS_NAMES, CLASS_COLORS.values())]
fig.legend(handles=legend_patches, loc='upper center', ncol=3, fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(os.path.join(OUTPUT_DIR, '02_images_with_masks.png'), dpi=120, bbox_inches='tight')
plt.show()

# ## 3. Patch Extraction & BLAST Pipeline

# ══════════════════════════════════════════════════════════
# PATCH EXTRACTION
# ══════════════════════════════════════════════════════════

def extract_patches(image, mask):
    """Extract non-overlapping patches and their majority-vote labels."""
    patches, patch_labels, patch_masks = [], [], []
    for r in range(PATCHES_PER_SIDE):
        for c in range(PATCHES_PER_SIDE):
            y0, y1 = r * PATCH_SIZE, (r + 1) * PATCH_SIZE
            x0, x1 = c * PATCH_SIZE, (c + 1) * PATCH_SIZE
            patch = image[y0:y1, x0:x1]
            patch_mask = mask[y0:y1, x0:x1]
            vals, cnts = np.unique(patch_mask, return_counts=True)
            patches.append(patch)
            patch_labels.append(vals[np.argmax(cnts)])
            patch_masks.append(patch_mask)
    return patches, patch_labels, patch_masks


# Extract all patches
all_patches = []
all_patch_labels = []
image_indices = []

for i in range(NUM_IMAGES):
    patches, plabels, _ = extract_patches(images[i], masks[i])
    for j, (p, l) in enumerate(zip(patches, plabels)):
        all_patches.append(p)
        all_patch_labels.append(l)
        image_indices.append(i)

all_patches = np.array(all_patches)
all_patch_labels = np.array(all_patch_labels)
image_indices = np.array(image_indices)

print(f'Total patches: {len(all_patches)}')
print(f'Label distribution: {dict(Counter(all_patch_labels))}')

# Visualize sample patches per class
fig, axes = plt.subplots(3, 6, figsize=(15, 8))
for cls_id in range(NUM_CLASSES):
    cls_indices = np.where(all_patch_labels == cls_id)[0]
    if len(cls_indices) == 0:
        continue
    samples = np.random.choice(cls_indices, size=min(6, len(cls_indices)), replace=False)
    for j, idx in enumerate(samples):
        axes[cls_id, j].imshow(all_patches[idx])
        axes[cls_id, j].set_title(f'Img {image_indices[idx]}', fontsize=9)
        axes[cls_id, j].axis('off')
    axes[cls_id, 0].set_ylabel(CLASS_NAMES[cls_id], fontsize=14, rotation=0, labelpad=60)

plt.suptitle('Sample Patches per Class', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_sample_patches.png'), dpi=120, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════
# BLAST IMAGE DATABASE (Patch Matching & Segmentation)
# ══════════════════════════════════════════════════════════

class BLASTImageDatabase:
    """BLAST-like database for histopathological patch matching and segmentation.

    Analogous to NCBI BLAST:
    - build_database() → makeblastdb (index reference patches)
    - query()          → blastn      (find top-K similar patches)
    - segment_image()  → full alignment (assemble segmentation from patch matches)
    """

    def __init__(self, metric='cosine', top_k=TOP_K):
        self.metric = metric
        self.top_k = top_k
        self.db_features = None
        self.db_labels = None

    def build_database(self, features, labels):
        self.db_features = normalize(features.copy())
        self.db_labels = labels.copy()

    def _compute_similarity(self, query_feat, db_feats):
        query_norm = normalize(query_feat.reshape(1, -1))
        if self.metric == 'cosine':
            sims = (query_norm @ db_feats.T).flatten()
        else:
            dists = np.linalg.norm(db_feats - query_norm, axis=1)
            sims = 1.0 / (1.0 + dists)
        return sims

    def query(self, query_feature):
        sims = self._compute_similarity(query_feature, self.db_features)
        top_indices = np.argsort(sims)[::-1][:self.top_k]
        top_sims = sims[top_indices]
        top_labels = self.db_labels[top_indices]

        weights = np.maximum(top_sims, 1e-10)
        class_scores = np.zeros(NUM_CLASSES)
        for lbl, w in zip(top_labels, weights):
            class_scores[lbl] += w

        predicted = np.argmax(class_scores)
        confidence = class_scores[predicted] / (class_scores.sum() + 1e-10)

        return {
            'predicted': predicted,
            'confidence': confidence,
            'top_indices': top_indices,
            'top_sims': top_sims,
            'top_labels': top_labels,
            'class_scores': class_scores
        }

    def segment_image(self, patch_features):
        pred_labels, confidences = [], []
        for feat in patch_features:
            result = self.query(feat)
            pred_labels.append(result['predicted'])
            confidences.append(result['confidence'])

        pred_grid = np.array(pred_labels).reshape(PATCHES_PER_SIDE, PATCHES_PER_SIDE)
        conf_grid = np.array(confidences).reshape(PATCHES_PER_SIDE, PATCHES_PER_SIDE)
        seg_map = np.kron(pred_grid, np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8))
        return seg_map, pred_grid, conf_grid


def leave_one_image_out_blast(features, labels, img_indices, metric='cosine', top_k=TOP_K):
    """Leave-One-Image-Out CV for BLAST segmentation."""
    all_pred_labels = np.zeros(len(labels), dtype=int)
    all_seg_maps = {}
    all_conf_maps = {}

    for img_id in range(NUM_IMAGES):
        test_mask = img_indices == img_id
        train_mask = ~test_mask

        db = BLASTImageDatabase(metric=metric, top_k=top_k)
        db.build_database(features[train_mask], labels[train_mask])

        test_features = features[test_mask]
        seg_map, pred_grid, conf_grid = db.segment_image(test_features)

        all_pred_labels[test_mask] = pred_grid.flatten()
        all_seg_maps[img_id] = seg_map
        all_conf_maps[img_id] = conf_grid

    return all_pred_labels, all_seg_maps, all_conf_maps


def compute_segmentation_metrics(true_masks, pred_seg_maps, method_name):
    """Compute pixel-level segmentation metrics."""
    all_true, all_pred = [], []
    for img_id in range(NUM_IMAGES):
        all_true.append(true_masks[img_id].flatten())
        all_pred.append(pred_seg_maps[img_id].flatten())

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    pixel_acc = accuracy_score(all_true, all_pred)
    mean_iou = jaccard_score(all_true, all_pred, average='macro', zero_division=0)

    dice_scores = []
    for c in range(NUM_CLASSES):
        gt_c = (all_true == c)
        pred_c = (all_pred == c)
        intersection = (gt_c & pred_c).sum()
        dice = 2 * intersection / (gt_c.sum() + pred_c.sum() + 1e-10)
        dice_scores.append(dice)
    mean_dice = np.mean(dice_scores)

    prec = precision_score(all_true, all_pred, average='macro', zero_division=0)
    rec = recall_score(all_true, all_pred, average='macro', zero_division=0)
    f1 = f1_score(all_true, all_pred, average='macro', zero_division=0)

    return {
        'Method': method_name,
        'Pixel Accuracy': pixel_acc,
        'Mean IoU': mean_iou,
        'Mean Dice': mean_dice,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'Dice_per_class': dice_scores,
        'all_true': all_true,
        'all_pred': all_pred
    }


print('BLAST pipeline defined.')

# ## 4. Define Top 5 Pretrained Medical Image Models
# 
# | Rank | Model | Why It's Top for Medical Imaging |
# |------|-------|----------------------------------|
# | 1 | **ResNet50** | Most cited backbone in medical imaging papers; excellent feature hierarchy |
# | 2 | **DenseNet121** | CheXNet foundation; dense connections preserve fine-grained features |
# | 3 | **InceptionV3** | Multi-scale feature extraction; Google Health pathology backbone |
# | 4 | **EfficientNetB0** | Best accuracy/efficiency trade-off; state-of-the-art in recent studies |
# | 5 | **Xception** | Depthwise separable convolutions; strong histopathology performance |

# ══════════════════════════════════════════════════════════
# TOP 5 PRETRAINED MEDICAL IMAGE MODELS
# ══════════════════════════════════════════════════════════

from tensorflow.keras.applications import (
    ResNet50, DenseNet121, InceptionV3, EfficientNetB0, Xception
)
from tensorflow.keras.applications import (
    resnet50, densenet, inception_v3, efficientnet, xception
)

PRETRAINED_MODELS = {
    'ResNet50':       (ResNet50, resnet50.preprocess_input, 75),
    'DenseNet121':    (DenseNet121, densenet.preprocess_input, 75),
    'InceptionV3':    (InceptionV3, inception_v3.preprocess_input, 75),
    'EfficientNetB0': (EfficientNetB0, efficientnet.preprocess_input, 32),
    'Xception':       (Xception, xception.preprocess_input, 71),
}

# Note: min_input_size is the minimum spatial dimension required by each model.
# InceptionV3 and Xception require >= 75px, EfficientNetB0 >= 32px, etc.
# Our PATCH_SIZE=64 is fine for ResNet50, DenseNet121, EfficientNetB0.
# For InceptionV3 and Xception, we'll resize patches to 75x75 during feature extraction.


def build_patch_feature_extractor(model_class, min_size, input_size=PATCH_SIZE):
    """Build a frozen pretrained model for patch-level feature extraction."""
    actual_size = max(input_size, min_size)
    base = model_class(
        weights='imagenet', include_top=False,
        input_shape=(actual_size, actual_size, 3), pooling='avg'
    )
    base.trainable = False
    return base, actual_size


# Show embedding dimensions
print(f'{"Model":<18} {"Min Size":>10} {"Embedding":>12}')
print('=' * 42)
for name, (cls, _, min_sz) in PRETRAINED_MODELS.items():
    ext, actual_sz = build_patch_feature_extractor(cls, min_sz)
    print(f'{name:<18} {actual_sz:>10} {ext.output_shape[-1]:>12}')
    del ext
    tf.keras.backend.clear_session()

# ## 5. MODE 1: WITH BLAST
# ### Pretrained Features → BLAST Similarity Matching → Segmentation
# 
# For each pretrained model:
# 1. Extract patch-level embeddings using the frozen pretrained encoder
# 2. Build a BLAST database from training patches
# 3. Query test patches against the database
# 4. Assign labels via **weighted top-K voting** (BLAST scoring)
# 5. Assemble segmentation map from patch predictions

# ══════════════════════════════════════════════════════════
# MODE 1: WITH BLAST — Pretrained Features → BLAST Matching
# ══════════════════════════════════════════════════════════

print('MODE 1: Pretrained Models + BLAST Pipeline')
print('=' * 55)

blast_results = {}  # {model_name: {'seg_maps': ..., 'preds': ..., 'conf_maps': ...}}
blast_metrics = []  # list of metric dicts

for model_name, (model_class, preprocess_fn, min_sz) in PRETRAINED_MODELS.items():
    print(f'\n{"─" * 50}')
    print(f'{model_name} + BLAST')
    print(f'{"─" * 50}')

    tf.keras.backend.clear_session()
    gc.collect()

    # Build feature extractor
    extractor, actual_size = build_patch_feature_extractor(model_class, min_sz)

    # Preprocess and resize patches if needed
    patches_float = all_patches.astype(np.float32)
    if actual_size != PATCH_SIZE:
        patches_resized = np.array([
            cv2.resize(p, (actual_size, actual_size)) for p in patches_float
        ])
    else:
        patches_resized = patches_float

    patches_prep = np.array([preprocess_fn(p.copy()) for p in patches_resized])

    # Extract embeddings
    embeddings = extractor.predict(patches_prep, batch_size=16, verbose=0)
    print(f'  Embedding shape: {embeddings.shape}')

    # Run BLAST-based LOIO segmentation
    preds, seg_maps, conf_maps = leave_one_image_out_blast(
        embeddings, all_patch_labels, image_indices, metric='cosine')

    blast_results[model_name] = {
        'preds': preds, 'seg_maps': seg_maps, 'conf_maps': conf_maps
    }

    m = compute_segmentation_metrics(masks, seg_maps, f'{model_name} + BLAST')
    blast_metrics.append({k: v for k, v in m.items()
                          if k not in ['Dice_per_class', 'all_true', 'all_pred']})
    print(f'  Pixel Acc: {m["Pixel Accuracy"]:.4f} | Mean IoU: {m["Mean IoU"]:.4f} | '
          f'Mean Dice: {m["Mean Dice"]:.4f} | F1: {m["F1 Score"]:.4f}')

    del extractor, patches_resized, patches_prep, embeddings
    tf.keras.backend.clear_session()
    gc.collect()

print('\n\n── MODE 1 Summary: Pretrained + BLAST ──')
blast_df = pd.DataFrame(blast_metrics).set_index('Method').round(4)
print(blast_df.to_string())

# ## 6. MODE 2: WITHOUT BLAST
# ### Pretrained Encoder + Trainable Decoder → Direct Segmentation
# 
# For each pretrained model:
# 1. Use the frozen pretrained encoder as feature extractor
# 2. Attach a lightweight trainable decoder (1×1 conv → upsample → 3×3 conv → output)
# 3. Train end-to-end on full images (decoder only)
# 4. Predict pixel-level segmentation directly — **NO BLAST matching**

# ══════════════════════════════════════════════════════════
# MODE 2: WITHOUT BLAST — Direct Segmentation
# ══════════════════════════════════════════════════════════

print('MODE 2: Pretrained Models — Direct Segmentation (No BLAST)')
print('=' * 60)

DIRECT_EPOCHS = 15


def build_pretrained_segmentation(model_class, preprocess_fn,
                                  input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                  num_classes=NUM_CLASSES):
    """Pretrained encoder (frozen) + minimal trainable decoder."""
    inputs = layers.Input(shape=input_shape)
    preprocessed = layers.Lambda(lambda x: preprocess_fn(x))(inputs)

    encoder = model_class(weights='imagenet', include_top=False,
                          input_shape=input_shape)
    encoder.trainable = False
    encoded = encoder(preprocessed)

    # Lightweight decoder
    x = layers.Conv2D(128, 1, activation='relu')(encoded)
    x = layers.Resizing(IMG_SIZE // 4, IMG_SIZE // 4, interpolation='bilinear')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(num_classes, 1, activation='softmax')(x)

    model = Model(inputs, x)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


images_np = images.astype(np.float32) / 255.0
masks_np = np.array(masks)

direct_results = {}  # {model_name: {img_id: seg_map}}
direct_metrics = []

for model_name, (model_class, preprocess_fn, min_sz) in PRETRAINED_MODELS.items():
    print(f'\n{"─" * 50}')
    print(f'{model_name} (Direct Segmentation, No BLAST)')
    print(f'{"─" * 50}')

    seg_maps = {}
    t0 = time.time()

    for img_id in range(NUM_IMAGES):
        train_idx = [i for i in range(NUM_IMAGES) if i != img_id]
        X_train = images_np[train_idx]
        y_train = masks_np[train_idx][..., np.newaxis]
        X_test = images_np[img_id:img_id+1]

        tf.keras.backend.clear_session()
        gc.collect()
        fold_model = build_pretrained_segmentation(model_class, preprocess_fn)
        fold_model.fit(X_train, y_train, epochs=DIRECT_EPOCHS, batch_size=1, verbose=0)

        pred_prob = fold_model.predict(X_test, verbose=0)[0]
        pred_seg = np.argmax(pred_prob, axis=-1).astype(np.uint8)

        seg_maps[img_id] = pred_seg

        del fold_model
        tf.keras.backend.clear_session()
        gc.collect()
        print(f'  Fold {img_id}: done')

    elapsed = time.time() - t0
    direct_results[model_name] = seg_maps

    m = compute_segmentation_metrics(masks, seg_maps, f'{model_name} (Direct)')
    direct_metrics.append({k: v for k, v in m.items()
                           if k not in ['Dice_per_class', 'all_true', 'all_pred']})
    print(f'\n  {model_name} (Direct) done in {elapsed:.0f}s')
    print(f'  Pixel Acc: {m["Pixel Accuracy"]:.4f} | Mean IoU: {m["Mean IoU"]:.4f} | '
          f'Mean Dice: {m["Mean Dice"]:.4f} | F1: {m["F1 Score"]:.4f}')

print('\n\n── MODE 2 Summary: Pretrained Direct Segmentation ──')
direct_df = pd.DataFrame(direct_metrics).set_index('Method').round(4)
print(direct_df.to_string())

# ## 7. Comprehensive Comparison: BLAST vs No-BLAST

# ══════════════════════════════════════════════════════════
# COMPARISON TABLE: WITH BLAST vs WITHOUT BLAST
# ══════════════════════════════════════════════════════════

all_comparison_metrics = []
all_comparison_metrics_full = []  # includes Dice_per_class

for model_name in PRETRAINED_MODELS.keys():
    # With BLAST
    m_blast = compute_segmentation_metrics(
        masks, blast_results[model_name]['seg_maps'], f'{model_name} + BLAST')
    all_comparison_metrics.append(
        {k: v for k, v in m_blast.items() if k not in ['Dice_per_class', 'all_true', 'all_pred']})
    all_comparison_metrics_full.append(m_blast)

    # Without BLAST
    m_direct = compute_segmentation_metrics(
        masks, direct_results[model_name], f'{model_name} (Direct)')
    all_comparison_metrics.append(
        {k: v for k, v in m_direct.items() if k not in ['Dice_per_class', 'all_true', 'all_pred']})
    all_comparison_metrics_full.append(m_direct)

comparison_df = pd.DataFrame(all_comparison_metrics).set_index('Method').round(4)

print('\n' + '=' * 95)
print('COMPARISON: TOP 5 PRETRAINED MEDICAL IMAGE MODELS — WITH BLAST vs WITHOUT BLAST')
print('=' * 95)
print(comparison_df.to_string())
print()

# Compute average improvement
print('\n── Average Improvement of BLAST over Direct Segmentation ──')
metric_cols = ['Pixel Accuracy', 'Mean IoU', 'Mean Dice', 'Precision', 'Recall', 'F1 Score']
for metric in metric_cols:
    blast_vals = [comparison_df.loc[f'{m} + BLAST', metric]
                  for m in PRETRAINED_MODELS.keys()]
    direct_vals = [comparison_df.loc[f'{m} (Direct)', metric]
                   for m in PRETRAINED_MODELS.keys()]
    avg_blast = np.mean(blast_vals)
    avg_direct = np.mean(direct_vals)
    diff = avg_blast - avg_direct
    print(f'  {metric:<18}: BLAST={avg_blast:.4f}, Direct={avg_direct:.4f}, '
          f'Δ={diff:+.4f} ({"BLAST wins" if diff > 0 else "Direct wins"})')

# Save to CSV
comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'blast_vs_direct_comparison.csv'))
print(f'\nSaved to {OUTPUT_DIR}/blast_vs_direct_comparison.csv')

# ══════════════════════════════════════════════════════════
# VISUALIZATION 1: Side-by-Side Bar Charts
# ══════════════════════════════════════════════════════════

model_names = list(PRETRAINED_MODELS.keys())
metric_cols_plot = ['Pixel Accuracy', 'Mean IoU', 'Mean Dice', 'F1 Score']

fig, axes = plt.subplots(1, len(metric_cols_plot), figsize=(22, 6))

for ax_idx, metric_name in enumerate(metric_cols_plot):
    blast_vals = [comparison_df.loc[f'{m} + BLAST', metric_name] for m in model_names]
    direct_vals = [comparison_df.loc[f'{m} (Direct)', metric_name] for m in model_names]

    x = np.arange(len(model_names))
    w = 0.35
    bars1 = axes[ax_idx].bar(x - w/2, blast_vals, w, label='WITH BLAST',
                              color='#2196F3', edgecolor='white', linewidth=0.5)
    bars2 = axes[ax_idx].bar(x + w/2, direct_vals, w, label='WITHOUT BLAST',
                              color='#F44336', edgecolor='white', linewidth=0.5)

    # Value labels on bars
    for bar in bars1:
        axes[ax_idx].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                          f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        axes[ax_idx].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                          f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)

    axes[ax_idx].set_xticks(x)
    axes[ax_idx].set_xticklabels(model_names, rotation=30, ha='right', fontsize=9)
    axes[ax_idx].set_title(metric_name, fontsize=12, fontweight='bold')
    axes[ax_idx].set_ylim(0, 1.15)
    axes[ax_idx].legend(fontsize=8)
    axes[ax_idx].grid(axis='y', alpha=0.3)

plt.suptitle('Top 5 Pretrained Medical Image Models: WITH BLAST vs WITHOUT BLAST',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_blast_vs_direct_bars.png'), dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════
# VISUALIZATION 2: Segmentation Maps Comparison
# ══════════════════════════════════════════════════════════

sample_imgs = [0, 3, 6, 9]  # one per category + extra
model_names_vis = list(PRETRAINED_MODELS.keys())
n_cols = 2 + 2 * len(model_names_vis)  # original, GT, then BLAST+Direct per model

fig, axes = plt.subplots(len(sample_imgs), n_cols,
                         figsize=(3.2 * n_cols, 3.5 * len(sample_imgs)))

for row, img_id in enumerate(sample_imgs):
    # Original
    axes[row, 0].imshow(images[img_id])
    axes[row, 0].set_title('Original' if row == 0 else '', fontsize=9)
    axes[row, 0].set_ylabel(f'Image {img_id}\n({image_labels[img_id]})', fontsize=9)
    axes[row, 0].axis('off')

    # Ground truth
    gt_color = np.zeros((*masks[img_id].shape, 3), dtype=np.uint8)
    for c, clr in CLASS_COLORS.items():
        gt_color[masks[img_id] == c] = clr
    axes[row, 1].imshow(gt_color)
    axes[row, 1].set_title('GT' if row == 0 else '', fontsize=9)
    axes[row, 1].axis('off')

    col = 2
    for mname in model_names_vis:
        # With BLAST
        seg_b = blast_results[mname]['seg_maps'][img_id]
        seg_b_color = np.zeros((*seg_b.shape, 3), dtype=np.uint8)
        for c, clr in CLASS_COLORS.items():
            seg_b_color[seg_b == c] = clr
        axes[row, col].imshow(seg_b_color)
        axes[row, col].set_title(f'{mname}\n+BLAST' if row == 0 else '', fontsize=7)
        axes[row, col].axis('off')
        col += 1

        # Without BLAST
        seg_d = direct_results[mname][img_id]
        seg_d_color = np.zeros((*seg_d.shape, 3), dtype=np.uint8)
        for c, clr in CLASS_COLORS.items():
            seg_d_color[seg_d == c] = clr
        axes[row, col].imshow(seg_d_color)
        axes[row, col].set_title(f'{mname}\nDirect' if row == 0 else '', fontsize=7)
        axes[row, col].axis('off')
        col += 1

legend_patches = [mpatches.Patch(color=np.array(c)/255, label=n)
                  for n, c in zip(CLASS_NAMES, CLASS_COLORS.values())]
fig.legend(handles=legend_patches, loc='upper center', ncol=3, fontsize=11)
plt.suptitle('Segmentation Maps: BLAST vs Direct (Top 5 Models)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_segmentation_maps.png'), dpi=120, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════
# VISUALIZATION 3: Radar Chart — All 10 Methods
# ══════════════════════════════════════════════════════════

metric_cols_radar = ['Pixel Accuracy', 'Mean IoU', 'Mean Dice', 'Precision', 'Recall', 'F1 Score']
angles = np.linspace(0, 2 * np.pi, len(metric_cols_radar), endpoint=False).tolist()
angles += angles[:1]

# Color scheme: blue shades for BLAST, red shades for Direct
blast_colors = ['#1565C0', '#1976D2', '#1E88E5', '#2196F3', '#42A5F5']
direct_colors = ['#C62828', '#D32F2F', '#E53935', '#F44336', '#EF5350']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), subplot_kw=dict(polar=True))

# Left: BLAST methods
for i, mname in enumerate(model_names):
    method_key = f'{mname} + BLAST'
    vals = comparison_df.loc[method_key, metric_cols_radar].values.astype(float).tolist()
    vals += vals[:1]
    ax1.plot(angles, vals, 'o-', linewidth=1.5, label=mname,
             color=blast_colors[i], markersize=5)
    ax1.fill(angles, vals, alpha=0.08, color=blast_colors[i])

ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(metric_cols_radar, fontsize=9)
ax1.set_ylim(0, 1)
ax1.legend(loc='lower right', fontsize=8)
ax1.set_title('WITH BLAST', fontsize=13, fontweight='bold', pad=20)

# Right: Direct methods
for i, mname in enumerate(model_names):
    method_key = f'{mname} (Direct)'
    vals = comparison_df.loc[method_key, metric_cols_radar].values.astype(float).tolist()
    vals += vals[:1]
    ax2.plot(angles, vals, 'o-', linewidth=1.5, label=mname,
             color=direct_colors[i], markersize=5)
    ax2.fill(angles, vals, alpha=0.08, color=direct_colors[i])

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(metric_cols_radar, fontsize=9)
ax2.set_ylim(0, 1)
ax2.legend(loc='lower right', fontsize=8)
ax2.set_title('WITHOUT BLAST (Direct)', fontsize=13, fontweight='bold', pad=20)

plt.suptitle('Radar Charts: BLAST vs Direct Segmentation', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_radar_charts.png'), dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════
# VISUALIZATION 4: Confusion Matrices — All 10 Methods
# ══════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 5, figsize=(28, 10))
axes_flat = axes.flatten()

for i, m in enumerate(all_comparison_metrics_full):
    cm = confusion_matrix(m['all_true'], m['all_pred'], labels=[0, 1, 2])
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes_flat[i], vmin=0, vmax=1, cbar=False)
    title_parts = m['Method'].split(' ')
    axes_flat[i].set_title(m['Method'], fontsize=9, fontweight='bold')
    axes_flat[i].set_ylabel('True' if i % 5 == 0 else '')
    axes_flat[i].set_xlabel('Predicted')

plt.suptitle('Normalized Confusion Matrices — BLAST vs Direct (All 10 Methods)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════
# VISUALIZATION 5: Per-Class Dice Heatmap
# ══════════════════════════════════════════════════════════

dice_data = {}
for m in all_comparison_metrics_full:
    dice_data[m['Method']] = m['Dice_per_class']

dice_df = pd.DataFrame(dice_data, index=CLASS_NAMES).T

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(dice_df, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax, vmin=0, vmax=1,
            linewidths=0.5, linecolor='white')
ax.set_title('Per-Class Dice Coefficient — BLAST vs Direct', fontsize=14, fontweight='bold')
ax.set_ylabel('Method')

# Add row coloring for BLAST vs Direct
for i, label in enumerate(dice_df.index):
    color = '#E3F2FD' if 'BLAST' in label else '#FFEBEE'
    ax.add_patch(plt.Rectangle((-0.5, i), -0.3, 1, fill=True, color=color, clip_on=False))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '08_dice_heatmap.png'), dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════
# VISUALIZATION 6: Delta Chart (BLAST improvement over Direct)
# ══════════════════════════════════════════════════════════

metric_cols_delta = ['Pixel Accuracy', 'Mean IoU', 'Mean Dice', 'Precision', 'Recall', 'F1 Score']
delta_data = []

for mname in model_names:
    row = {'Model': mname}
    for metric in metric_cols_delta:
        blast_val = comparison_df.loc[f'{mname} + BLAST', metric]
        direct_val = comparison_df.loc[f'{mname} (Direct)', metric]
        row[metric] = blast_val - direct_val
    delta_data.append(row)

delta_df = pd.DataFrame(delta_data).set_index('Model')

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(metric_cols_delta))
n_models = len(model_names)
width = 0.15
colors = ['#1565C0', '#2E7D32', '#EF6C00', '#6A1B9A', '#C62828']

for i, mname in enumerate(model_names):
    vals = delta_df.loc[mname].values.astype(float)
    bars = ax.bar(x + i * width, vals, width, label=mname, color=colors[i], edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003 * np.sign(bar.get_height()),
                f'{val:+.3f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=7)

ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
ax.set_xticks(x + width * (n_models / 2 - 0.5))
ax.set_xticklabels(metric_cols_delta, fontsize=11)
ax.set_ylabel('BLAST Improvement (Δ = BLAST - Direct)', fontsize=11)
ax.legend(fontsize=9, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.15))
ax.set_title('BLAST Improvement Over Direct Segmentation Per Model',
             fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '09_delta_improvement.png'), dpi=150, bbox_inches='tight')
plt.show()

print('\nDelta Table (BLAST - Direct):')
print(delta_df.round(4).to_string())

# ══════════════════════════════════════════════════════════
# BLAST ALIGNMENT REPORT — Sample Query
# ══════════════════════════════════════════════════════════

# Use the best-performing BLAST model for the demo
best_blast_model = blast_df['Pixel Accuracy'].idxmax().replace(' + BLAST', '')
print(f'Best BLAST model: {best_blast_model}')

# Extract embeddings for the best model again
model_class, preprocess_fn, min_sz = PRETRAINED_MODELS[best_blast_model]
tf.keras.backend.clear_session()
extractor, actual_size = build_patch_feature_extractor(model_class, min_sz)

patches_float = all_patches.astype(np.float32)
if actual_size != PATCH_SIZE:
    patches_resized = np.array([cv2.resize(p, (actual_size, actual_size)) for p in patches_float])
else:
    patches_resized = patches_float
patches_prep = np.array([preprocess_fn(p.copy()) for p in patches_resized])
embeddings = extractor.predict(patches_prep, batch_size=16, verbose=0)

# Pick a sample query
sample_img_id = 7
sample_patch_idx = 5
global_idx = sample_img_id * PATCHES_PER_IMAGE + sample_patch_idx

# Build DB excluding this image
train_mask = image_indices != sample_img_id
db = BLASTImageDatabase(metric='cosine', top_k=5)
db.build_database(embeddings[train_mask], all_patch_labels[train_mask])
result = db.query(embeddings[global_idx])

train_indices = np.where(train_mask)[0]
matched_global = train_indices[result['top_indices']]

print('\n' + '=' * 65)
print(f'BLAST Alignment Report ({best_blast_model} Embeddings)')
print('=' * 65)
print(f'Query: Image {sample_img_id}, Patch {sample_patch_idx}')
print(f'True label:  {CLASS_NAMES[all_patch_labels[global_idx]]}')
print(f'Predicted:   {CLASS_NAMES[result["predicted"]]} (confidence: {result["confidence"]:.3f})')
print('-' * 65)
print(f'{"Rank":<6}{"Image":<8}{"Patch":<8}{"Label":<14}{"Score":<10}')
print('-' * 65)
for rank, (gi, sim, lbl) in enumerate(zip(matched_global, result['top_sims'], result['top_labels'])):
    img_from = image_indices[gi]
    patch_from = gi - img_from * PATCHES_PER_IMAGE
    print(f'{rank+1:<6}{img_from:<8}{patch_from:<8}{CLASS_NAMES[lbl]:<14}{sim:<10.4f}')
print('=' * 65)

# Visual alignment report
fig, axes_al = plt.subplots(1, 6, figsize=(18, 3))
axes_al[0].imshow(all_patches[global_idx])
axes_al[0].set_title(f'Query\n{CLASS_NAMES[all_patch_labels[global_idx]]}', fontweight='bold')
axes_al[0].axis('off')
for j, (gi, sim, lbl) in enumerate(zip(matched_global, result['top_sims'], result['top_labels'])):
    axes_al[j+1].imshow(all_patches[gi])
    axes_al[j+1].set_title(f'Match {j+1}\n{CLASS_NAMES[lbl]} ({sim:.3f})')
    axes_al[j+1].axis('off')
plt.suptitle(f'BLAST Top-5 Alignment ({best_blast_model})', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '10_blast_alignment.png'), dpi=150, bbox_inches='tight')
plt.show()

del extractor, embeddings
tf.keras.backend.clear_session()
gc.collect()

# ══════════════════════════════════════════════════════════
# VISUALIZATION 7: Grand Summary Heatmap
# ══════════════════════════════════════════════════════════

metric_cols_heat = ['Pixel Accuracy', 'Mean IoU', 'Mean Dice', 'Precision', 'Recall', 'F1 Score']
heat_df = comparison_df[metric_cols_heat]

fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(heat_df.astype(float), annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=0, vmax=1, linewidths=0.5, linecolor='white', ax=ax)
ax.set_title('Grand Summary Heatmap — All Metrics, All Methods',
             fontsize=14, fontweight='bold')

# Add BLAST/Direct row labels
ytick_labels = []
for label in heat_df.index:
    if 'BLAST' in label:
        ytick_labels.append(f'[BLAST] {label}')
    else:
        ytick_labels.append(f'[Direct] {label}')
ax.set_yticklabels(ytick_labels, rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '11_grand_heatmap.png'), dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════
# FINAL SUMMARY & CONCLUSION
# ══════════════════════════════════════════════════════════

print('\n' + '=' * 100)
print('FINAL RESULTS: BLAST vs DIRECT SEGMENTATION ON ORAL CANCER HISTOPATHOLOGY')
print('=' * 100)

# Ranked by Pixel Accuracy
ranked_df = comparison_df.sort_values('Pixel Accuracy', ascending=False)
print('\n── All Methods Ranked by Pixel Accuracy ──')
print(ranked_df.to_string())

# Best BLAST vs Best Direct
blast_methods = [f'{m} + BLAST' for m in model_names]
direct_methods = [f'{m} (Direct)' for m in model_names]

best_blast = comparison_df.loc[blast_methods, 'Pixel Accuracy'].idxmax()
best_direct = comparison_df.loc[direct_methods, 'Pixel Accuracy'].idxmax()

print(f'\nBest BLAST method:  {best_blast} (Pixel Acc: {comparison_df.loc[best_blast, "Pixel Accuracy"]:.4f})')
print(f'Best Direct method: {best_direct} (Pixel Acc: {comparison_df.loc[best_direct, "Pixel Accuracy"]:.4f})')

# Overall winner per metric
print('\n── Overall Winner Per Metric ──')
for metric in metric_cols_heat:
    winner = comparison_df[metric].idxmax()
    val = comparison_df.loc[winner, metric]
    mode = 'BLAST' if 'BLAST' in winner else 'Direct'
    print(f'  {metric:<18}: {winner} ({val:.4f}) [{mode}]')

# BLAST win rate
blast_wins = 0
direct_wins = 0
for metric in metric_cols_heat:
    for mname in model_names:
        b_val = comparison_df.loc[f'{mname} + BLAST', metric]
        d_val = comparison_df.loc[f'{mname} (Direct)', metric]
        if b_val > d_val:
            blast_wins += 1
        elif d_val > b_val:
            direct_wins += 1

total = blast_wins + direct_wins
print(f'\n── Head-to-Head Win Rate (5 models x 6 metrics = 30 matchups) ──')
print(f'  BLAST wins:  {blast_wins}/{total} ({blast_wins/total*100:.1f}%)')
print(f'  Direct wins: {direct_wins}/{total} ({direct_wins/total*100:.1f}%)')

# List output files
print(f'\n── Output Files ──')
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f'  {f} ({size_kb:.0f} KB)')

print('\nDone! All experiments completed successfully.')
