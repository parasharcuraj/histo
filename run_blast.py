import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── GPU Setup ──
nvidia_base = "/usr/local/lib/python3.10/dist-packages/nvidia"
nvidia_libs = [
    f"{nvidia_base}/cuda_runtime/lib",
    f"{nvidia_base}/cudnn/lib",
    f"{nvidia_base}/cublas/lib",
    f"{nvidia_base}/cufft/lib",
    f"{nvidia_base}/curand/lib",
    f"{nvidia_base}/cusolver/lib",
    f"{nvidia_base}/cusparse/lib",
    f"{nvidia_base}/nvjitlink/lib",
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
from collections import Counter

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, jaccard_score
)
from sklearn.preprocessing import normalize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print(f'NumPy: {np.__version__}')
print(f'TensorFlow: {tf.__version__}')
print(f'OpenCV: {cv2.__version__}')
print(f'GPU devices: {gpus}')


# ── Configuration ──
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
CLASS_COLORS = {0: [144, 238, 144], 1: [255, 255, 102], 2: [255, 99, 71]}  # green, yellow, red

# LBP params
LBP_RADIUS = 3
LBP_POINTS = 24
MLBP_RADII = [1, 2, 3, 5]

# GLCM params
GLCM_DISTANCES = [1, 3, 5]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_PROPERTIES = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'energy']

# BLAST params
TOP_K = 5

# Paths
DATA_DIR = 'data/synthetic_images'
OUTPUT_DIR = 'outputs'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f'Image size: {IMG_SIZE}x{IMG_SIZE}, Patch size: {PATCH_SIZE}x{PATCH_SIZE}')
print(f'Patches per image: {PATCHES_PER_IMAGE}, Total images: {NUM_IMAGES}')

def draw_nuclei(img, mask, region_mask, cls, rng):
    """Draw nuclei into a region based on tissue class."""
    ys, xs = np.where(region_mask)
    if len(ys) == 0:
        return
    
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    area = len(ys)
    
    if cls == 0:  # Normal: few, small, uniform, organized
        n_nuclei = max(3, area // 900)
        radius_range = (2, 4)
        color_base = np.array([80, 40, 120])  # lighter purple
    elif cls == 1:  # Dysplasia: medium density, varied sizes
        n_nuclei = max(5, area // 500)
        radius_range = (3, 6)
        color_base = np.array([60, 20, 100])  # darker purple
    else:  # Carcinoma: dense, large, irregular, dark
        n_nuclei = max(8, area // 250)
        radius_range = (3, 8)
        color_base = np.array([40, 10, 80])  # darkest
    
    for _ in range(n_nuclei):
        idx = rng.randint(0, len(ys))
        cy, cx = ys[idx], xs[idx]
        r = rng.randint(radius_range[0], radius_range[1] + 1)
        color_var = rng.randint(-15, 16, size=3)
        color = np.clip(color_base + color_var, 0, 255).tolist()
        cv2.circle(img, (cx, cy), r, color, -1)
        if cls == 2 and rng.random() > 0.5:  # irregular shapes for carcinoma
            offset = rng.randint(-3, 4, size=2)
            cv2.circle(img, (cx + offset[0], cy + offset[1]), r - 1, color, -1)


def generate_tissue_image(img_id, rng):
    """Generate one synthetic H&E image with a segmentation mask containing mixed regions."""
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    
    # Background: pinkish H&E stain
    bg_colors = {
        0: np.array([230, 200, 210]),  # light pink (normal stroma)
        1: np.array([215, 180, 200]),  # slightly darker
        2: np.array([200, 160, 185])   # darker pink (dense tissue)
    }
    
    # Create 2-4 random regions per image using a simple Voronoi-like partition
    n_regions = rng.randint(2, 5)
    centers = rng.randint(30, IMG_SIZE - 30, size=(n_regions, 2))
    
    # Assign classes to regions based on image_id patterns
    if img_id < 3:  # mostly normal with some dysplasia
        region_classes = rng.choice([0, 0, 1], size=n_regions)
        region_classes[0] = 0  # ensure at least one normal
    elif img_id < 6:  # mixed dysplasia and carcinoma
        region_classes = rng.choice([0, 1, 2], size=n_regions)
        region_classes[0] = 1  # ensure at least one dysplasia
    else:  # mostly carcinoma
        region_classes = rng.choice([1, 2, 2], size=n_regions)
        region_classes[0] = 2  # ensure at least one carcinoma
    
    # Build mask via nearest-center assignment
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    dist_maps = np.stack([
        (yy - cy)**2 + (xx - cx)**2 for cy, cx in centers
    ], axis=-1)
    nearest = np.argmin(dist_maps, axis=-1)
    
    for r_idx in range(n_regions):
        region_mask = nearest == r_idx
        cls = region_classes[r_idx]
        mask[region_mask] = cls
        
        # Fill background color with some noise
        bg = bg_colors[cls].astype(np.float64)
        noise = rng.normal(0, 5, size=(IMG_SIZE, IMG_SIZE, 3))
        for c in range(3):
            img[:, :, c][region_mask] = np.clip(bg[c] + noise[:, :, c][region_mask], 0, 255).astype(np.uint8)
        
        # Draw nuclei
        draw_nuclei(img, mask, region_mask, cls, rng)
    
    # Light Gaussian blur for realism
    img = cv2.GaussianBlur(img, (3, 3), 0.7)
    return img, mask


# Generate all images and masks
images = []
masks = []
labels_records = []

for i in range(NUM_IMAGES):
    rng = np.random.RandomState(SEED + i)
    img, msk = generate_tissue_image(i, rng)
    images.append(img)
    masks.append(msk)
    
    # Save image and mask
    cv2.imwrite(os.path.join(DATA_DIR, f'image_{i:02d}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(DATA_DIR, f'mask_{i:02d}.png'), msk)
    
    # Compute dominant class for metadata
    unique, counts = np.unique(msk, return_counts=True)
    dominant = CLASS_NAMES[unique[np.argmax(counts)]]
    labels_records.append({'image_id': i, 'filename': f'image_{i:02d}.png',
                           'mask_file': f'mask_{i:02d}.png', 'dominant_class': dominant,
                           'normal_pct': round((msk == 0).mean() * 100, 1),
                           'dysplasia_pct': round((msk == 1).mean() * 100, 1),
                           'carcinoma_pct': round((msk == 2).mean() * 100, 1)})

labels_df = pd.DataFrame(labels_records)
labels_df.to_csv('data/labels.csv', index=False)
print('Generated', NUM_IMAGES, 'images + masks')
print(labels_df.to_string())

# Visualize all images with their ground truth segmentation masks
fig, axes = plt.subplots(NUM_IMAGES, 3, figsize=(12, 4 * NUM_IMAGES))

for i in range(NUM_IMAGES):
    axes[i, 0].imshow(images[i])
    axes[i, 0].set_title(f'Image {i}')
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
plt.savefig(os.path.join(OUTPUT_DIR, 'all_images_masks.png'), dpi=120, bbox_inches='tight')
plt.show()

def extract_patches(image, mask):
    """Extract non-overlapping patches and their majority-vote labels."""
    patches = []
    patch_labels = []
    patch_masks = []
    for r in range(PATCHES_PER_SIDE):
        for c in range(PATCHES_PER_SIDE):
            y0, y1 = r * PATCH_SIZE, (r + 1) * PATCH_SIZE
            x0, x1 = c * PATCH_SIZE, (c + 1) * PATCH_SIZE
            patch = image[y0:y1, x0:x1]
            patch_mask = mask[y0:y1, x0:x1]
            # Majority vote label
            vals, cnts = np.unique(patch_mask, return_counts=True)
            label = vals[np.argmax(cnts)]
            patches.append(patch)
            patch_labels.append(label)
            patch_masks.append(patch_mask)
    return patches, patch_labels, patch_masks


# Extract all patches
all_patches = []      # list of (img_idx, patch_idx, patch_array)
all_patch_labels = []  # majority-vote label
all_patch_masks = []   # full patch mask
image_indices = []     # which image each patch came from

for i in range(NUM_IMAGES):
    patches, plabels, pmasks = extract_patches(images[i], masks[i])
    for j, (p, l, m) in enumerate(zip(patches, plabels, pmasks)):
        all_patches.append(p)
        all_patch_labels.append(l)
        all_patch_masks.append(m)
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
    samples = np.random.choice(cls_indices, size=min(6, len(cls_indices)), replace=False)
    for j, idx in enumerate(samples):
        axes[cls_id, j].imshow(all_patches[idx])
        axes[cls_id, j].set_title(f'Img {image_indices[idx]}')
        axes[cls_id, j].axis('off')
    axes[cls_id, 0].set_ylabel(CLASS_NAMES[cls_id], fontsize=14, rotation=0, labelpad=60)

plt.suptitle('Sample Patches per Class', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sample_patches.png'), dpi=120, bbox_inches='tight')
plt.show()

# ── LBP Feature Extraction ──
def extract_lbp(patch, radius=LBP_RADIUS, n_points=LBP_POINTS):
    gray = rgb2gray(patch)
    gray_uint8 = (gray * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray_uint8, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
    return hist


# ── MLBP Feature Extraction ──
def extract_mlbp(patch):
    features = []
    gray = rgb2gray(patch)
    gray_uint8 = (gray * 255).astype(np.uint8)
    for radius in MLBP_RADII:
        n_points = 8 * radius
        lbp = local_binary_pattern(gray_uint8, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
        features.extend(hist)
    return np.array(features)


# ── GLCM Feature Extraction ──
def extract_glcm(patch):
    gray = rgb2gray(patch)
    gray_uint8 = (gray * 255).astype(np.uint8)
    features = []
    for d in GLCM_DISTANCES:
        glcm = graycomatrix(gray_uint8, distances=[d], angles=GLCM_ANGLES,
                            levels=256, symmetric=True, normed=True)
        for prop in GLCM_PROPERTIES:
            vals = graycoprops(glcm, prop).flatten()
            features.extend(vals)
    return np.array(features)


print('Feature extraction functions defined.')
# Quick dimension check
sample = all_patches[0]
print(f'LBP dims:  {extract_lbp(sample).shape[0]}')
print(f'MLBP dims: {extract_mlbp(sample).shape[0]}')
print(f'GLCM dims: {extract_glcm(sample).shape[0]}')

# ── CNN Feature Extractor ──
def build_cnn_model(input_shape=(PATCH_SIZE, PATCH_SIZE, 3), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    embedding = layers.Dense(64, activation='relu', name='embedding')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(embedding)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Also create an embedding extractor
    embed_model = Model(inputs, embedding)
    return model, embed_model


model, embed_model = build_cnn_model()
model.summary()

# ── Extract LBP, MLBP, GLCM features for all patches ──
print('Extracting LBP features...')
lbp_features = np.array([extract_lbp(p) for p in all_patches])
print(f'  Shape: {lbp_features.shape}')

print('Extracting MLBP features...')
mlbp_features = np.array([extract_mlbp(p) for p in all_patches])
print(f'  Shape: {mlbp_features.shape}')

print('Extracting GLCM features...')
glcm_features = np.array([extract_glcm(p) for p in all_patches])
print(f'  Shape: {glcm_features.shape}')

print('Done.')

# Visualize LBP images for one sample per class
fig, axes = plt.subplots(3, 3, figsize=(12, 10))

for cls_id in range(NUM_CLASSES):
    idx = np.where(all_patch_labels == cls_id)[0][0]
    patch = all_patches[idx]
    gray = (rgb2gray(patch) * 255).astype(np.uint8)
    lbp_img = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method='uniform')
    
    axes[cls_id, 0].imshow(patch)
    axes[cls_id, 0].set_title(f'{CLASS_NAMES[cls_id]} - Original')
    axes[cls_id, 0].axis('off')
    
    axes[cls_id, 1].imshow(lbp_img, cmap='hot')
    axes[cls_id, 1].set_title(f'{CLASS_NAMES[cls_id]} - LBP')
    axes[cls_id, 1].axis('off')
    
    axes[cls_id, 2].bar(range(len(lbp_features[idx])), lbp_features[idx], color='steelblue')
    axes[cls_id, 2].set_title(f'{CLASS_NAMES[cls_id]} - LBP Histogram')
    axes[cls_id, 2].set_xlabel('Bin')

plt.suptitle('LBP Visualization per Class', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'lbp_visualization.png'), dpi=120, bbox_inches='tight')
plt.show()

# Visualize GLCM feature distributions per class
glcm_df = pd.DataFrame(glcm_features[:, :5], columns=GLCM_PROPERTIES)
glcm_df['class'] = [CLASS_NAMES[l] for l in all_patch_labels]

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, prop in enumerate(GLCM_PROPERTIES):
    sns.boxplot(data=glcm_df, x='class', y=prop, ax=axes[i], palette='Set2')
    axes[i].set_title(prop.capitalize())
    axes[i].set_xlabel('')

plt.suptitle('GLCM Feature Distributions by Class (distance=1)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'glcm_features.png'), dpi=120, bbox_inches='tight')
plt.show()

class BLASTImageDatabase:
    """BLAST-like database for histopathological patch matching and segmentation."""
    
    def __init__(self, metric='cosine', top_k=TOP_K):
        self.metric = metric
        self.top_k = top_k
        self.db_features = None
        self.db_labels = None
    
    def build_database(self, features, labels):
        """Analogous to makeblastdb — index reference patches."""
        self.db_features = normalize(features.copy())
        self.db_labels = labels.copy()
    
    def _compute_similarity(self, query_feat, db_feats):
        query_norm = normalize(query_feat.reshape(1, -1))
        if self.metric == 'cosine':
            sims = (query_norm @ db_feats.T).flatten()
        else:  # euclidean — convert distance to similarity
            dists = np.linalg.norm(db_feats - query_norm, axis=1)
            sims = 1.0 / (1.0 + dists)
        return sims
    
    def query(self, query_feature):
        """Analogous to blastn — find top-K matches and predict label."""
        sims = self._compute_similarity(query_feature, self.db_features)
        top_indices = np.argsort(sims)[::-1][:self.top_k]
        top_sims = sims[top_indices]
        top_labels = self.db_labels[top_indices]
        
        # Weighted voting
        weights = np.maximum(top_sims, 1e-10)
        class_scores = np.zeros(NUM_CLASSES)
        for lbl, w in zip(top_labels, weights):
            class_scores[lbl] += w
        
        predicted = np.argmax(class_scores)
        confidence = class_scores[predicted] / class_scores.sum()
        
        return {
            'predicted': predicted,
            'confidence': confidence,
            'top_indices': top_indices,
            'top_sims': top_sims,
            'top_labels': top_labels,
            'class_scores': class_scores
        }
    
    def segment_image(self, patch_features):
        """Query all patches of one image and assemble a segmentation map."""
        pred_labels = []
        confidences = []
        for feat in patch_features:
            result = self.query(feat)
            pred_labels.append(result['predicted'])
            confidences.append(result['confidence'])
        
        # Reshape to spatial grid
        pred_grid = np.array(pred_labels).reshape(PATCHES_PER_SIDE, PATCHES_PER_SIDE)
        conf_grid = np.array(confidences).reshape(PATCHES_PER_SIDE, PATCHES_PER_SIDE)
        
        # Upscale to full image size
        seg_map = np.kron(pred_grid, np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8))
        
        return seg_map, pred_grid, conf_grid


print('BLASTImageDatabase class defined.')

def leave_one_image_out_blast(features, labels, img_indices, metric='cosine', top_k=TOP_K):
    """Leave-One-Image-Out CV for BLAST-like segmentation."""
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
        
        # Store predictions
        pred_flat = pred_grid.flatten()
        all_pred_labels[test_mask] = pred_flat
        all_seg_maps[img_id] = seg_map
        all_conf_maps[img_id] = conf_grid
    
    return all_pred_labels, all_seg_maps, all_conf_maps


print('LOIO-CV function defined.')


def compute_segmentation_metrics(true_masks, pred_seg_maps, method_name):
    """Compute pixel-level segmentation metrics."""
    all_true = []
    all_pred = []

    for img_id in range(NUM_IMAGES):
        gt = true_masks[img_id].flatten()
        pred = pred_seg_maps[img_id].flatten()
        all_true.append(gt)
        all_pred.append(pred)

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    pixel_acc = accuracy_score(all_true, all_pred)
    mean_iou = jaccard_score(all_true, all_pred, average='macro')

    # Dice per class
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


# Run BLAST-like segmentation for LBP, MLBP, GLCM
results = {}
blast_interim_metrics = []  # collect for running comparison

for name, feats in [('LBP', lbp_features), ('MLBP', mlbp_features), ('GLCM', glcm_features)]:
    print(f'Running BLAST segmentation with {name}...')
    preds, seg_maps, conf_maps = leave_one_image_out_blast(
        feats, all_patch_labels, image_indices, metric='cosine')
    results[name] = {'preds': preds, 'seg_maps': seg_maps, 'conf_maps': conf_maps}
    m = compute_segmentation_metrics(masks, seg_maps, f'BLAST-{name}')
    blast_interim_metrics.append({k: v for k, v in m.items() if k not in ['Dice_per_class', 'all_true', 'all_pred']})
    print(f'  ✓ BLAST-{name} complete:')
    print(f'    Pixel Accuracy: {m["Pixel Accuracy"]:.4f}  |  Mean IoU: {m["Mean IoU"]:.4f}  |  Mean Dice: {m["Mean Dice"]:.4f}  |  F1: {m["F1 Score"]:.4f}')
    print()

print('Done with handcrafted features.')
print(pd.DataFrame(blast_interim_metrics).set_index('Method').round(4).to_string())


# CNN: Leave-One-Image-Out with retraining per fold
import gc
print('Running BLAST segmentation with CNN (retraining per fold)...')

cnn_preds = np.zeros(len(all_patch_labels), dtype=int)
cnn_seg_maps = {}
cnn_conf_maps = {}

patches_normalized = all_patches.astype(np.float32) / 255.0

for img_id in range(NUM_IMAGES):
    test_mask = image_indices == img_id
    train_mask = ~test_mask
    
    X_train = patches_normalized[train_mask]
    y_train = all_patch_labels[train_mask]
    X_test = patches_normalized[test_mask]
    
    tf.keras.backend.clear_session()
    gc.collect()
    fold_model, fold_embed = build_cnn_model()
    fold_model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
    
    train_embeddings = fold_embed.predict(X_train, verbose=0)
    test_embeddings = fold_embed.predict(X_test, verbose=0)
    
    db = BLASTImageDatabase(metric='cosine', top_k=TOP_K)
    db.build_database(train_embeddings, all_patch_labels[train_mask])
    seg_map, pred_grid, conf_grid = db.segment_image(test_embeddings)
    
    cnn_preds[test_mask] = pred_grid.flatten()
    cnn_seg_maps[img_id] = seg_map
    cnn_conf_maps[img_id] = conf_grid
    
    del fold_model, fold_embed, train_embeddings, test_embeddings
    tf.keras.backend.clear_session()
    gc.collect()
    print(f'  Fold {img_id}: done')

results['CNN'] = {'preds': cnn_preds, 'seg_maps': cnn_seg_maps, 'conf_maps': cnn_conf_maps}
m = compute_segmentation_metrics(masks, cnn_seg_maps, 'BLAST-CNN')
print(f'\n  ✓ BLAST-CNN complete:')
print(f'    Pixel Accuracy: {m["Pixel Accuracy"]:.4f}  |  Mean IoU: {m["Mean IoU"]:.4f}  |  Mean Dice: {m["Mean Dice"]:.4f}  |  F1: {m["F1 Score"]:.4f}')

# Show running comparison so far
blast_interim_metrics.append({k: v for k, v in m.items() if k not in ['Dice_per_class', 'all_true', 'all_pred']})
print('\n── BLAST Methods Summary ──')
print(pd.DataFrame(blast_interim_metrics).set_index('Method').round(4).to_string())


# Visualize segmentation results for 3 sample images
sample_imgs = [0, 4, 8]
methods = ['LBP', 'MLBP', 'GLCM', 'CNN']

fig, axes = plt.subplots(len(sample_imgs), len(methods) + 2, figsize=(22, 4 * len(sample_imgs)))

for row, img_id in enumerate(sample_imgs):
    # Original image
    axes[row, 0].imshow(images[img_id])
    axes[row, 0].set_title(f'Image {img_id}')
    axes[row, 0].axis('off')
    
    # Ground truth
    gt_color = np.zeros((*masks[img_id].shape, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS.items():
        gt_color[masks[img_id] == cls_id] = color
    axes[row, 1].imshow(gt_color)
    axes[row, 1].set_title('Ground Truth')
    axes[row, 1].axis('off')
    
    # Each method's segmentation
    for col, method in enumerate(methods):
        seg = results[method]['seg_maps'][img_id]
        seg_color = np.zeros((*seg.shape, 3), dtype=np.uint8)
        for cls_id, color in CLASS_COLORS.items():
            seg_color[seg == cls_id] = color
        axes[row, col + 2].imshow(seg_color)
        axes[row, col + 2].set_title(f'{method}')
        axes[row, col + 2].axis('off')

legend_patches = [mpatches.Patch(color=np.array(c)/255, label=n) for n, c in zip(CLASS_NAMES, CLASS_COLORS.values())]
fig.legend(handles=legend_patches, loc='upper center', ncol=3, fontsize=12)
plt.suptitle('BLAST-like Segmentation Results', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'blast_segmentation_results.png'), dpi=120, bbox_inches='tight')
plt.show()

# BLAST-like alignment report for a sample query patch
sample_img_id = 5
sample_patch_idx = 7  # patch within the image
global_idx = sample_img_id * PATCHES_PER_IMAGE + sample_patch_idx

# Build DB excluding this image
train_mask = image_indices != sample_img_id
db = BLASTImageDatabase(metric='cosine', top_k=5)
db.build_database(lbp_features[train_mask], all_patch_labels[train_mask])
result = db.query(lbp_features[global_idx])

# Get actual indices in the full array for the top matches
train_indices = np.where(train_mask)[0]
matched_global = train_indices[result['top_indices']]

print('═' * 60)
print('BLAST-like Alignment Report (LBP Scoring Matrix)')
print('═' * 60)
print(f'Query: Image {sample_img_id}, Patch {sample_patch_idx}')
print(f'True label: {CLASS_NAMES[all_patch_labels[global_idx]]}')
print(f'Predicted:  {CLASS_NAMES[result["predicted"]]} (confidence: {result["confidence"]:.3f})')
print('─' * 60)
print(f'{"Rank":<6}{"Image":<8}{"Patch":<8}{"Label":<14}{"Score":<10}')
print('─' * 60)
for rank, (gi, sim, lbl) in enumerate(zip(matched_global, result['top_sims'], result['top_labels'])):
    img_from = image_indices[gi]
    patch_from = gi - img_from * PATCHES_PER_IMAGE
    print(f'{rank+1:<6}{img_from:<8}{patch_from:<8}{CLASS_NAMES[lbl]:<14}{sim:<10.4f}')
print('═' * 60)

# Visual
fig, axes = plt.subplots(1, 6, figsize=(18, 3))
axes[0].imshow(all_patches[global_idx])
axes[0].set_title(f'Query\n{CLASS_NAMES[all_patch_labels[global_idx]]}', fontweight='bold')
axes[0].axis('off')
for j, (gi, sim, lbl) in enumerate(zip(matched_global, result['top_sims'], result['top_labels'])):
    axes[j+1].imshow(all_patches[gi])
    axes[j+1].set_title(f'Match {j+1}\n{CLASS_NAMES[lbl]} ({sim:.3f})')
    axes[j+1].axis('off')
plt.suptitle('BLAST-like Top-5 Alignment', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'blast_alignment_report.png'), dpi=120, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════
# Model 1: Vanilla U-Net
# ══════════════════════════════════════════════════════════
def build_vanilla_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)
    # Encoder
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D(2)(c1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D(2)(c2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    p3 = layers.MaxPooling2D(2)(c3)
    # Bottleneck
    b = layers.Conv2D(128, 3, activation='relu', padding='same')(p3)
    # Decoder
    u3 = layers.Concatenate()([layers.UpSampling2D(2)(b), c3])
    d3 = layers.Conv2D(64, 3, activation='relu', padding='same')(u3)
    u2 = layers.Concatenate()([layers.UpSampling2D(2)(d3), c2])
    d2 = layers.Conv2D(32, 3, activation='relu', padding='same')(u2)
    u1 = layers.Concatenate()([layers.UpSampling2D(2)(d2), c1])
    d1 = layers.Conv2D(16, 3, activation='relu', padding='same')(u1)
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(d1)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ══════════════════════════════════════════════════════════
# Model 2: Attention U-Net
# ══════════════════════════════════════════════════════════
def attention_gate(x, g, inter_channels):
    """Attention gate: x=skip, g=gating signal from decoder."""
    theta_x = layers.Conv2D(inter_channels, 1, padding='same')(x)
    phi_g = layers.Conv2D(inter_channels, 1, padding='same')(g)
    add_xg = layers.Add()([theta_x, phi_g])
    act = layers.Activation('relu')(add_xg)
    psi = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(act)
    return layers.Multiply()([x, psi])

def build_attention_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D(2)(c1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D(2)(c2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    p3 = layers.MaxPooling2D(2)(c3)
    b = layers.Conv2D(128, 3, activation='relu', padding='same')(p3)
    # Decoder with attention gates
    g3 = layers.UpSampling2D(2)(b)
    a3 = attention_gate(c3, g3, 32)
    u3 = layers.Concatenate()([g3, a3])
    d3 = layers.Conv2D(64, 3, activation='relu', padding='same')(u3)
    g2 = layers.UpSampling2D(2)(d3)
    a2 = attention_gate(c2, g2, 16)
    u2 = layers.Concatenate()([g2, a2])
    d2 = layers.Conv2D(32, 3, activation='relu', padding='same')(u2)
    g1 = layers.UpSampling2D(2)(d2)
    a1 = attention_gate(c1, g1, 8)
    u1 = layers.Concatenate()([g1, a1])
    d1 = layers.Conv2D(16, 3, activation='relu', padding='same')(u1)
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(d1)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ══════════════════════════════════════════════════════════
# Model 3: ResU-Net (Residual U-Net)
# ══════════════════════════════════════════════════════════
def res_block(x, filters):
    """Residual block with skip-add connection."""
    shortcut = layers.Conv2D(filters, 1, padding='same')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

def build_resunet(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)
    c1 = res_block(inputs, 16)
    p1 = layers.MaxPooling2D(2)(c1)
    c2 = res_block(p1, 32)
    p2 = layers.MaxPooling2D(2)(c2)
    c3 = res_block(p2, 64)
    p3 = layers.MaxPooling2D(2)(c3)
    b = res_block(p3, 128)
    u3 = layers.Concatenate()([layers.UpSampling2D(2)(b), c3])
    d3 = res_block(u3, 64)
    u2 = layers.Concatenate()([layers.UpSampling2D(2)(d3), c2])
    d2 = res_block(u2, 32)
    u1 = layers.Concatenate()([layers.UpSampling2D(2)(d2), c1])
    d1 = res_block(u1, 16)
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(d1)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ══════════════════════════════════════════════════════════
# Model 4: Dense U-Net (DenseNet-style encoder blocks)
# ══════════════════════════════════════════════════════════
def dense_block(x, filters, n_layers=2):
    """Dense block: each layer concatenates all previous outputs."""
    concat_list = [x]
    for _ in range(n_layers):
        out = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        concat_list.append(out)
        x = layers.Concatenate()(concat_list)
    return x

def build_dense_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)
    c1 = dense_block(inputs, 12)
    c1_red = layers.Conv2D(16, 1, activation='relu')(c1)  # transition
    p1 = layers.MaxPooling2D(2)(c1_red)
    c2 = dense_block(p1, 16)
    c2_red = layers.Conv2D(32, 1, activation='relu')(c2)
    p2 = layers.MaxPooling2D(2)(c2_red)
    c3 = dense_block(p2, 24)
    c3_red = layers.Conv2D(64, 1, activation='relu')(c3)
    p3 = layers.MaxPooling2D(2)(c3_red)
    b = layers.Conv2D(128, 3, activation='relu', padding='same')(p3)
    u3 = layers.Concatenate()([layers.UpSampling2D(2)(b), c3_red])
    d3 = layers.Conv2D(64, 3, activation='relu', padding='same')(u3)
    u2 = layers.Concatenate()([layers.UpSampling2D(2)(d3), c2_red])
    d2 = layers.Conv2D(32, 3, activation='relu', padding='same')(u2)
    u1 = layers.Concatenate()([layers.UpSampling2D(2)(d2), c1_red])
    d1 = layers.Conv2D(16, 3, activation='relu', padding='same')(u1)
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(d1)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ══════════════════════════════════════════════════════════
# Model 5: U-Net++ (Nested U-Net / Dense Skip Pathways)
# ══════════════════════════════════════════════════════════
def build_unetpp(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """Simplified U-Net++ with nested skip connections."""
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    x00 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    p0 = layers.MaxPooling2D(2)(x00)
    x10 = layers.Conv2D(32, 3, activation='relu', padding='same')(p0)
    p1 = layers.MaxPooling2D(2)(x10)
    x20 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D(2)(x20)
    x30 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)  # bottleneck
    
    # Nested skip pathways
    # Level 0→1 intermediate node
    x01 = layers.Conv2D(16, 3, activation='relu', padding='same')(
        layers.Concatenate()([x00, layers.UpSampling2D(2)(x10)]))
    # Level 1→2 intermediate node
    x11 = layers.Conv2D(32, 3, activation='relu', padding='same')(
        layers.Concatenate()([x10, layers.UpSampling2D(2)(x20)]))
    # Level 2→3 decode
    x21 = layers.Conv2D(64, 3, activation='relu', padding='same')(
        layers.Concatenate()([x20, layers.UpSampling2D(2)(x30)]))
    # Level 1→2 with nested
    x12 = layers.Conv2D(32, 3, activation='relu', padding='same')(
        layers.Concatenate()([x10, x11, layers.UpSampling2D(2)(x21)]))
    # Level 0→1 with nested
    x02 = layers.Conv2D(16, 3, activation='relu', padding='same')(
        layers.Concatenate()([x00, x01, layers.UpSampling2D(2)(x12)]))
    
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(x02)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ── Print parameter counts ──
unet_builders = {
    'Vanilla U-Net': build_vanilla_unet,
    'Attention U-Net': build_attention_unet,
    'ResU-Net': build_resunet,
    'Dense U-Net': build_dense_unet,
    'U-Net++': build_unetpp,
}

print(f'{"Model":<20} {"Parameters":>12}')
print('─' * 34)
for name, builder in unet_builders.items():
    m = builder()
    print(f'{name:<20} {m.count_params():>12,}')
    del m

# ── Train all 5 U-Net variants with Leave-One-Image-Out CV ──
import time

images_np = np.array(images, dtype=np.float32) / 255.0
masks_np = np.array(masks)  # (N, H, W)

UNET_EPOCHS = 10

unet_all_seg_maps = {}   # {model_name: {img_id: seg_map}}
unet_all_metrics_raw = {} # store pixel preds/trues
unet_interim_metrics = []  # collect for running comparison

for model_name, builder_fn in unet_builders.items():
    print(f'\n{"═"*50}')
    print(f'Training {model_name} (LOIO-CV, {UNET_EPOCHS} epochs)...')
    print(f'{"═"*50}')
    
    seg_maps = {}
    pixel_preds = []
    pixel_trues = []
    t0 = time.time()
    
    for img_id in range(NUM_IMAGES):
        train_idx = [i for i in range(NUM_IMAGES) if i != img_id]
        X_train = images_np[train_idx]
        y_train = masks_np[train_idx][..., np.newaxis]
        X_test = images_np[img_id:img_id+1]
        
        fold_model = builder_fn()
        fold_model.fit(X_train, y_train, epochs=UNET_EPOCHS, batch_size=4, verbose=0)
        
        pred_prob = fold_model.predict(X_test, verbose=0)[0]
        pred_seg = np.argmax(pred_prob, axis=-1).astype(np.uint8)
        
        seg_maps[img_id] = pred_seg
        pixel_preds.append(pred_seg.flatten())
        pixel_trues.append(masks_np[img_id].flatten())
        
        del fold_model
        tf.keras.backend.clear_session()
        print(f'  Fold {img_id}: done')
    
    elapsed = time.time() - t0
    unet_all_seg_maps[model_name] = seg_maps
    unet_all_metrics_raw[model_name] = {
        'preds': np.concatenate(pixel_preds),
        'trues': np.concatenate(pixel_trues)
    }
    
    m = compute_segmentation_metrics(masks, seg_maps, model_name)
    unet_interim_metrics.append({k: v for k, v in m.items() if k not in ['Dice_per_class', 'all_true', 'all_pred']})
    print(f'\n  ✓ {model_name} complete ({elapsed:.0f}s):')
    print(f'    Pixel Accuracy: {m["Pixel Accuracy"]:.4f}  |  Mean IoU: {m["Mean IoU"]:.4f}  |  Mean Dice: {m["Mean Dice"]:.4f}  |  F1: {m["F1 Score"]:.4f}')

print('\n── U-Net Variants Summary ──')
print(pd.DataFrame(unet_interim_metrics).set_index('Method').round(4).to_string())


# Visualize all 5 U-Net variants vs BLAST-CNN for sample images
sample_imgs = [0, 4, 8]
unet_names = list(unet_builders.keys())
n_cols = 2 + 1 + len(unet_names)  # original, GT, BLAST-CNN, 5 U-Nets

fig, axes = plt.subplots(len(sample_imgs), n_cols, figsize=(4 * n_cols, 4 * len(sample_imgs)))

for row, img_id in enumerate(sample_imgs):
    # Original
    axes[row, 0].imshow(images[img_id])
    axes[row, 0].set_title('Original' if row == 0 else '')
    axes[row, 0].axis('off')
    
    # Ground truth
    gt_color = np.zeros((*masks[img_id].shape, 3), dtype=np.uint8)
    for c, clr in CLASS_COLORS.items():
        gt_color[masks[img_id] == c] = clr
    axes[row, 1].imshow(gt_color)
    axes[row, 1].set_title('Ground Truth' if row == 0 else '')
    axes[row, 1].axis('off')
    
    # BLAST-CNN
    blast_seg = results['CNN']['seg_maps'][img_id]
    blast_color = np.zeros((*blast_seg.shape, 3), dtype=np.uint8)
    for c, clr in CLASS_COLORS.items():
        blast_color[blast_seg == c] = clr
    axes[row, 2].imshow(blast_color)
    axes[row, 2].set_title('BLAST-CNN' if row == 0 else '')
    axes[row, 2].axis('off')
    
    # 5 U-Net variants
    for j, uname in enumerate(unet_names):
        seg = unet_all_seg_maps[uname][img_id]
        seg_color = np.zeros((*seg.shape, 3), dtype=np.uint8)
        for c, clr in CLASS_COLORS.items():
            seg_color[seg == c] = clr
        axes[row, 3 + j].imshow(seg_color)
        axes[row, 3 + j].set_title(uname if row == 0 else '')
        axes[row, 3 + j].axis('off')

legend_patches = [mpatches.Patch(color=np.array(c)/255, label=n) for n, c in zip(CLASS_NAMES, CLASS_COLORS.values())]
fig.legend(handles=legend_patches, loc='upper center', ncol=3, fontsize=12)
plt.suptitle('BLAST-CNN vs 5 U-Net Variants — Segmentation Comparison', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'blast_vs_5unet_segmentation.png'), dpi=120, bbox_inches='tight')
plt.show()

# ── Compute metrics for all methods ──
metrics_list = []

# 4 BLAST methods
for method in ['LBP', 'MLBP', 'GLCM', 'CNN']:
    m = compute_segmentation_metrics(masks, results[method]['seg_maps'], f'BLAST-{method}')
    metrics_list.append(m)

# 5 U-Net variants
for unet_name in unet_builders.keys():
    m = compute_segmentation_metrics(masks, unet_all_seg_maps[unet_name], unet_name)
    metrics_list.append(m)

# Summary table
metrics_df = pd.DataFrame([{k: v for k, v in m.items() if k not in ['Dice_per_class', 'all_true', 'all_pred']}
                           for m in metrics_list])
metrics_df = metrics_df.set_index('Method')
metrics_df = metrics_df.round(4)
print('\n' + '=' * 80)
print('FULL SEGMENTATION METRICS — BLAST (4) vs U-Net Variants (5)')
print('=' * 80)
print(metrics_df.to_string())

# Confusion matrices — 9 methods (2 rows)
n_methods = len(metrics_list)
fig, axes = plt.subplots(2, 5, figsize=(28, 10))
axes_flat = axes.flatten()

for i, m in enumerate(metrics_list):
    cm = confusion_matrix(m['all_true'], m['all_pred'], labels=[0, 1, 2])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes_flat[i], vmin=0, vmax=1)
    axes_flat[i].set_title(m['Method'], fontsize=11)
    axes_flat[i].set_ylabel('True' if i % 5 == 0 else '')
    axes_flat[i].set_xlabel('Predicted')

# Hide unused subplot
if n_methods < 10:
    axes_flat[9].axis('off')

plt.suptitle('Normalized Confusion Matrices (Pixel-Level) — All 9 Methods', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices_all.png'), dpi=120, bbox_inches='tight')
plt.show()

# Grouped bar chart — all 9 methods
metric_cols = ['Pixel Accuracy', 'Mean IoU', 'Mean Dice', 'Precision', 'Recall', 'F1 Score']
x = np.arange(len(metric_cols))
method_names = metrics_df.index.tolist()
n = len(method_names)
width = 0.8 / n
colors_9 = ['#4C72B0', '#55A868', '#C44E52', '#8172B2',   # BLAST methods
             '#CCB974', '#64B5CD', '#E377C2', '#FF7F0E', '#2CA02C']  # U-Net variants

fig, ax = plt.subplots(figsize=(18, 7))
for i, method in enumerate(method_names):
    vals = metrics_df.loc[method, metric_cols].values.astype(float)
    ax.bar(x + i * width, vals, width, label=method, color=colors_9[i])

ax.set_xticks(x + width * (n / 2 - 0.5))
ax.set_xticklabels(metric_cols, fontsize=11)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.15)
ax.legend(fontsize=8, ncol=3, loc='upper right')
ax.set_title('Segmentation Metrics — BLAST vs 5 U-Net Variants', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'metrics_comparison_all.png'), dpi=120, bbox_inches='tight')
plt.show()

# Radar chart — all 9 methods
angles = np.linspace(0, 2 * np.pi, len(metric_cols), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

for i, method in enumerate(method_names):
    vals = metrics_df.loc[method, metric_cols].values.astype(float).tolist()
    vals += vals[:1]
    ax.plot(angles, vals, 'o-', linewidth=1.5, label=method, color=colors_9[i], markersize=4)
    ax.fill(angles, vals, alpha=0.05, color=colors_9[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_cols, fontsize=10)
ax.set_ylim(0, 1)
ax.legend(loc='lower right', fontsize=8, ncol=2)
ax.set_title('Radar Chart — All Methods', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'radar_chart_all.png'), dpi=120, bbox_inches='tight')
plt.show()

# Per-class Dice heatmap
dice_data = {}
for m in metrics_list:
    dice_data[m['Method']] = m['Dice_per_class']

dice_df = pd.DataFrame(dice_data, index=CLASS_NAMES).T

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(dice_df, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax, vmin=0, vmax=1)
ax.set_title('Per-Class Dice Coefficient by Method', fontsize=14)
ax.set_ylabel('Method')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dice_heatmap.png'), dpi=120, bbox_inches='tight')
plt.show()

# Euclidean distance comparison
euclidean_metrics = []
cosine_metrics_subset = []

for name, feats in [('LBP', lbp_features), ('MLBP', mlbp_features), ('GLCM', glcm_features)]:
    # Euclidean
    preds_e, seg_maps_e, _ = leave_one_image_out_blast(feats, all_patch_labels, image_indices, metric='euclidean')
    m_e = compute_segmentation_metrics(masks, seg_maps_e, f'{name}-Euclidean')
    euclidean_metrics.append({'Method': name, 'Metric': 'Euclidean',
                              'Pixel Accuracy': m_e['Pixel Accuracy'],
                              'Mean IoU': m_e['Mean IoU'], 'Mean Dice': m_e['Mean Dice']})
    
    # Cosine (already computed)
    m_c = compute_segmentation_metrics(masks, results[name]['seg_maps'], f'{name}-Cosine')
    cosine_metrics_subset.append({'Method': name, 'Metric': 'Cosine',
                                  'Pixel Accuracy': m_c['Pixel Accuracy'],
                                  'Mean IoU': m_c['Mean IoU'], 'Mean Dice': m_c['Mean Dice']})

dist_df = pd.DataFrame(cosine_metrics_subset + euclidean_metrics)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, metric_name in enumerate(['Pixel Accuracy', 'Mean IoU', 'Mean Dice']):
    pivot = dist_df.pivot(index='Method', columns='Metric', values=metric_name)
    pivot.plot(kind='bar', ax=axes[i], color=['#4C72B0', '#C44E52'])
    axes[i].set_title(metric_name)
    axes[i].set_ylabel('Score')
    axes[i].set_ylim(0, 1.1)
    axes[i].tick_params(axis='x', rotation=0)
    axes[i].legend(title='Distance')

plt.suptitle('Cosine vs Euclidean Distance Comparison', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'distance_metric_comparison.png'), dpi=120, bbox_inches='tight')
plt.show()

# 8a: Effect of K
K_values = [1, 3, 5, 7, 9, 11]
k_results = {name: [] for name in ['LBP', 'MLBP', 'GLCM']}

for K in K_values:
    for name, feats in [('LBP', lbp_features), ('MLBP', mlbp_features), ('GLCM', glcm_features)]:
        preds, seg_maps, _ = leave_one_image_out_blast(feats, all_patch_labels, image_indices, top_k=K)
        m = compute_segmentation_metrics(masks, seg_maps, name)
        k_results[name].append(m['Pixel Accuracy'])

fig, ax = plt.subplots(figsize=(10, 6))
for name, accs in k_results.items():
    ax.plot(K_values, accs, 'o-', linewidth=2, markersize=8, label=name)

ax.set_xlabel('K (Number of Top Matches)', fontsize=12)
ax.set_ylabel('Pixel Accuracy', fontsize=12)
ax.set_title('Effect of K on Segmentation Accuracy', fontsize=14)
ax.legend(fontsize=11)
ax.set_xticks(K_values)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sensitivity_k.png'), dpi=120, bbox_inches='tight')
plt.show()

# 8b: Effect of patch size
PATCH_SIZES = [32, 64, 128]
ps_results = {name: [] for name in ['LBP', 'MLBP', 'GLCM']}

for ps in PATCH_SIZES:
    pps = IMG_SIZE // ps
    n_patches = pps ** 2
    
    # Re-extract patches at this size
    ps_patches = []
    ps_labels = []
    ps_img_idx = []
    for i in range(NUM_IMAGES):
        for r in range(pps):
            for c in range(pps):
                y0, y1 = r * ps, (r + 1) * ps
                x0, x1 = c * ps, (c + 1) * ps
                patch = images[i][y0:y1, x0:x1]
                patch_mask = masks[i][y0:y1, x0:x1]
                vals, cnts = np.unique(patch_mask, return_counts=True)
                ps_patches.append(patch)
                ps_labels.append(vals[np.argmax(cnts)])
                ps_img_idx.append(i)
    
    ps_patches = np.array(ps_patches)
    ps_labels = np.array(ps_labels)
    ps_img_idx = np.array(ps_img_idx)
    
    # Extract features and run BLAST
    for name, extractor in [('LBP', extract_lbp), ('MLBP', extract_mlbp), ('GLCM', extract_glcm)]:
        feats = np.array([extractor(p) for p in ps_patches])
        
        # LOIO with custom patch grid
        all_preds_ps = np.zeros(len(ps_labels), dtype=int)
        seg_maps_ps = {}
        for img_id in range(NUM_IMAGES):
            test_m = ps_img_idx == img_id
            train_m = ~test_m
            db = BLASTImageDatabase(metric='cosine', top_k=TOP_K)
            db.build_database(feats[train_m], ps_labels[train_m])
            
            test_feats = feats[test_m]
            pred_labels_ps = []
            for f in test_feats:
                res = db.query(f)
                pred_labels_ps.append(res['predicted'])
            
            pred_grid = np.array(pred_labels_ps).reshape(pps, pps)
            seg_map = np.kron(pred_grid, np.ones((ps, ps), dtype=np.uint8))
            seg_maps_ps[img_id] = seg_map
            all_preds_ps[test_m] = pred_grid.flatten()
        
        m = compute_segmentation_metrics(masks, seg_maps_ps, name)
        ps_results[name].append(m['Pixel Accuracy'])

fig, ax = plt.subplots(figsize=(10, 6))
for name, accs in ps_results.items():
    ax.plot(PATCH_SIZES, accs, 'o-', linewidth=2, markersize=8, label=name)

ax.set_xlabel('Patch Size (pixels)', fontsize=12)
ax.set_ylabel('Pixel Accuracy', fontsize=12)
ax.set_title('Effect of Patch Size on Segmentation Accuracy', fontsize=14)
ax.legend(fontsize=11)
ax.set_xticks(PATCH_SIZES)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sensitivity_patch_size.png'), dpi=120, bbox_inches='tight')
plt.show()

# ── Define 5 pretrained feature extractors ──
from tensorflow.keras.applications import (
    ResNet50, VGG16, DenseNet121, MobileNetV2, EfficientNetB0
)
from tensorflow.keras.applications import (
    resnet50, vgg16, densenet, mobilenet_v2, efficientnet
)

PRETRAINED_MODELS = {
    'ResNet50':       (ResNet50, resnet50.preprocess_input),
    'VGG16':          (VGG16, vgg16.preprocess_input),
    'DenseNet121':    (DenseNet121, densenet.preprocess_input),
    'MobileNetV2':    (MobileNetV2, mobilenet_v2.preprocess_input),
    'EfficientNetB0': (EfficientNetB0, efficientnet.preprocess_input),
}

def build_patch_feature_extractor(model_class, input_shape=(PATCH_SIZE, PATCH_SIZE, 3)):
    """Build a frozen pretrained model for patch-level feature extraction."""
    base = model_class(weights='imagenet', include_top=False,
                       input_shape=input_shape, pooling='avg')
    base.trainable = False
    return base

# Show embedding dimensions
print(f'{"Model":<18} {"Embedding Dims":>15}')
print('─' * 35)
for name, (cls, _) in PRETRAINED_MODELS.items():
    ext = build_patch_feature_extractor(cls)
    print(f'{name:<18} {ext.output_shape[-1]:>15}')
    del ext
    tf.keras.backend.clear_session()

# ══════════════════════════════════════════════════════════════
# MODE 1: WITH BLAST — Pretrained features → BLAST matching
# ══════════════════════════════════════════════════════════════
import gc

print('MODE 1: Pretrained Models + BLAST Pipeline')
print('=' * 55)

pretrained_blast_results = {}
pretrained_blast_interim = []

for model_name, (model_class, preprocess_fn) in PRETRAINED_MODELS.items():
    print(f'\n── {model_name} ──')
    
    tf.keras.backend.clear_session()
    gc.collect()
    
    extractor = build_patch_feature_extractor(model_class)
    patches_float = all_patches.astype(np.float32)
    patches_prep = np.array([preprocess_fn(p.copy()) for p in patches_float])
    embeddings = extractor.predict(patches_prep, batch_size=16, verbose=0)
    print(f'  Embedding shape: {embeddings.shape}')
    
    preds, seg_maps, conf_maps = leave_one_image_out_blast(
        embeddings, all_patch_labels, image_indices, metric='cosine')
    
    pretrained_blast_results[model_name] = {
        'preds': preds, 'seg_maps': seg_maps, 'conf_maps': conf_maps
    }
    
    m = compute_segmentation_metrics(masks, seg_maps, f'{model_name} + BLAST')
    pretrained_blast_interim.append({k: v for k, v in m.items() if k not in ['Dice_per_class', 'all_true', 'all_pred']})
    print(f'  ✓ {model_name} + BLAST complete:')
    print(f'    Pixel Accuracy: {m["Pixel Accuracy"]:.4f}  |  Mean IoU: {m["Mean IoU"]:.4f}  |  Mean Dice: {m["Mean Dice"]:.4f}  |  F1: {m["F1 Score"]:.4f}')
    
    del extractor, patches_prep, embeddings
    tf.keras.backend.clear_session()
    gc.collect()

print('\n── Pretrained + BLAST Summary ──')
print(pd.DataFrame(pretrained_blast_interim).set_index('Method').round(4).to_string())


# ══════════════════════════════════════════════════════════════
# MODE 2: WITHOUT BLAST — Pretrained encoder + decoder (direct segmentation)
# ══════════════════════════════════════════════════════════════
import gc

print('MODE 2: Pretrained Models — Direct Segmentation (No BLAST)')
print('=' * 60)

def build_pretrained_segmentation(model_class, preprocess_fn,
                                  input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                  num_classes=NUM_CLASSES):
    """Pretrained encoder (frozen) + minimal trainable decoder with Resizing."""
    inputs = layers.Input(shape=input_shape)
    preprocessed = layers.Lambda(lambda x: preprocess_fn(x))(inputs)
    
    encoder = model_class(weights='imagenet', include_top=False,
                          input_shape=input_shape)
    encoder.trainable = False
    encoded = encoder(preprocessed)
    
    x = layers.Conv2D(64, 1, activation='relu')(encoded)
    x = layers.Resizing(IMG_SIZE, IMG_SIZE, interpolation='bilinear')(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(num_classes, 1, activation='softmax')(x)
    
    model = Model(inputs, x)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

pretrained_direct_seg_maps = {}
pretrained_direct_interim = []

for model_name, (model_class, preprocess_fn) in PRETRAINED_MODELS.items():
    print(f'\n── {model_name} (Direct Segmentation) ──')
    
    seg_maps = {}
    pixel_preds = []
    pixel_trues = []
    t0 = time.time()
    
    for img_id in range(NUM_IMAGES):
        train_idx = [i for i in range(NUM_IMAGES) if i != img_id]
        X_train = images_np[train_idx]
        y_train = masks_np[train_idx][..., np.newaxis]
        X_test = images_np[img_id:img_id+1]
        
        tf.keras.backend.clear_session()
        gc.collect()
        fold_model = build_pretrained_segmentation(model_class, preprocess_fn)
        fold_model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=0)
        
        pred_prob = fold_model.predict(X_test, verbose=0)[0]
        pred_seg = np.argmax(pred_prob, axis=-1).astype(np.uint8)
        
        seg_maps[img_id] = pred_seg
        pixel_preds.append(pred_seg.flatten())
        pixel_trues.append(masks_np[img_id].flatten())
        
        del fold_model
        tf.keras.backend.clear_session()
        gc.collect()
        print(f'  Fold {img_id}: done')
    
    elapsed = time.time() - t0
    pretrained_direct_seg_maps[model_name] = seg_maps
    
    m = compute_segmentation_metrics(masks, seg_maps, f'{model_name} (Direct)')
    pretrained_direct_interim.append({k: v for k, v in m.items() if k not in ['Dice_per_class', 'all_true', 'all_pred']})
    print(f'\n  ✓ {model_name} (Direct) complete ({elapsed:.0f}s):')
    print(f'    Pixel Accuracy: {m["Pixel Accuracy"]:.4f}  |  Mean IoU: {m["Mean IoU"]:.4f}  |  Mean Dice: {m["Mean Dice"]:.4f}  |  F1: {m["F1 Score"]:.4f}')

print('\n── Pretrained Direct Segmentation Summary ──')
print(pd.DataFrame(pretrained_direct_interim).set_index('Method').round(4).to_string())


# ══════════════════════════════════════════════════════════════
# COMPARISON TABLE: With BLAST vs Without BLAST
# ══════════════════════════════════════════════════════════════
pretrained_metrics = []

for model_name in PRETRAINED_MODELS.keys():
    # With BLAST
    m_blast = compute_segmentation_metrics(
        masks, pretrained_blast_results[model_name]['seg_maps'],
        f'{model_name} + BLAST')
    pretrained_metrics.append(m_blast)
    
    # Without BLAST (direct segmentation)
    m_direct = compute_segmentation_metrics(
        masks, pretrained_direct_seg_maps[model_name],
        f'{model_name} (Direct)')
    pretrained_metrics.append(m_direct)

# Build comparison DataFrame
pretrained_df = pd.DataFrame([
    {k: v for k, v in m.items() if k not in ['Dice_per_class', 'all_true', 'all_pred']}
    for m in pretrained_metrics
]).set_index('Method').round(4)

print('\n' + '=' * 90)
print('PRETRAINED MODELS: WITH BLAST vs WITHOUT BLAST (Direct Segmentation)')
print('=' * 90)
print(pretrained_df.to_string())

# ── Side-by-side bar chart: BLAST vs Direct for each pretrained model ──
model_names = list(PRETRAINED_MODELS.keys())
metric_cols_sub = ['Pixel Accuracy', 'Mean IoU', 'Mean Dice', 'F1 Score']

fig, axes = plt.subplots(1, len(metric_cols_sub), figsize=(20, 6))

for ax_idx, metric_name in enumerate(metric_cols_sub):
    blast_vals = [pretrained_df.loc[f'{m} + BLAST', metric_name] for m in model_names]
    direct_vals = [pretrained_df.loc[f'{m} (Direct)', metric_name] for m in model_names]
    
    x = np.arange(len(model_names))
    w = 0.35
    axes[ax_idx].bar(x - w/2, blast_vals, w, label='With BLAST', color='#4C72B0')
    axes[ax_idx].bar(x + w/2, direct_vals, w, label='Without BLAST', color='#C44E52')
    axes[ax_idx].set_xticks(x)
    axes[ax_idx].set_xticklabels(model_names, rotation=30, ha='right', fontsize=9)
    axes[ax_idx].set_title(metric_name, fontsize=12)
    axes[ax_idx].set_ylim(0, 1.1)
    axes[ax_idx].legend(fontsize=8)

plt.suptitle('Pretrained Models: With BLAST vs Without BLAST', fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pretrained_blast_vs_direct.png'), dpi=120, bbox_inches='tight')
plt.show()

# ── Visual comparison: BLAST vs Direct segmentation maps ──
sample_imgs = [0, 4, 8]
model_names_vis = list(PRETRAINED_MODELS.keys())

fig, axes = plt.subplots(len(sample_imgs), 2 + 2 * len(model_names_vis),
                         figsize=(4 * (2 + 2 * len(model_names_vis)), 4 * len(sample_imgs)))

for row, img_id in enumerate(sample_imgs):
    # Original
    axes[row, 0].imshow(images[img_id])
    axes[row, 0].set_title('Original' if row == 0 else '', fontsize=9)
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
        seg_b = pretrained_blast_results[mname]['seg_maps'][img_id]
        seg_b_color = np.zeros((*seg_b.shape, 3), dtype=np.uint8)
        for c, clr in CLASS_COLORS.items():
            seg_b_color[seg_b == c] = clr
        axes[row, col].imshow(seg_b_color)
        axes[row, col].set_title(f'{mname}\n+BLAST' if row == 0 else '', fontsize=8)
        axes[row, col].axis('off')
        col += 1
        
        # Without BLAST
        seg_d = pretrained_direct_seg_maps[mname][img_id]
        seg_d_color = np.zeros((*seg_d.shape, 3), dtype=np.uint8)
        for c, clr in CLASS_COLORS.items():
            seg_d_color[seg_d == c] = clr
        axes[row, col].imshow(seg_d_color)
        axes[row, col].set_title(f'{mname}\nDirect' if row == 0 else '', fontsize=8)
        axes[row, col].axis('off')
        col += 1

legend_patches = [mpatches.Patch(color=np.array(c)/255, label=n) for n, c in zip(CLASS_NAMES, CLASS_COLORS.values())]
fig.legend(handles=legend_patches, loc='upper center', ncol=3, fontsize=11)
plt.suptitle('Pretrained Models: BLAST vs Direct Segmentation Maps', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pretrained_segmentation_maps.png'), dpi=100, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════
# GRAND SUMMARY TABLE — All methods combined
# ══════════════════════════════════════════════════════════════

# Merge all metrics into one grand table
all_metrics_combined = []

# BLAST with handcrafted/CNN features
for method in ['LBP', 'MLBP', 'GLCM', 'CNN']:
    m = compute_segmentation_metrics(masks, results[method]['seg_maps'], f'BLAST-{method}')
    all_metrics_combined.append({**{k:v for k,v in m.items() if k not in ['Dice_per_class','all_true','all_pred']},
                                  'Category': 'BLAST (Custom)'})

# 5 U-Net variants
for unet_name in unet_builders.keys():
    m = compute_segmentation_metrics(masks, unet_all_seg_maps[unet_name], unet_name)
    all_metrics_combined.append({**{k:v for k,v in m.items() if k not in ['Dice_per_class','all_true','all_pred']},
                                  'Category': 'U-Net Variant'})

# Pretrained + BLAST
for model_name in PRETRAINED_MODELS.keys():
    m = compute_segmentation_metrics(masks, pretrained_blast_results[model_name]['seg_maps'],
                                     f'{model_name} + BLAST')
    all_metrics_combined.append({**{k:v for k,v in m.items() if k not in ['Dice_per_class','all_true','all_pred']},
                                  'Category': 'Pretrained + BLAST'})

# Pretrained Direct
for model_name in PRETRAINED_MODELS.keys():
    m = compute_segmentation_metrics(masks, pretrained_direct_seg_maps[model_name],
                                     f'{model_name} (Direct)')
    all_metrics_combined.append({**{k:v for k,v in m.items() if k not in ['Dice_per_class','all_true','all_pred']},
                                  'Category': 'Pretrained Direct'})

grand_df = pd.DataFrame(all_metrics_combined).set_index('Method').round(4)
grand_df = grand_df.sort_values('Pixel Accuracy', ascending=False)

# Also save to CSV
grand_df.to_csv(os.path.join(OUTPUT_DIR, 'grand_comparison.csv'))

print('\n' + '=' * 100)
print('GRAND COMPARISON TABLE — All 24 Methods Ranked by Pixel Accuracy')
print('=' * 100)
print(grand_df.to_string())

# Save final metrics to CSV
metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'segmentation_metrics.csv'))
print('Metrics saved to outputs/segmentation_metrics.csv')
print(f'\nAll output files:')
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f'  {f}')
print(f'\nData files:')
for f in sorted(os.listdir(DATA_DIR)):
    print(f'  {f}')
print(f'\nLabels:')
print(f'  data/labels.csv')
print('\nDone! All outputs generated successfully.')
