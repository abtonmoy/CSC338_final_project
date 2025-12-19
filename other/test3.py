# VesselMNIST3D: Intracranial Aneurysm Detection
# Building upon the starter notebook with CNN architecture

"""
MILESTONE 1: Dataset Understanding
Dataset: VesselMNIST3D
Source: IntrA dataset - 3D intracranial aneurysm dataset
Imaging Modality: MRA (Magnetic Resonance Angiography)
Task: Binary classification - detecting presence of aneurysms in brain vessel segments
Classes: 0 = Healthy vessel segment, 1 = Aneurysm segment
Images: 28x28x28 3D volumes

Medical Context:
An intracranial aneurysm is a bulging or ballooning in a blood vessel in the brain. 
Early detection is crucial as rupture can lead to hemorrhagic stroke. MRA imaging 
is a non-invasive technique to visualize brain vasculature without contrast agents.
"""

# ============================================================================
# PREAMBLE: Import Libraries, Download Dataset, Preprocess Data
# ============================================================================

import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import colors
from matplotlib.widgets import Slider
import matplotlib
import matplotlib.font_manager

from medmnist import VesselMNIST3D
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
print("Available devices:", tf.config.list_physical_devices())

# ============================================================================
# LOAD AND PREPARE DATA (using starter notebook approach)
# ============================================================================

train_dataset = VesselMNIST3D(split='train', size=28, download=True)
trainx = []
trainy = []

test_dataset = VesselMNIST3D(split='test', size=28, download=True)
testx = []
testy = []

val_dataset = VesselMNIST3D(split='val', size=28, download=True)  # Fixed: was 'train'
valx = []
valy = []

for i in range(len(train_dataset)):
    trainx.append(train_dataset[i][0])
    trainy.append(train_dataset[i][1])

for i in range(len(test_dataset)):
    testx.append(test_dataset[i][0])
    testy.append(test_dataset[i][1])

for i in range(len(val_dataset)):
    valx.append(val_dataset[i][0])
    valy.append(val_dataset[i][1])

trainx_tensor = tf.convert_to_tensor(trainx, dtype=tf.float32)  # Using float32 for now
trainy_tensor = tf.convert_to_tensor(trainy, dtype=tf.float32)
testx_tensor = tf.convert_to_tensor(testx, dtype=tf.float32)
testy_tensor = tf.convert_to_tensor(testy, dtype=tf.float32)
valx_tensor = tf.convert_to_tensor(valx, dtype=tf.float32)
valy_tensor = tf.convert_to_tensor(valy, dtype=tf.float32)

print(f"\nData loaded successfully!")
print(f"Training set: {trainx_tensor.shape}")
print(f"Validation set: {valx_tensor.shape}")
print(f"Test set: {testx_tensor.shape}")

# ============================================================================
# MILESTONE 1: UNDERSTAND AND VISUALIZE DATA
# ============================================================================

print("\n" + "="*70)
print("DATASET ANALYSIS")
print("="*70)

# Analyze class distribution
train_labels = np.array(trainy).flatten()
val_labels = np.array(valy).flatten()
test_labels = np.array(testy).flatten()

unique_train, counts_train = np.unique(train_labels, return_counts=True)
unique_val, counts_val = np.unique(val_labels, return_counts=True)
unique_test, counts_test = np.unique(test_labels, return_counts=True)

print(f"Training - Class 0 (Healthy): {counts_train[0]}, Class 1 (Aneurysm): {counts_train[1]}")
print(f"Validation - Class 0 (Healthy): {counts_val[0]}, Class 1 (Aneurysm): {counts_val[1]}")
print(f"Test - Class 0 (Healthy): {counts_test[0]}, Class 1 (Aneurysm): {counts_test[1]}")
print(f"\nClass imbalance ratio (train): {counts_train[1]/counts_train[0]:.3f}")

# Visualize class distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, counts) in zip(axes, 
    [('Training', counts_train), ('Validation', counts_val), ('Test', counts_test)]):
    ax.bar(['Healthy', 'Aneurysm'], counts, color=['steelblue', 'coral'])
    ax.set_ylabel('Count')
    ax.set_title(f'{name} Set Distribution')
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================================
# VISUALIZATIONS FROM STARTER NOTEBOOK
# ============================================================================

# Example 1: 3D Voxel Visualization
fig = plt.figure(figsize=(10, 8))
vol = np.squeeze(trainx[1], axis=0)  # shape (28, 28, 28)
ax = fig.add_subplot(111, projection='3d')

filled = vol > 0
norm = colors.Normalize(vmin=vol.min(), vmax=vol.max())
cmap = plt.cm.viridis
facecolors = cmap(norm(vol))
alpha = np.clip(vol, 0, 1)
facecolors[..., 3] = alpha
facecolors[~filled, 3] = 0.0

ax.voxels(filled, facecolors=facecolors)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.title(f'3D Voxel Visualization - Label: {trainy[1]}')
plt.show()

# Example 2: Orthogonal Slices (Middle Slices)
vol = np.array(trainx[0]).squeeze()
i_mid = vol.shape[0] // 2
j_mid = vol.shape[1] // 2
k_mid = vol.shape[2] // 2

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(vol[i_mid, :, :], cmap='gray')
axes[0].set_title(f'Axial (i={i_mid}) - Label: {trainy[0]}')
axes[0].axis('off')

axes[1].imshow(vol[:, j_mid, :], cmap='gray')
axes[1].set_title(f'Coronal (j={j_mid})')
axes[1].axis('off')

axes[2].imshow(vol[:, :, k_mid], cmap='gray')
axes[2].set_title(f'Sagittal (k={k_mid})')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# Example 3: Show healthy vs aneurysm examples
healthy_idx = np.where(train_labels == 0)[0][0]
aneurysm_idx = np.where(train_labels == 1)[0][0]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('Healthy vs Aneurysm Comparison', fontsize=14, fontweight='bold')

# Healthy example
vol_healthy = np.array(trainx[healthy_idx]).squeeze()
mid = vol_healthy.shape[0] // 2
axes[0, 0].imshow(vol_healthy[mid, :, :], cmap='gray')
axes[0, 0].set_title('Healthy - Axial')
axes[0, 0].axis('off')
axes[0, 1].imshow(vol_healthy[:, mid, :], cmap='gray')
axes[0, 1].set_title('Healthy - Coronal')
axes[0, 1].axis('off')
axes[0, 2].imshow(vol_healthy[:, :, mid], cmap='gray')
axes[0, 2].set_title('Healthy - Sagittal')
axes[0, 2].axis('off')

# Aneurysm example
vol_aneurysm = np.array(trainx[aneurysm_idx]).squeeze()
mid = vol_aneurysm.shape[0] // 2
axes[1, 0].imshow(vol_aneurysm[mid, :, :], cmap='gray')
axes[1, 0].set_title('Aneurysm - Axial')
axes[1, 0].axis('off')
axes[1, 1].imshow(vol_aneurysm[:, mid, :], cmap='gray')
axes[1, 1].set_title('Aneurysm - Coronal')
axes[1, 1].axis('off')
axes[1, 2].imshow(vol_aneurysm[:, :, mid], cmap='gray')
axes[1, 2].set_title('Aneurysm - Sagittal')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# ============================================================================
# MILESTONE 2: DATA PREPROCESSING FOR NEURAL NETWORK
# ============================================================================

print("\n" + "="*70)
print("DATA PREPROCESSING")
print("="*70)

# Normalize data to [0, 1] range
trainx_norm = trainx_tensor / 255.0
valx_norm = valx_tensor / 255.0
testx_norm = testx_tensor / 255.0

# Flatten labels
trainy_flat = tf.squeeze(trainy_tensor)
valy_flat = tf.squeeze(valy_tensor)
testy_flat = tf.squeeze(testy_tensor)

print(f"Normalized training data shape: {trainx_norm.shape}")
print(f"Training labels shape: {trainy_flat.shape}")
print(f"Data range: [{tf.reduce_min(trainx_norm):.3f}, {tf.reduce_max(trainx_norm):.3f}]")

# Calculate class weights for imbalanced data
class_weight = {
    0: len(train_labels) / (2 * counts_train[0]),
    1: len(train_labels) / (2 * counts_train[1])
}
print(f"\nClass weights to handle imbalance: {class_weight}")

# ============================================================================
# MILESTONE 2: NEURAL NETWORK ARCHITECTURE
# ============================================================================

print("\n" + "="*70)
print("BUILDING 3D CNN ARCHITECTURE")
print("="*70)

"""
Architecture Design Rationale:
- 3D Convolutional layers to capture spatial relationships in volumetric data
- Progressive downsampling (28→14→7→3) to build hierarchical features
- Batch normalization for training stability
- Dropout for regularization (critical for small dataset)
- Dense layers for final classification
- Sigmoid activation for binary classification

The architecture processes the 3D volume through multiple convolutional blocks,
each extracting increasingly abstract features before making a classification.
"""

def build_3d_cnn(input_shape=(28, 28, 28, 1), dropout_rate=0.3):
    """
    Build a 3D CNN for binary classification of vessel segments
    
    Parameters:
    - input_shape: Shape of input 3D volume with channel
    - dropout_rate: Dropout probability for regularization
    
    Returns:
    - Compiled Keras model
    """
    
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Block 1: Initial feature extraction (28x28x28 → 14x14x14)
        layers.Conv3D(32, kernel_size=(3, 3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),
        layers.Dropout(dropout_rate),
        
        # Block 2: Deeper features (14x14x14 → 7x7x7)
        layers.Conv3D(64, kernel_size=(3, 3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),
        layers.Dropout(dropout_rate),
        
        # Block 3: High-level features (7x7x7 → 3x3x3)
        layers.Conv3D(128, kernel_size=(3, 3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),
        layers.Dropout(dropout_rate),
        
        # Global pooling to reduce parameters
        layers.GlobalAveragePooling3D(),
        
        # Dense layers for classification
        layers.Dense(256, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_rate),
        
        # Output layer (sigmoid for binary classification)
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

# Build and display model
model = build_3d_cnn(input_shape=(28, 28, 28, 1), dropout_rate=0.3)
model.summary()

print("\nTotal parameters:", model.count_params())

# ============================================================================
# MILESTONE 2: MODEL COMPILATION
# ============================================================================

print("\n" + "="*70)
print("COMPILING MODEL")
print("="*70)

"""
Loss Function: Binary Crossentropy
- Standard for binary classification tasks
- Measures difference between predicted and actual probabilities

Optimizer: Adam
- Adaptive learning rate optimizer
- Works well with default parameters for most problems

Metrics:
- Accuracy: Overall correctness
- AUC: Area Under ROC Curve - crucial for imbalanced medical datasets
  Measures the model's ability to distinguish between classes
"""

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

print("Model compiled with:")
print("- Optimizer: Adam (lr=0.001)")
print("- Loss: Binary Crossentropy")
print("- Metrics: Accuracy, AUC")

# ============================================================================
# MILESTONE 2: TRAINING CALLBACKS
# ============================================================================

# Early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_auc',
    patience=15,
    restore_best_weights=True,
    mode='max',
    verbose=1
)

# Reduce learning rate when validation loss plateaus
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Save best model
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_vessel_model.h5',
    monitor='val_auc',
    save_best_only=True,
    mode='max',
    verbose=1
)

print("\nCallbacks configured:")
print("- Early Stopping: patience=15, monitor=val_auc")
print("- ReduceLROnPlateau: factor=0.5, patience=5")
print("- ModelCheckpoint: saves best val_auc")

# ============================================================================
# MILESTONE 2: INITIAL TEST RUN (2 EPOCHS)
# ============================================================================

print("\n" + "="*70)
print("RUNNING INITIAL TEST EPOCHS (MILESTONE 2)")
print("="*70)
print("Training for 2 epochs to verify architecture is learning...\n")

# Train for 2 epochs to confirm the model is working
history = model.fit(
    trainx_norm, trainy_flat,
    batch_size=16,
    epochs=2,
    validation_data=(valx_norm, valy_flat),
    class_weight=class_weight,
    verbose=1
)

print("\n" + "="*70)
print("INITIAL TEST RESULTS")
print("="*70)
print(f"Epoch 1 → Training Loss: {history.history['loss'][0]:.4f}, "
      f"Val Loss: {history.history['val_loss'][0]:.4f}")
print(f"Epoch 2 → Training Loss: {history.history['loss'][1]:.4f}, "
      f"Val Loss: {history.history['val_loss'][1]:.4f}")
print(f"\nTraining AUC: {history.history['auc'][-1]:.4f}")
print(f"Validation AUC: {history.history['val_auc'][-1]:.4f}")

# Check if loss is decreasing (sign of learning)
if history.history['loss'][1] < history.history['loss'][0]:
    print("\n Loss is decreasing - model is learning!")
else:
    print("\n Loss not decreasing - may need to adjust hyperparameters")

print("\n" + "="*70)
print("MILESTONE 2 COMPLETE!")
print("="*70)
print("✓ Dataset loaded and explored")
print("✓ Neural network architecture designed and built")
print("✓ Model compiled with appropriate loss function (binary crossentropy)")
print("✓ Initial test epochs completed - architecture is working!")

# ============================================================================
# MILESTONE 3: FULL TRAINING ON GPU
# ============================================================================

print("\n" + "="*70)
print("READY FOR MILESTONE 3: GPU TRAINING")
print("="*70)
print("Uncomment the section below when ready to run on shared GPU system")

"""
# UNCOMMENT FOR FULL TRAINING (MILESTONE 3)

print("Starting full training session...")

# For GPU: can use float16 to save memory
# trainx_tensor = tf.cast(trainx_tensor, tf.float16) / 255.0
# valx_tensor = tf.cast(valx_tensor, tf.float16) / 255.0
# model = build_3d_cnn(input_shape=(28, 28, 28, 1), dropout_rate=0.3)
# model.compile(...) # recompile for float16

history_full = model.fit(
    trainx_norm, trainy_flat,
    batch_size=32,  # Larger batch for GPU
    epochs=100,  # Will stop early with callback
    validation_data=(valx_norm, valy_flat),
    class_weight=class_weight,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

# Plot training history
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss
axes[0].plot(history_full.history['loss'], label='Training Loss', linewidth=2)
axes[0].plot(history_full.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Model Loss Over Time')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Accuracy
axes[1].plot(history_full.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[1].plot(history_full.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Model Accuracy Over Time')
axes[1].legend()
axes[1].grid(alpha=0.3)

# AUC
axes[2].plot(history_full.history['auc'], label='Training AUC', linewidth=2)
axes[2].plot(history_full.history['val_auc'], label='Validation AUC', linewidth=2)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('AUC')
axes[2].set_title('Model AUC Over Time')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# MILESTONE 3: MODEL EVALUATION
# ============================================================================

print("\n" + "="*70)
print("MODEL EVALUATION ON TEST SET")
print("="*70)

# Load best model
model = keras.models.load_model('best_vessel_model.h5')

# Evaluate on test set
test_loss, test_acc, test_auc = model.evaluate(testx_norm, testy_flat, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# Get predictions
y_pred_proba = model.predict(testx_norm, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Classification report
print("\nClassification Report:")
print(classification_report(test_labels, y_pred, 
                          target_names=['Healthy', 'Aneurysm'],
                          digits=4))

# Confusion matrix
cm = confusion_matrix(test_labels, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Aneurysm'],
            yticklabels=['Healthy', 'Aneurysm'])
plt.title('Confusion Matrix - Test Set')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# ROC curve
fpr, tpr, thresholds = roc_curve(test_labels, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'Model (AUC = {test_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ============================================================================
# COMPARISON WITH MEDMNIST BENCHMARKS
# ============================================================================

print("\n" + "="*70)
print("COMPARISON WITH MEDMNIST BENCHMARKS")
print("="*70)
print("VesselMNIST3D Baseline Performance (from MedMNIST paper):")
print("- ResNet18 (3D): AUC ~0.920, ACC ~0.890")
print("- Auto-sklearn: AUC ~0.917, ACC ~0.887")
print("\nYour Model Performance:")
print(f"- Test AUC: {test_auc:.4f}")
print(f"- Test ACC: {test_acc:.4f}")

if test_auc >= 0.920:
    print("\n🎉 Excellent! Your model matches or exceeds the baseline!")
elif test_auc >= 0.900:
    print("\n✓ Good performance! Close to the baseline.")
else:
    print("\n→ Room for improvement. Consider:")
    print("  - Data augmentation (rotations, flips)")
    print("  - Deeper architecture or residual connections")
    print("  - Different regularization strategies")
    print("  - Ensemble methods")
"""

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("1. Run full training on GPU (uncomment Milestone 3 code)")
print("2. Analyze results and compare to benchmarks")
print("3. Prepare presentation covering:")
print("   - Dataset and medical task")
print("   - Architecture design choices")
print("   - Results vs benchmarks")
print("   - Ethical considerations")