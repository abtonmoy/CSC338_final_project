# VesselMNIST3D: Intracranial Aneurysm Detection
# Dataset Exploration and Neural Network Architecture

"""
Dataset Information:
- Source: IntrA dataset - 3D intracranial aneurysm dataset
- Imaging Modality: MRA (Magnetic Resonance Angiography)
- Task: Binary classification - detecting presence of aneurysms in brain vessel segments
- Classes: 0 = Healthy vessel segment, 1 = Aneurysm segment
- Images: 28x28x28 3D volumes
- Total samples: 1,909 (1,694 healthy + 215 aneurysm segments)
- Class imbalance: ~11% aneurysm cases

Medical Context:
An intracranial aneurysm is a bulging or ballooning in a blood vessel in the brain. 
Early detection is crucial as rupture can lead to hemorrhagic stroke. MRA imaging 
is a non-invasive technique to visualize brain vasculature.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import medmnist
from medmnist import INFO
import tensorflow as tf
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
# SECTION 1: DATA LOADING AND EXPLORATION
# ============================================================================

# Load dataset information
data_flag = 'vesselmnist3d'
info = INFO[data_flag]

print("\n" + "="*70)
print("DATASET INFORMATION")
print("="*70)
print(f"Dataset: {info['python_class']}")
print(f"Task: {info['task']}")
print(f"Number of classes: {info['n_channels']}")
print(f"Number of samples: {info['n_samples']}")

# Download and load data
DataClass = getattr(medmnist, info['python_class'])
train_dataset = DataClass(split='train', download=True, size=28)
val_dataset = DataClass(split='val', download=True, size=28)
test_dataset = DataClass(split='test', download=True, size=28)

# Extract data
x_train, y_train = train_dataset.imgs, train_dataset.labels
x_val, y_val = val_dataset.imgs, val_dataset.labels
x_test, y_test = test_dataset.imgs, test_dataset.labels

print(f"\nTraining set: {x_train.shape}, Labels: {y_train.shape}")
print(f"Validation set: {x_val.shape}, Labels: {y_val.shape}")
print(f"Test set: {x_test.shape}, Labels: {y_test.shape}")

# Analyze class distribution
unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_val, counts_val = np.unique(y_val, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)

print("\n" + "="*70)
print("CLASS DISTRIBUTION ANALYSIS")
print("="*70)
print(f"Training - Class 0 (Healthy): {counts_train[0]}, Class 1 (Aneurysm): {counts_train[1]}")
print(f"Validation - Class 0 (Healthy): {counts_val[0]}, Class 1 (Aneurysm): {counts_val[1]}")
print(f"Test - Class 0 (Healthy): {counts_test[0]}, Class 1 (Aneurysm): {counts_test[1]}")
print(f"\nClass imbalance ratio (train): {counts_train[1]/counts_train[0]:.3f}")

# Visualize class distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, counts, unique) in zip(axes, 
    [('Training', counts_train, unique_train),
     ('Validation', counts_val, unique_val),
     ('Test', counts_test, unique_test)]):
    ax.bar(['Healthy', 'Aneurysm'], counts, color=['steelblue', 'coral'])
    ax.set_ylabel('Count')
    ax.set_title(f'{name} Set Distribution')
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================================
# SECTION 2: DATA VISUALIZATION
# ============================================================================

def visualize_3d_slices(volume, label, title="3D Volume Slices"):
    """Visualize three orthogonal slices through a 3D volume"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    mid_x, mid_y, mid_z = volume.shape[0]//2, volume.shape[1]//2, volume.shape[2]//2
    
    axes[0].imshow(volume[mid_x, :, :], cmap='gray')
    axes[0].set_title(f'Sagittal (X={mid_x})')
    axes[0].axis('off')
    
    axes[1].imshow(volume[:, mid_y, :], cmap='gray')
    axes[1].set_title(f'Coronal (Y={mid_y})')
    axes[1].axis('off')
    
    axes[2].imshow(volume[:, :, mid_z], cmap='gray')
    axes[2].set_title(f'Axial (Z={mid_z})')
    axes[2].axis('off')
    
    class_name = "Aneurysm" if label == 1 else "Healthy"
    fig.suptitle(f'{title} - {class_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Visualize examples from each class
print("\n" + "="*70)
print("VISUALIZING SAMPLE DATA")
print("="*70)

# Find indices of each class
healthy_idx = np.where(y_train == 0)[0]
aneurysm_idx = np.where(y_train == 1)[0]

# Visualize healthy vessel
visualize_3d_slices(x_train[healthy_idx[0]], y_train[healthy_idx[0]], 
                    "Healthy Vessel Segment")

# Visualize aneurysm
visualize_3d_slices(x_train[aneurysm_idx[0]], y_train[aneurysm_idx[0]], 
                    "Aneurysm Vessel Segment")

# Analyze intensity distributions
print("\nIntensity Statistics:")
print(f"Training data range: [{x_train.min()}, {x_train.max()}]")
print(f"Training data mean: {x_train.mean():.3f}, std: {x_train.std():.3f}")

# ============================================================================
# SECTION 3: DATA PREPROCESSING
# ============================================================================

print("\n" + "="*70)
print("DATA PREPROCESSING")
print("="*70)

# Normalize to [0, 1]
x_train_norm = x_train.astype('float32') / 255.0
x_val_norm = x_val.astype('float32') / 255.0
x_test_norm = x_test.astype('float32') / 255.0

# Add channel dimension for TensorFlow (batch, height, width, depth, channels)
x_train_norm = np.expand_dims(x_train_norm, axis=-1)
x_val_norm = np.expand_dims(x_val_norm, axis=-1)
x_test_norm = np.expand_dims(x_test_norm, axis=-1)

# Flatten labels
y_train_flat = y_train.flatten()
y_val_flat = y_val.flatten()
y_test_flat = y_test.flatten()

print(f"Preprocessed training shape: {x_train_norm.shape}")
print(f"Label shape: {y_train_flat.shape}")

# ============================================================================
# SECTION 4: NEURAL NETWORK ARCHITECTURE
# ============================================================================

print("\n" + "="*70)
print("BUILDING 3D CNN ARCHITECTURE")
print("="*70)

"""
Architecture Design Rationale:
- 3D Convolutional layers to capture spatial relationships in volumetric data
- Progressive downsampling with pooling to build hierarchical features
- Batch normalization for training stability
- Dropout for regularization (important given small dataset)
- Dense layers for final classification
- Sigmoid activation for binary classification
"""

def build_3d_cnn(input_shape=(28, 28, 28, 1), dropout_rate=0.3):
    """
    Build a 3D CNN for binary classification of vessel segments
    
    Architecture:
    - 3 convolutional blocks with increasing filters (32, 64, 128)
    - Each block: Conv3D -> BatchNorm -> ReLU -> MaxPool3D
    - Global Average Pooling to reduce parameters
    - Dense layers with dropout
    - Output: Single neuron with sigmoid for binary classification
    """
    
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Block 1: Initial feature extraction
        layers.Conv3D(32, kernel_size=(3, 3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),
        layers.Dropout(dropout_rate),
        
        # Block 2: Deeper features
        layers.Conv3D(64, kernel_size=(3, 3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),
        layers.Dropout(dropout_rate),
        
        # Block 3: High-level features
        layers.Conv3D(128, kernel_size=(3, 3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),
        layers.Dropout(dropout_rate),
        
        # Global pooling instead of flatten to reduce parameters
        layers.GlobalAveragePooling3D(),
        
        # Dense layers
        layers.Dense(256, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_rate),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

# Build model
model = build_3d_cnn(input_shape=(28, 28, 28, 1))
model.summary()

# ============================================================================
# SECTION 5: MODEL COMPILATION
# ============================================================================

print("\n" + "="*70)
print("COMPILING MODEL")
print("="*70)

"""
Loss Function: Binary Crossentropy
- Standard for binary classification
- Could consider weighted loss due to class imbalance

Optimizer: Adam
- Adaptive learning rate
- Good default choice

Metrics: 
- Accuracy: Overall correctness
- AUC: Important for imbalanced datasets, measures ranking quality
"""

# Calculate class weights to handle imbalance
class_weight = {
    0: len(y_train_flat) / (2 * counts_train[0]),
    1: len(y_train_flat) / (2 * counts_train[1])
}
print(f"Class weights: {class_weight}")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

# ============================================================================
# SECTION 6: TRAINING SETUP
# ============================================================================

print("\n" + "="*70)
print("TRAINING CONFIGURATION")
print("="*70)

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_auc',
    patience=10,
    restore_best_weights=True,
    mode='max',
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Model checkpoint
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_vessel_model.h5',
    monitor='val_auc',
    save_best_only=True,
    mode='max',
    verbose=1
)

print("Callbacks configured:")
print("- Early Stopping (patience=10, monitor=val_auc)")
print("- Reduce LR on Plateau (factor=0.5, patience=5)")
print("- Model Checkpoint (save best val_auc)")

# ============================================================================
# SECTION 7: INITIAL TRAINING (TEST RUN)
# ============================================================================

print("\n" + "="*70)
print("RUNNING INITIAL TEST EPOCHS")
print("="*70)
print("Training for 2 epochs to verify architecture is learning...")

# Train for just 2 epochs to verify everything works
history = model.fit(
    x_train_norm, y_train_flat,
    batch_size=16,
    epochs=2,
    validation_data=(x_val_norm, y_val_flat),
    class_weight=class_weight,
    verbose=1
)

print("\nInitial test complete!")
print(f"Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Training AUC: {history.history['auc'][-1]:.4f}")
print(f"Validation Loss: {history.history['val_loss'][-1]:.4f}")
print(f"Validation AUC: {history.history['val_auc'][-1]:.4f}")

# ============================================================================
# SECTION 8: FULL TRAINING (FOR GPU)
# ============================================================================

"""
UNCOMMENT THE FOLLOWING SECTION FOR FULL TRAINING ON GPU

# Full training configuration
epochs = 100  # Will likely stop early due to early stopping

print("\n" + "="*70)
print("STARTING FULL TRAINING")
print("="*70)

history_full = model.fit(
    x_train_norm, y_train_flat,
    batch_size=32,  # Larger batch size for GPU
    epochs=epochs,
    validation_data=(x_val_norm, y_val_flat),
    class_weight=class_weight,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss plot
axes[0].plot(history_full.history['loss'], label='Training Loss')
axes[0].plot(history_full.history['val_loss'], label='Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Model Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

# AUC plot
axes[1].plot(history_full.history['auc'], label='Training AUC')
axes[1].plot(history_full.history['val_auc'], label='Validation AUC')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('AUC')
axes[1].set_title('Model AUC')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# SECTION 9: EVALUATION
# ============================================================================

print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

# Load best model
model = keras.models.load_model('best_vessel_model.h5')

# Evaluate on test set
test_loss, test_acc, test_auc = model.evaluate(x_test_norm, y_test_flat, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# Get predictions
y_pred_proba = model.predict(x_test_norm, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Classification report
print("\nClassification Report:")
print(classification_report(y_test_flat, y_pred, 
                          target_names=['Healthy', 'Aneurysm']))

# Confusion matrix
cm = confusion_matrix(y_test_flat, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Aneurysm'],
            yticklabels=['Healthy', 'Aneurysm'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test_flat, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {test_auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("\n" + "="*70)
print("COMPARISON WITH MEDMNIST BENCHMARKS")
print("="*70)
print("According to the MedMNIST paper:")
print("- ResNet18 (3D) baseline: AUC ~0.920, ACC ~0.890")
print("- Your model's performance will be compared here")
print(f"- Your model AUC: {test_auc:.4f}")
print(f"- Your model ACC: {test_acc:.4f}")
"""

print("\n" + "="*70)
print("MILESTONE 2 COMPLETE")
print("="*70)
print("✓ Dataset loaded and explored")
print("✓ Neural network architecture designed")
print("✓ Model compiled with appropriate loss function")
print("✓ Initial test epochs completed successfully")
print("\nNext Steps:")
print("1. Run full training on GPU system (uncomment Section 8)")
print("2. Evaluate against MedMNIST benchmarks")
print("3. Consider ethical implications for presentation")