"""
VesselMNIST3D Test Training Notebook - WITH CUSTOM 3D AUGMENTATION
Complete implementation with working 3D data augmentation
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
from medmnist import VesselMNIST3D

# ============================================================================
# CUSTOM 3D AUGMENTATION LAYER
# ============================================================================

class Augment3D(layers.Layer):
    """Custom 3D data augmentation layer for volumetric medical images"""
    
    def __init__(self, 
                 flip_prob=0.5,
                 rotate_prob=0.3,
                 brightness_delta=0.1,
                 contrast_range=(0.9, 1.1),
                 **kwargs):
        super().__init__(**kwargs)
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
    
    def call(self, images, training=None):
        # Default to False if training is not specified
        if training is None:
            training = False
        
        # Only augment during training
        if not training:
            return images
        
        x = images
        
        # Random flips along different axes (depth, height, width)
        # Each flip is independent with flip_prob
        x = tf.cond(
            tf.random.uniform([]) < self.flip_prob,
            lambda: tf.reverse(x, axis=[1]),  # Flip along depth
            lambda: x
        )
        
        x = tf.cond(
            tf.random.uniform([]) < self.flip_prob,
            lambda: tf.reverse(x, axis=[2]),  # Flip along height
            lambda: x
        )
        
        x = tf.cond(
            tf.random.uniform([]) < self.flip_prob,
            lambda: tf.reverse(x, axis=[3]),  # Flip along width
            lambda: x
        )
        
        # Random 90-degree rotations in the XY plane (around Z axis)
        # Use tf.switch_case for multiple rotation options
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        
        def rotate_0():
            return x
        
        def rotate_90():
            rotated = tf.transpose(x, [0, 1, 3, 2, 4])  # Swap H and W
            return tf.reverse(rotated, axis=[2])  # Flip to complete rotation
        
        def rotate_180():
            return tf.reverse(x, axis=[2, 3])
        
        def rotate_270():
            rotated = tf.transpose(x, [0, 1, 3, 2, 4])  # Swap H and W
            return tf.reverse(rotated, axis=[3])  # Flip to complete rotation
        
        x = tf.cond(
            tf.random.uniform([]) < self.rotate_prob,
            lambda: tf.switch_case(k, {0: rotate_0, 1: rotate_90, 2: rotate_180, 3: rotate_270}),
            lambda: x
        )
        
        # Random brightness adjustment
        delta = tf.random.uniform([], -self.brightness_delta, self.brightness_delta)
        x = tf.cond(
            tf.random.uniform([]) < 0.5,
            lambda: x + delta,
            lambda: x
        )
        
        # Random contrast adjustment
        factor = tf.random.uniform([], self.contrast_range[0], self.contrast_range[1])
        mean = tf.reduce_mean(x, axis=[1, 2, 3, 4], keepdims=True)
        x = tf.cond(
            tf.random.uniform([]) < 0.5,
            lambda: (x - mean) * factor + mean,
            lambda: x
        )
        
        # Clip values to valid range [0, 1]
        x = tf.clip_by_value(x, 0.0, 1.0)
        
        return x
    
    def compute_output_shape(self, input_shape):
        """Output shape is same as input shape"""
        return input_shape
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'flip_prob': self.flip_prob,
            'rotate_prob': self.rotate_prob,
            'brightness_delta': self.brightness_delta,
            'contrast_range': self.contrast_range,
        })
        return config

# ============================================================================
# PART 1: BUILDING BLOCKS
# ============================================================================

def inception_block_3d(x, filters):
    """3D Inception module with multiple kernel sizes"""
    branch1 = layers.Conv3D(filters, (1, 1, 1), padding='same', activation='relu')(x)
    
    branch2 = layers.Conv3D(filters, (1, 1, 1), padding='same', activation='relu')(x)
    branch2 = layers.Conv3D(filters, (3, 3, 3), padding='same', activation='relu')(branch2)
    
    branch3 = layers.Conv3D(filters, (1, 1, 1), padding='same', activation='relu')(x)
    branch3 = layers.Conv3D(filters, (3, 3, 3), padding='same', activation='relu')(branch3)
    branch3 = layers.Conv3D(filters, (3, 3, 3), padding='same', activation='relu')(branch3)
    
    branch4 = layers.MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch4 = layers.Conv3D(filters, (1, 1, 1), padding='same', activation='relu')(branch4)
    
    output = layers.Concatenate()([branch1, branch2, branch3, branch4])
    return output

def residual_block_3d(x, filters):
    """3D Residual block with skip connection"""
    shortcut = x
    
    x = layers.Conv3D(filters, (3, 3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters, (3, 3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv3D(filters, (1, 1, 1), padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def squeeze_excitation_block_3d(x, ratio=16):
    """3D Squeeze-and-Excitation block for channel attention"""
    channels = x.shape[-1]
    
    se = layers.GlobalAveragePooling3D()(x)
    se = layers.Dense(channels // ratio, activation='relu')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, 1, channels))(se)
    
    return layers.Multiply()([x, se])

def MyNet3D(num_classes=2):
    """Full 3D Network Architecture with Custom 3D Augmentation"""
    inputs = layers.Input(shape=(28, 28, 28, 1))
    
    # Custom 3D data augmentation (only active during training)
    x = Augment3D(
        flip_prob=0.5,
        rotate_prob=0.3,
        brightness_delta=0.1,
        contrast_range=(0.9, 1.1)
    )(inputs)
    
    # Initial feature extraction
    x = layers.Conv3D(16, (3, 3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Stage 1: 28x28x28
    inception1 = inception_block_3d(x, 8)
    inception1 = layers.BatchNormalization()(inception1)
    inception1 = layers.Dropout(0.2)(inception1)
    
    residual1 = residual_block_3d(inception1, 32)
    se1 = squeeze_excitation_block_3d(residual1)
    x = layers.MaxPooling3D((2, 2, 2))(se1)
    
    # Stage 2: 14x14x14
    inception2 = inception_block_3d(x, 12)
    inception2 = layers.BatchNormalization()(inception2)
    inception2 = layers.Dropout(0.3)(inception2)
    
    residual2 = residual_block_3d(inception2, 48)
    se2 = squeeze_excitation_block_3d(residual2)
    
    # Dense connection
    se1_pooled = layers.MaxPooling3D((2, 2, 2))(se1)
    se1_adjusted = layers.Conv3D(48, (1, 1, 1), padding='same')(se1_pooled)
    dense_concat1 = layers.Add()([se2, se1_adjusted])
    x = layers.MaxPooling3D((2, 2, 2))(dense_concat1)
    
    # Stage 3: 7x7x7
    residual3a = residual_block_3d(x, 64)
    residual3a = layers.Dropout(0.35)(residual3a)
    
    residual3b = residual_block_3d(residual3a, 64)
    se3 = squeeze_excitation_block_3d(residual3b)
    
    # Dual attention
    spatial_attention = layers.Conv3D(1, (7, 7, 7), padding='same', activation='sigmoid')(se3)
    spatial_features = layers.Multiply()([se3, spatial_attention])
    
    channel_features = layers.Conv3D(64, (1, 1, 1), activation='relu')(se3)
    
    x = layers.Concatenate()([spatial_features, channel_features])
    x = layers.Conv3D(128, (1, 1, 1), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Conv3D(96, (3, 3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Global pooling
    gap = layers.GlobalAveragePooling3D()(x)
    gmp = layers.GlobalMaxPooling3D()(x)
    x = layers.Concatenate()([gap, gmp])
    
    # Classification head
    x = layers.Dense(192, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(96, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================================================================
# PART 2: DATA LOADING
# ============================================================================

def load_vesselmnist3d():
    """
    Load VesselMNIST3D dataset
    Returns: (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)
    """
    print("Loading VesselMNIST3D dataset...")
    
    # Download and load the dataset
    train_dataset = VesselMNIST3D(split='train', download=True)
    val_dataset = VesselMNIST3D(split='val', download=True)
    test_dataset = VesselMNIST3D(split='test', download=True)
    
    # Extract images and labels
    train_images = train_dataset.imgs
    train_labels = train_dataset.labels.squeeze()
    
    val_images = val_dataset.imgs
    val_labels = val_dataset.labels.squeeze()
    
    test_images = test_dataset.imgs
    test_labels = test_dataset.labels.squeeze()
    
    # Normalize to [0, 1]
    train_images = train_images.astype('float32') / 255.0
    val_images = val_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    
    # Add channel dimension if needed
    if len(train_images.shape) == 4:
        train_images = np.expand_dims(train_images, axis=-1)
        val_images = np.expand_dims(val_images, axis=-1)
        test_images = np.expand_dims(test_images, axis=-1)
    
    print(f"Train: {train_images.shape}, Labels: {train_labels.shape}")
    print(f"Val: {val_images.shape}, Labels: {val_labels.shape}")
    print(f"Test: {test_images.shape}, Labels: {test_labels.shape}")
    print(f"Number of classes: {len(np.unique(train_labels))}")
    print(f"Class distribution: {np.bincount(train_labels)}")
    
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)

# ============================================================================
# PART 3: VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_3d_slice(volume, title="3D Volume Slices"):
    """
    Visualize a 3D volume by showing slices along each axis
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Middle slices along each axis
    mid_x = volume.shape[0] // 2
    mid_y = volume.shape[1] // 2
    mid_z = volume.shape[2] // 2
    
    axes[0].imshow(volume[mid_x, :, :], cmap='gray')
    axes[0].set_title(f'X-axis slice (at x={mid_x})')
    axes[0].axis('off')
    
    axes[1].imshow(volume[:, mid_y, :], cmap='gray')
    axes[1].set_title(f'Y-axis slice (at y={mid_y})')
    axes[1].axis('off')
    
    axes[2].imshow(volume[:, :, mid_z], cmap='gray')
    axes[2].set_title(f'Z-axis slice (at z={mid_z})')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def visualize_augmentation_effects(model, sample_image, num_augmentations=6):
    """
    Visualize the effect of data augmentation on a single sample
    """
    # Get the augmentation layer
    aug_layer = None
    for layer in model.layers:
        if isinstance(layer, Augment3D):
            aug_layer = layer
            break
    
    if aug_layer is None:
        print("No augmentation layer found in model")
        return
    
    # Generate augmented versions
    sample_batch = np.expand_dims(sample_image, axis=0)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(num_augmentations):
        # Apply augmentation
        augmented = aug_layer(sample_batch, training=True)
        augmented_np = augmented.numpy()[0, :, :, :, 0]
        
        # Show middle slice
        mid_z = augmented_np.shape[2] // 2
        axes[i].imshow(augmented_np[:, :, mid_z], cmap='gray')
        axes[i].set_title(f'Augmented Version {i+1}')
        axes[i].axis('off')
    
    plt.suptitle('Data Augmentation Examples (Middle Z-slice)', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# PART 4: TEST TRAINING (QUICK EPOCHS)
# ============================================================================

def quick_test_training():
    """
    Run a quick test with just a few epochs on CPU/local hardware
    This is to verify the architecture works before running on GPU
    """
    print("="*70)
    print("QUICK TEST TRAINING - 2 EPOCHS (WITH 3D AUGMENTATION)")
    print("="*70)
    
    # Load data
    (train_images, train_labels), (val_images, val_labels), _ = load_vesselmnist3d()
    
    # Use only a small subset for quick testing
    print("\nUsing small subset for quick testing...")
    train_subset = train_images[:100]
    train_labels_subset = train_labels[:100]
    val_subset = val_images[:50]
    val_labels_subset = val_labels[:50]
    
    print(f"Training on {len(train_subset)} samples")
    print(f"Validating on {len(val_subset)} samples")
    
    # Visualize a few samples
    print("\nVisualizing sample data...")
    for i in range(2):
        visualize_3d_slice(
            train_subset[i, :, :, :, 0], 
            title=f"Training Sample {i} - Label: {train_labels_subset[i]}"
        )
    
    # Build model
    print("\nBuilding model...")
    num_classes = len(np.unique(train_labels))
    model = MyNet3D(num_classes=num_classes)
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Count parameters
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    print(f"\nTotal trainable parameters: {trainable_params:,}")
    
    # Visualize augmentation effects
    print("\nVisualizing data augmentation effects...")
    visualize_augmentation_effects(model, train_subset[0])
    
    # Train for 2 epochs
    print("\n" + "="*70)
    print("Starting training for 2 test epochs...")
    print("="*70)
    
    history = model.fit(
        train_subset, train_labels_subset,
        validation_data=(val_subset, val_labels_subset),
        epochs=2,
        batch_size=8,  # Small batch size for testing
        verbose=1
    )
    
    # Plot results
    print("\nPlotting training history...")
    plot_training_history(history)
    
    # Evaluate on validation set
    print("\n" + "="*70)
    print("Evaluating on validation subset...")
    print("="*70)
    val_loss, val_acc = model.evaluate(val_subset, val_labels_subset, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    # Make predictions on a few samples
    print("\nMaking predictions on 5 validation samples...")
    predictions = model.predict(val_subset[:5])
    
    for i in range(5):
        pred_class = np.argmax(predictions[i])
        true_class = val_labels_subset[i]
        confidence = predictions[i][pred_class]
        print(f"Sample {i}: True={true_class}, Predicted={pred_class}, Confidence={confidence:.4f}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. If training worked, save this notebook")
    print("2. Prepare for full training on GPU")
    print("3. Use more epochs (50-100) and full dataset")
    print("4. Add callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint")
    
    return model, history

# ============================================================================
# PART 5: FULL TRAINING SETUP (For GPU)
# ============================================================================

def full_training_setup():
    """
    Setup for full training on GPU - run this after quick test works
    """
    print("="*70)
    print("FULL TRAINING SETUP (WITH 3D AUGMENTATION)")
    print("="*70)
    
    # Load full dataset
    (train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_vesselmnist3d()
    
    # Build model
    num_classes = len(np.unique(train_labels))
    model = MyNet3D(num_classes=num_classes)
    
    # Setup callbacks
    callbacks = [
        # Reduce learning rate when validation loss plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Stop training if no improvement
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            'best_vesselmnist3d_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]
    
    print("\nTraining configuration:")
    print(f"- Training samples: {len(train_images)}")
    print(f"- Validation samples: {len(val_images)}")
    print(f"- Test samples: {len(test_images)}")
    print(f"- Number of classes: {num_classes}")
    print(f"- Batch size: 32")
    print(f"- Max epochs: 100 (with early stopping)")
    print(f"- Augmentation: ENABLED (3D custom augmentation)")
    
    # Train
    print("\nStarting full training...")
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Choose which to run:
    
    # Option 1: Quick test (run this first on your local machine)
    print("Running quick test training with 3D augmentation...")
    model, history = quick_test_training()
    
    # Option 2: Full training (uncomment when ready to run on GPU)
    # print("Running full training...")
    # model, history = full_training_setup()