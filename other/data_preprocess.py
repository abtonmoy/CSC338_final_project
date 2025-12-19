import numpy as np
from scipy.ndimage import rotate, zoom, shift
import tensorflow as tf

class Augment3D:

    def elastic_transform_3d(volume, alpha=10, sigma=3):
        """Apply elastic deformation to 3D volume."""
        # Only apply to spatial dimensions (not channel)
        shape = volume.shape[1:]  # Spatial dimensions
        
        # Generate random displacement fields
        dx = np.random.randn(*shape) * sigma
        dy = np.random.randn(*shape) * sigma
        dz = np.random.randn(*shape) * sigma
        
        # Smooth the displacement fields
        from scipy.ndimage import gaussian_filter
        dx = gaussian_filter(dx, sigma, mode='constant') * alpha
        dy = gaussian_filter(dy, sigma, mode='constant') * alpha
        dz = gaussian_filter(dz, sigma, mode='constant') * alpha
        
        # Create meshgrid
        z, y, x = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing='ij'
        )
        
        indices = [
            np.clip(z + dz, 0, shape[0] - 1).astype(int),
            np.clip(y + dy, 0, shape[1] - 1).astype(int),
            np.clip(x + dx, 0, shape[2] - 1).astype(int)
        ]
        
        # Apply transformation to each channel
        result = np.zeros_like(volume)
        for c in range(volume.shape[0]):
            result[c] = volume[c][indices[0], indices[1], indices[2]]
        
        return result

    def resize_to_original(volume, target_shape):
        """Resize volume to match target shape by cropping or padding."""
        current_shape = volume.shape
        result = volume.copy()
        
        for i in range(len(target_shape)):
            if current_shape[i] > target_shape[i]:
                # Crop
                diff = current_shape[i] - target_shape[i]
                start = diff // 2
                end = start + target_shape[i]
                result = np.take(result, range(start, end), axis=i)
            elif current_shape[i] < target_shape[i]:
                # Pad
                diff = target_shape[i] - current_shape[i]
                pad_before = diff // 2
                pad_after = diff - pad_before
                pad_width = [(0, 0)] * len(target_shape)
                pad_width[i] = (pad_before, pad_after)
                result = np.pad(result, pad_width, mode='edge')
        
        return result
        
    def augment_3d_volume(volume, num_augmentations=5):
        """
        Apply various 3D augmentations to a volume.
        
        Args:
            volume: numpy array with shape (C, D, H, W) where C is channel dimension
            num_augmentations: number of augmented copies to generate
        
        Returns:
            List of augmented volumes
        """
        augmented_volumes = []
        
        # Convert to numpy if it's a tensor
        if hasattr(volume, 'numpy'):
            volume = volume.numpy()
        
        original_shape = volume.shape
        
        for _ in range(num_augmentations):
            aug_volume = volume.copy()
            
            # Random rotation around spatial axes
            # Shape is (1, 28, 28, 28) so channel is first
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-15, 15)
                # Choose axes for rotation (avoid channel dimension 0)
                axes_options = [(1, 2), (1, 3), (2, 3)]
                axes = axes_options[np.random.randint(0, len(axes_options))]
                aug_volume = rotate(aug_volume, angle, axes=axes, reshape=False, mode='nearest')
            
            # Random scaling/zoom
            if np.random.rand() > 0.5:
                scale_factor = np.random.uniform(0.9, 1.1)
                # Apply scale to spatial dimensions only (not channel)
                zoom_factors = [1, scale_factor, scale_factor, scale_factor]
                aug_volume = zoom(aug_volume, zoom_factors, mode='nearest')
                aug_volume = resize_to_original(aug_volume, original_shape)
            
            # Random flip along spatial axes
            if np.random.rand() > 0.5:
                axis = np.random.randint(1, 4)  # Flip along axis 1, 2, or 3 (not channel 0)
                aug_volume = np.flip(aug_volume, axis=axis)
            
            # Random shift along spatial dimensions
            if np.random.rand() > 0.5:
                # No shift for channel dimension, random shift for spatial
                shift_amount = [0] + [np.random.randint(-3, 4) for _ in range(3)]
                aug_volume = shift(aug_volume, shift_amount, mode='nearest')
            
            # Add slight Gaussian noise
            if np.random.rand() > 0.5:
                noise = np.random.normal(0, 0.01, aug_volume.shape)
                aug_volume = aug_volume + noise
                aug_volume = np.clip(aug_volume, 0, 1)
            
            # Random brightness adjustment
            if np.random.rand() > 0.5:
                brightness_factor = np.random.uniform(0.9, 1.1)
                aug_volume = aug_volume * brightness_factor
                aug_volume = np.clip(aug_volume, 0, 1)
            
            # Random contrast adjustment
            if np.random.rand() > 0.5:
                mean = np.mean(aug_volume)
                aug_volume = (aug_volume - mean) * np.random.uniform(0.9, 1.1) + mean
                aug_volume = np.clip(aug_volume, 0, 1)
            
            # Random elastic deformation (subtle)
            if np.random.rand() > 0.7:
                aug_volume = elastic_transform_3d(aug_volume, alpha=5, sigma=3)
            
            augmented_volumes.append(aug_volume)
        
        return augmented_volumes

    

