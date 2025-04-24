import cv2
import numpy as np
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameAugmenter:
    """Frame augmentation class for video frames."""
    
    def __init__(self, seed=None):
        """Initialize augmenter with optional random seed."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
    def random_brightness_contrast(self, image, brightness_limit=0.2, contrast_limit=0.2, p=0.7):
        """
        Apply random brightness and contrast adjustments.
        
        Args:
            image: Input image
            brightness_limit: Maximum brightness adjustment (±)
            contrast_limit: Maximum contrast adjustment (±)
            p: Probability of applying the transform
            
        Returns:
            Augmented image
        """
        if random.random() > p:
            return image
        
        # Brightness adjustment
        if random.random() < 0.5:
            alpha = 1.0 + random.uniform(-brightness_limit, brightness_limit)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        
        # Contrast adjustment
        if random.random() < 0.5:
            alpha = 1.0 + random.uniform(-contrast_limit, contrast_limit)
            mean = np.mean(image, axis=(0, 1), keepdims=True)
            image = cv2.convertScaleAbs((image - mean) * alpha + mean)
            
        return image
    
    def random_rotate(self, image, rotate_limit=10, p=0.5):
        """
        Apply random rotation within limits.
        
        Args:
            image: Input image
            rotate_limit: Maximum rotation in degrees
            p: Probability of applying the transform
            
        Returns:
            Augmented image
        """
        if random.random() > p:
            return image
        
        h, w = image.shape[:2]
        angle = random.uniform(-rotate_limit, rotate_limit)
        
        # Get rotation matrix
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return rotated
    
    def random_crop_and_pad(self, image, crop_height=400, crop_width=600, p=0.3):
        """
        Apply random crop followed by padding to original size.
        
        Args:
            image: Input image
            crop_height: Height after cropping
            crop_width: Width after cropping
            p: Probability of applying the transform
            
        Returns:
            Augmented image
        """
        if random.random() > p:
            return image
        
        h, w = image.shape[:2]
        
        # Don't crop if image is too small
        if h <= crop_height or w <= crop_width:
            return image
        
        # Random crop coordinates
        top = random.randint(0, h - crop_height)
        left = random.randint(0, w - crop_width)
        
        # Crop image
        cropped = image[top:top+crop_height, left:left+crop_width]
        
        # Pad back to original size
        padded = np.zeros_like(image)
        pad_top = (h - crop_height) // 2
        pad_left = (w - crop_width) // 2
        padded[pad_top:pad_top+crop_height, pad_left:pad_left+crop_width] = cropped
        
        return padded
    
    def augment(self, image):
        """
        Apply all augmentations with probability.
        
        Args:
            image: Input image
            
        Returns:
            Augmented image
        """
        image = self.random_brightness_contrast(image)
        image = self.random_rotate(image)
        image = self.random_crop_and_pad(image)
        return image
    
    def augment_frames(self, frames, augment_probability=0.5):
        """
        Apply augmentations to a sequence of frames.
        
        Args:
            frames: List of video frames
            augment_probability: Probability of augmenting each individual frame
            
        Returns:
            List of augmented frames
        """
        augmented_frames = []
        
        for frame in frames:
            # Apply augmentation with probability
            if random.random() < augment_probability:
                augmented_frames.append(self.augment(frame))
            else:
                augmented_frames.append(frame)
                
        return augmented_frames