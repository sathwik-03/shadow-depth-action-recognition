import cv2
import numpy as np

class ShadowDetector:
    def __init__(self):
        # LAB parameters
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def get_skin_mask(self, img_bgr):
        """
        Create a binary mask for skin regions using HSV.
        """
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Typical skin color range in HSV
        # Lower: [0, 20, 70], Upper: [20, 255, 255]
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Morphological clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def detect_shadow(self, face_roi):
        """
        Detects shadow in the Face ROI.
        Args:
            face_roi: BGR image of the face.
        Returns:
            shadow_mask: Binary mask (255=shadow, 0=non-shadow)
            metrics: dict containing 'area', 'mean_intensity_drop', 'shadow_intensity', 'bg_intensity'
        """
        if face_roi is None or face_roi.size == 0:
            return None, {}

        # 1. Convert to LAB and extract L channel
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        
        # 2. Preprocessing
        # Blur to reduce noise
        l_blurred = cv2.GaussianBlur(l_channel, (5, 5), 0)
        
        # 3. Shadow Detection Strategy
        # Shadows are darker regions. We can use Otsu or Adaptive Thresholding.
        # Check if we assume 'shadow' is significantly darker than 'skin'.
        
        # Method A: Otsu on inverted L
        # Invert so dark becomes bright -> Otsu finds the bright (dark) regions
        l_inv = cv2.bitwise_not(l_blurred)
        _, raw_thresh = cv2.threshold(l_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method B: Adaptive (good for local shadows)
        # thresh_adaptive = cv2.adaptiveThreshold(l_inv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Let's use Otsu as it's more stable for distinct shadows vs skin.
        shadow_candidates = raw_thresh
        
        # 4. Refine with Skin Mask
        # We only care about shadows ON skin, not background/hair (rough approx)
        skin_mask = self.get_skin_mask(face_roi)
        shadow_on_skin = cv2.bitwise_and(shadow_candidates, shadow_candidates, mask=skin_mask)
        
        # 5. Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # Remove small noise
        shadow_mask = cv2.morphologyEx(shadow_on_skin, cv2.MORPH_OPEN, kernel, iterations=1)
        # Close gaps
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 6. Compute Metrics
        # Area
        area = cv2.countNonZero(shadow_mask)
        
        # Intensity Stats
        # Mean L of Shadow
        mean_shadow = cv2.mean(l_channel, mask=shadow_mask)[0]
        
        # Mean L of Non-Shadow (Background skin)
        # Use skin mask but remove shadow area
        bg_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(shadow_mask))
        mean_bg = cv2.mean(l_channel, mask=bg_mask)[0]
        
        # Avoid division by zero
        if mean_bg == 0: mean_bg = 0.001
        
        intensity_drop = (mean_bg - mean_shadow) / mean_bg
        
        metrics = {
            'area': area,
            'mean_shadow': mean_shadow,
            'mean_bg': mean_bg,
            'intensity_drop': intensity_drop
        }
        
        return shadow_mask, metrics
