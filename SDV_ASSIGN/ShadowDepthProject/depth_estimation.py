import numpy as np

class DepthEstimator:
    def __init__(self, calibration_k=10.0, touch_threshold_cm=5.0):
        self.k = calibration_k
        self.touch_threshold = touch_threshold_cm
        
        # Smoothing buffers
        self.depth_history = []
        self.max_history = 5
    
    def estimate_depth(self, shadow_metrics):
        """
        Estimate depth based on shadow metrics using Inverse Square Law approximation.
        
        Physics:
        Intensity Drop (I_drop) is proportional to Occlusion, which is inversely proportional to Distance^2?
        Actually, simpler approximation:
        As object gets closer, shadow gets darker (Intensity Drop increases) and Area might increase/stabilize.
        
        Let's model: Depth ~ 1 / Intensity_Drop
        
        Args:
            shadow_metrics: dict with 'intensity_drop', 'area', etc.
        
        Returns:
            depth_cm: Estimated depth in cm.
            action_label: "Touching Face" or "Hand Away"
        """
        
        intensity_drop = shadow_metrics.get('intensity_drop', 0)
        area = shadow_metrics.get('area', 0)
        
        if intensity_drop <= 0.05:
            # Very weak shadow -> Far away or no shadow
            depth_cm = 50.0 # Max depth assumption
        else:
            # Physics-based heuristic
            # d \propto 1 / intensity_drop
            # Add epsilon to prevent div by zero
            raw_val = 1.0 / (intensity_drop + 0.1)
            
            # Map raw_val to cm
            # If drop=0.8 (strong shadow), raw=1.1 -> Depth should be ~0cm
            # If drop=0.1 (weak), raw=10 -> Depth should be ~30-50cm
            
            # Linear mapping from raw_val to cm
            # We want: 
            # drop 0.8 -> depth 0
            # drop 0.0 -> depth 50
            
            # Let's derive a simpler interpolation
            # depth = 50 * (1 - intensity_drop) ^ 2
            # Squared falloff (Inverse Square Law intuition reversed for distance)
            
            depth_cm = 50.0 * ((1.0 - min(intensity_drop, 1.0)) ** 2)
            
        
        # Smoothing
        self.depth_history.append(depth_cm)
        if len(self.depth_history) > self.max_history:
            self.depth_history.pop(0)
            
        avg_depth = sum(self.depth_history) / len(self.depth_history)
        
        # Action Classification
        if avg_depth < self.touch_threshold:
            action = "TOUCHING FACE / EATING"
        else:
            action = "HAND AWAY"
            
        return avg_depth, action
