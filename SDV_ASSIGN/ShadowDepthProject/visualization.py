import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        pass

    def draw_overlay(self, frame, face_bbox, hands_data, depth, action_label):
        """
        Draws bounding boxes, text, and other overlays on the main frame.
        """
        out_frame = frame.copy()
        
        # Draw Face
        if face_bbox:
            x1, y1, x2, y2 = face_bbox
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out_frame, "FACE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw Hands
        for hand in hands_data:
            hx1, hy1, hx2, hy2 = hand['bbox']
            label = hand['label']
            cv2.rectangle(out_frame, (hx1, hy1), (hx2, hy2), (255, 0, 0), 2)
            cv2.putText(out_frame, f"HAND: {label}", (hx1, hy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
        # Draw Info Panel
        h, w, _ = out_frame.shape
        panel_h = 100
        cv2.rectangle(out_frame, (0, h - panel_h), (w, h), (0, 0, 0), -1)
        
        # Depth Text
        depth_color = (0, 255, 255) if action_label == "HAND AWAY" else (0, 0, 255)
        cv2.putText(out_frame, f"Est. Depth: {depth:.2f} cm", (20, h - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, depth_color, 2)
        
        # Action Text
        cv2.putText(out_frame, f"Action: {action_label}", (20, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, depth_color, 2)
        
        return out_frame

    def overlay_shadow(self, frame, face_bbox, shadow_mask):
        """
        Overlays the shadow mask in GREEN/RED on the face area.
        """
        if face_bbox is None or shadow_mask is None:
            return frame
            
        x1, y1, x2, y2 = face_bbox
        # ROI from frame
        roi = frame[y1:y2, x1:x2]
        
        # Resize mask to fit ROI if needed (should match if processing correct ROI)
        if shadow_mask.shape[:2] != roi.shape[:2]:
            shadow_mask = cv2.resize(shadow_mask, (roi.shape[1], roi.shape[0]))
            
        # Color the shadow region red
        colored_shadow = np.zeros_like(roi)
        colored_shadow[:] = (0, 0, 255) # Red
        
        # Blend - SAFE METHOD
        # Create blended version of full ROI
        blended_roi = cv2.addWeighted(roi, 0.7, colored_shadow, 0.3, 0)
        
        # Apply blended pixels ONLY where mask is set
        mask_indices = shadow_mask > 0
        roi[mask_indices] = blended_roi[mask_indices]
        
        return frame

    def generate_heatmap(self, intensity_drop):
        """
        Generates a dummy heatmap visualization of scalar intensity drop.
        Real spatial heatmap would require diffing pixel-wise.
        This provides a visual indicator bar.
        """
        height, width = 200, 300
        heatmap = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Gradient
        # Map intensity drop (0.0 - 1.0) to color
        val = int(intensity_drop * 255)
        val = max(0, min(255, val))
        
        # Bar chart style
        bar_width = int(width * intensity_drop)
        cv2.rectangle(heatmap, (0, 50), (bar_width, 150), (0, val, 255-val), -1)
        
        cv2.putText(heatmap, f"Shadow Intensity: {intensity_drop:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        return heatmap
