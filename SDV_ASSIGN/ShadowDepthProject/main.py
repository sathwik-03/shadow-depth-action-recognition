import cv2
import time
import numpy as np

from detection import Detector
from shadow_detection import ShadowDetector
from depth_estimation import DepthEstimator
from visualization import Visualizer

def main():
    # 1. Initialize Modules
    detector = Detector()
    shadow_detector = ShadowDetector()
    depth_estimator = DepthEstimator()
    visualizer = Visualizer()
    
    # 2. Setup Camera
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Shadow Depth System...")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        frame = cv2.flip(frame, 1) # Mirror view
        
        # 3. Detection
        det_results = detector.process_frame(frame)
        
        face_bbox = det_results['face_bbox']
        hands_data = det_results['hands']
        
        # Defaults
        shadow_mask = None
        metrics = {}
        depth = 50.0
        action = "WAITING"
        
        # 4. Process Face & Shadow
        if face_bbox:
            x1, y1, x2, y2 = face_bbox
            # Ensure ROI is valid
            if x2 > x1 and y2 > y1:
                face_roi = frame[y1:y2, x1:x2]
                
                # Detect Shadow
                shadow_mask, metrics = shadow_detector.detect_shadow(face_roi)
                
                # Estimate Depth
                # Note: We should ideally correlate with hand presence, 
                # but for now we follow the shadow physics strictly.
                # If a hand is present, we assume the shadow is likely caused by it or ambient.
                depth, action = depth_estimator.estimate_depth(metrics)
                
                # 5. Visualization Overlay (Shadow on Face)
                frame = visualizer.overlay_shadow(frame, face_bbox, shadow_mask)
        
        # 6. Main Overlay
        frame = visualizer.draw_overlay(frame, face_bbox, hands_data, depth, action)
        
        # 7. Heatmap Window
        intensity_drop = metrics.get('intensity_drop', 0)
        heatmap = visualizer.generate_heatmap(intensity_drop)
        
        # Display
        cv2.imshow('Shadow Depth Action Recognition', frame)
        cv2.imshow('Shadow Intensity Heatmap', heatmap)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
