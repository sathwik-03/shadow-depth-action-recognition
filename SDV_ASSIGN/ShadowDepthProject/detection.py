import cv2
import mediapipe as mp
import numpy as np

class Detector:
    def __init__(self, max_num_faces=1, max_num_hands=2, min_detection_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence
        )

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process_frame(self, frame):
        """
        Process a single frame to detect face and hands.
        Args:
            frame: Input BGR image.
        Returns:
            dict: Contains 'face_landmarks', 'face_bbox', 'hand_landmarks', 'hand_bbox', 'hand_label'
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Results container
        results_data = {
            'face_landmarks': None,
            'face_bbox': None,
            'face_roi_poly': None,  # Polygon for mask
            'hands': [] # List of {'landmarks': ..., 'bbox': ..., 'label': ...}
        }

        # Face Detection
        face_results = self.face_mesh.process(rgb_frame)
        if face_results.multi_face_landmarks:
            # Assume 1 face for now
            detected_face = face_results.multi_face_landmarks[0]
            results_data['face_landmarks'] = detected_face
            
            # Calculate Bounding Box
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            
            # 2D landmarks for ROI
            face_poly = []
            
            for lm in detected_face.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
                face_poly.append((x, y))
            
            # Clamp to image
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
            
            results_data['face_bbox'] = (x_min, y_min, x_max, y_max)
            results_data['face_roi_poly'] = np.array(face_poly)

        # Hand Detection
        hand_results = self.hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for idx, hand_lms in enumerate(hand_results.multi_hand_landmarks):
                # Calculate Bounding Box
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for lm in hand_lms.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                # Check label (Left/Right)
                label = "Unknown"
                if hand_results.multi_handedness:
                    if idx < len(hand_results.multi_handedness):
                         label = hand_results.multi_handedness[idx].classification[0].label

                results_data['hands'].append({
                    'landmarks': hand_lms,
                    'bbox': (x_min, y_min, x_max, y_max),
                    'label': label
                })

        return results_data
