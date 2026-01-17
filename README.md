# Physics-Based Shadow Depth Action Recognition

This project implements a computer vision system that estimates the depth between a hand and a face using a single RGB camera. It leverages **shadow physics** (inverse square law approximation) to infer proximity without external depth sensors.

## 🚀 How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Application**:
    ```bash
    python main.py
    ```

3.  **Usage**:
    - Ensure your face is clearly visible.
    - Ensure there is a dominant light source (e.g., ceiling light) above you.
    - Bring your hand closer to your face.
    - Observe the **Shadow Overlay** (green/red on face) and the **Estimated Depth**.
    - When depth drops below 5cm, the action "TOUCHING FACE / EATING" is triggered.
    - Press `q` to quit.

## 🧠 Physics & Logic

### 1. System Pipeline
1.  **Detection**: MediaPipe Face Mesh & Hands detect face ROI and hand bounding boxes.
2.  **Shadow Detection**: The Face ROI is converted to **LAB color space**. We analyze the **L-channel (Lightness)** using adaptive thresholding to find dark regions on the skin (filtered by a skin color mask).
3.  **Depth Estimation**: Physics-based heuristics are applied to the shadow metrics.
4.  **Action Recognition**: A threshold-based logic determines if the hand is effectively "touching" the face.

### 2. Physics Intuition
The system is inspired by the **Inverse Square Law** of light intensity:
$$ I \propto \frac{1}{d^2} $$
As an occluding object (hand) gets closer to a surface (face), the cast shadow becomes:
1.  **Darker** (higher Intensity Drop).
2.  **Sharper** (harder edges).

We approximate depth ($Z$) using the **Intensity Drop**:
$$ Z \approx \frac{k}{\text{Intensity Drop} + \epsilon} $$
where $k$ is a calibration constant.

- **High Intensity Drop** (Very dark shadow) $\rightarrow$ Object is **CLOSE**.
- **Low Intensity Drop** (Faint/No shadow) $\rightarrow$ Object is **FAR**.

## 📂 Project Structure
- `main.py`: Entry point, main loop.
- `detection.py`: MediaPipe wrappers.
- `shadow_detection.py`: LAB/HSV image processing.
- `depth_estimation.py`: Physics logic.
- `visualization.py`: Drawing overlays and heatmaps.
- `utils.py`: Helper functions.
