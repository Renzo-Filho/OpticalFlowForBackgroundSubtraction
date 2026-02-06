import cv2
import numpy as np

class OpticalFlowEngine:
    def __init__(self, scale=0.5, blur_k=(15, 15)):
        self.scale = scale
        self.blur_k = blur_k
        self.prev_gray = None

    def update(self, frame_bgr):
        """
        Processes a new frame and returns the flow vectors.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Initialize if it's the first frame
        if self.prev_gray is None:
            self.prev_gray = gray
            h, w = gray.shape
            return np.zeros((h, w, 2), dtype=np.float32)

        # 1. Downscale for performance
        prev_small = cv2.resize(self.prev_gray, None, fx=self.scale, fy=self.scale)
        curr_small = cv2.resize(gray, None, fx=self.scale, fy=self.scale)

        # 2. Calculate Farneback Flow
        flow_small = cv2.calcOpticalFlowFarneback(
            prev_small, curr_small, None, 
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        # 3. Spatial Smoothing (Blur)
        flow_small = cv2.GaussianBlur(flow_small, self.blur_k, 5.0)

        # 4. Upscale back to original resolution
        h, w = gray.shape
        flow = cv2.resize(flow_small, (w, h))
        flow *= (1.0 / self.scale)

        # Store for next iteration
        self.prev_gray = gray.copy()
        
        return flow