import cv2
import numpy as np
import time

class BackgroundProcessor:
    def __init__(self):
        self.bg_model = None
        self.use_motion_mode = False
        self.flow_acc = None  # Accumulator for robust motion mask
        
    def set_mode(self, use_motion):
        """Toggle between Static BG and Motion Flow masking."""
        self.use_motion_mode = use_motion

    def capture_static_model(self, cap, num_frames=60):
        """
        Captures the 'empty' room. 
        Averages frames to reduce sensor noise.
        """
        print("Capturing background... stay out of frame!")
        acc = None
        count = 0
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret: continue
            
            frame = cv2.flip(frame, 1)
            f_data = frame.astype(np.float32)
            acc = f_data if acc is None else acc + f_data
            count += 1
            
        if acc is not None:
            self.bg_model = (acc / count).astype(np.uint8)
            return True
        return False

    def get_mask(self, frame, flow):
        """Dispatches to the correct masking logic based on current mode."""
        if self.use_motion_mode:
            return self._mask_from_flow(flow)
        else:
            return self._mask_from_static(frame)

    def _mask_from_static(self, frame):
        """The original YCrCb difference logic."""
        if self.bg_model is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        # Difference in YCrCb space for stability
        f_ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        b_ycc = cv2.cvtColor(self.bg_model, cv2.COLOR_BGR2YCrCb)
        diff = cv2.absdiff(f_ycc, b_ycc)

        # Weighted score (Cr and Cb usually hold more 'person' info)
        score = 0.5 * diff[...,0] + 1.0 * diff[...,1] + 1.0 * diff[...,2]
        score_u8 = np.clip(score, 0, 255).astype(np.uint8)

        # Automatic Thresholding
        _, mask = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self._post_process(mask)

    def _mask_from_flow(self, flow, threshold=2.0, decay=0.90):
        """Optical Flow based masking with temporal 'memory'."""
        h, w = flow.shape[:2]
        if self.flow_acc is None or self.flow_acc.shape != (h, w):
            self.flow_acc = np.zeros((h, w), dtype=np.float32)

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Keep mask alive briefly after motion stops (Decay)
        self.flow_acc = np.maximum(mag, self.flow_acc * decay)
        
        _, mask = cv2.threshold(self.flow_acc, threshold, 255, cv2.THRESH_BINARY)
        return self._post_process(mask.astype(np.uint8))

    def _post_process(self, mask):
        """Morphology to clean up noise and fill holes."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return cv2.GaussianBlur(mask, (11, 11), 0)