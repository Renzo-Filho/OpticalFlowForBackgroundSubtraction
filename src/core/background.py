import cv2
import numpy as np
import time

class BackgroundProcessor:
    def __init__(self):
        self.bg_base = None       # Static Background Model
        self.flow_acc = None      # Motion Accumulator
        self.use_flow_mask = False # Toggle: False=Static, True=Flow
        
        # Ported Parameters from your script
        self.BIG_K_SIZE = 31
        self.SMALL_K_SIZE = 11
        self.NOISE_THRESHOLD = 2.0
        self.DECAY = 0.90

    def set_mode(self, use_flow):
        """Switches between Static BG and Optical Flow masking."""
        self.use_flow_mask = use_flow

    def capture_static_model(self, cap, num_frames=100):
        """
        Ported from capture_background_average.
        Averages frames to create a clean background model.
        """
        print("Capturing background... Please stay out of frame.")
        acc = None
        count = 0
        
        # Settle time (1 second)
        time.sleep(1.0)
        
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1) # Consistency with pipeline
            f_f = frame.astype(np.float32)
            acc = f_f if acc is None else acc + f_f
            count += 1
            
        if acc is not None:
            self.bg_base = (acc / count).astype(np.uint8)
            print("Background Captured!")
            return True
        return False

    def get_mask(self, frame, flow):
        """Dispatches to the correct logic based on user selection."""
        if self.use_flow_mask:
            return self._mask_from_flow(flow)
        else:
            return self._mask_from_static(frame)

    def _mask_from_static(self, frame):
        """
        Ported from make_foreground_mask.
        Uses YCrCb difference and Otsu thresholding.
        """
        if self.bg_base is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        # 1. YCrCb Difference
        f_ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        b_ycc = cv2.cvtColor(self.bg_base, cv2.COLOR_BGR2YCrCb)
        diff = cv2.absdiff(f_ycc, b_ycc)

        # 2. Weighted Score Port
        # Y=0.5, Cr=1.0, Cb=1.0
        score = 0.3 * diff[..., 0] + 1.2 * diff[..., 1] + 1.2 * diff[..., 2]
        score_u8 = np.clip(score, 0, 255).astype(np.uint8)

        # 3. Otsu Thresholding
        _, mask = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return self._post_process(mask)

    def _mask_from_flow(self, flow):
        """
        Ported from make_mask_from_flow_robust.
        Generates mask based on motion magnitude with decay.
        """
        h, w = flow.shape[:2]
        if self.flow_acc is None or self.flow_acc.shape != (h, w):
            self.flow_acc = np.zeros((h, w), dtype=np.float32)

        # Magnitude of flow vectors
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Accumulate with decay (solves the 'statue' problem)
        self.flow_acc = np.maximum(mag, self.flow_acc * self.DECAY)
        
        _, mask = cv2.threshold(self.flow_acc, self.NOISE_THRESHOLD, 255, cv2.THRESH_BINARY)
        return self._post_process(mask.astype(np.uint8))
   
    def _post_process(self, mask):
        """
        Advanced morphological reconstruction to fix 'missing limbs'.
        Strategy: Bridge Gaps -> Find Contours -> Fill Hulls.
        """
        h, w = mask.shape[:2]

        # 1. Morphological Closing (The "Bridge")
        # Connects disjoint parts (e.g., hand separated from wrist)
        # We use a rectangular kernel here as it bridges gaps better than ellipses
        kernel_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_bridge, iterations=2)

        # 2. Find Contours (External only - we don't care about holes inside yet)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 3. Smart Filling (The "Solid Silhouette")
        # Create a clean new mask to draw on
        filled_mask = np.zeros_like(mask)
        
        if contours:
            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Keep the top N largest blobs (e.g., 2 people max)
            # This filters out small noise automatically
            for cnt in contours[:2]:
                if cv2.contourArea(cnt) > 500: # Minimum size threshold
                    # Option A: Draw the filled contour (Exact shape)
                    cv2.drawContours(filled_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                    
                    # Option B (Optional): Convex Hull
                    # If you still have gaps, uncomment this to wrap a "shrink wrap" around the person
                    # hull = cv2.convexHull(cnt)
                    # cv2.drawContours(filled_mask, [hull], -1, 255, thickness=cv2.FILLED)

        # 4. Final Smoothing (Anti-alias the edges)
        return cv2.GaussianBlur(filled_mask, (11, 11), 0)