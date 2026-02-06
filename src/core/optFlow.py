import cv2
import numpy as np

class OpticalFlowEngine:
    def __init__(self, method="DIS", scale=0.25):
        self.scale = scale
        self.prev_gray = None
        self.method = method.upper()
        
        # Initialize DIS (Fastest advanced CPU method)
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        
        self.tvl1 = None
        if hasattr(cv2, 'optflow'):
            self.tvl1 = cv2.optflow.createOptFlow_DualTVL1()
            self.tvl1.setInnerIterations(5)  
            self.tvl1.setOuterIterations(2)
            self.tvl1.setTau(0.15)

    def set_method(self, method_name):
        """Allows swapping methods via keyboard during exhibition"""
        self.method = method_name.upper()
        self.prev_gray = None # Clear history to avoid dimension mismatch

    def update(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.zeros((gray.shape[0], gray.shape[1], 2), dtype=np.float32)

        # 1. Downscale for real-time performance
        prev_small = cv2.resize(self.prev_gray, None, fx=self.scale, fy=self.scale)
        curr_small = cv2.resize(gray, None, fx=self.scale, fy=self.scale)

        # 2. Dispatch to selected Algorithm
        if self.method == "DIS":
            flow_small = self.dis.calc(prev_small, curr_small, None)
        
        elif self.method == "TVL1":
            flow_small = self.tvl1.calc(prev_small, curr_small, None)
            
        else: # Default: Farneback (from your backSubtr.py)
            flow_small = cv2.calcOpticalFlowFarneback(
                prev_small, curr_small, None, 
                0.5, 3, 15, 3, 5, 1.2, 0
            )

        # 3. Post-Process (Blur for consistency)
        flow_small = cv2.GaussianBlur(flow_small, (15, 15), 5.0)

        # 4. Upscale back to original resolution
        h, w = gray.shape
        flow = cv2.resize(flow_small, (w, h))
        flow *= (1.0 / self.scale)

        self.prev_gray = gray.copy()
        return flow