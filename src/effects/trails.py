import cv2
import numpy as np
from .baseEffect import BaseEffect

class GhostTrailEffect(BaseEffect):
    def __init__(self, alpha=0.90):
        super().__init__("GHOST_TRAILS")
        self.alpha = alpha
        self.acc = None

    def apply(self, frame, flow, mask):
        if self.acc is None: self.acc = frame.astype(np.float32)
        
        # The logic from your script
        frame_f = frame.astype(np.float32)
        self.acc = cv2.addWeighted(self.acc, self.alpha, frame_f, (1.0 - self.alpha), 0)
        
        # Apply mask so trails only appear on the person/motion
        mask_3 = (mask.astype(np.float32) / 255.0)[..., None]
        out = (self.acc * mask_3) + (frame_f * (1.0 - mask_3))
        return np.clip(out, 0, 255).astype(np.uint8)