import cv2
import numpy as np
from .baseEffect import BaseEffect

class ArrowEffect(BaseEffect):
    def __init__(self, step=30, threshold=2.0, color=(0, 255, 0)):
        super().__init__("ARROWS")
        self.step = step
        self.threshold = threshold
        self.color = color

    def apply(self, frame, flow, mask=None):
        out = frame.copy()
        h, w = frame.shape[:2]

        for y in range(self.step // 2, h, self.step):
            for x in range(self.step // 2, w, self.step):
                fx, fy = flow[y, x]

                # Ignore noise
                if (fx**2 + fy**2) < (self.threshold**2):
                    continue

                cv2.arrowedLine(
                    out, (x, y), (int(x + fx * 4), int(y + fy * 4)),
                    self.color, 1, tipLength=0.3
                )
        return out

    def reset(self):
        pass # Arrows don't need persistent memory