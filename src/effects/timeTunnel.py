import cv2
import numpy as np
from collections import deque
from .baseEffect import BaseEffect

class TimeTunnelEffect(BaseEffect):
    def __init__(self, max_clones=8, frame_delay=6, color_shift=True):
        """
        :param max_clones: Number of clones in the tunnel.
        :param frame_delay: Space (in frames) between each clone.
        :param color_shift: If True, each deeper clone gets a different tint.
        """
        super().__init__("TIME_TUNNEL")
        self.max_clones = max_clones
        self.frame_delay = frame_delay
        self.color_shift = color_shift
        
        # Buffer stores (Isolated_Person_Image, Mask)
        self.buffer = deque(maxlen=max_clones * frame_delay + 1)

    def apply(self, frame, flow, mask, pose_results=None):
        h, w = frame.shape[:2]
        
        # 1. Isolate the person immediately (Black background)
        # We only store the 'cutout' to save memory and processing
        person_isolated = np.zeros_like(frame)
        cv2.copyTo(frame, mask, person_isolated)
        
        # 2. Store current cutout and mask in history
        self.buffer.append((person_isolated.copy(), mask.copy()))
        
        # 3. Create the Black Canvas for the effect
        canvas = np.zeros_like(frame)

        # 4. Layer Clones: Deepest (oldest) first, Current (newest) last
        for i in range(self.max_clones, 0, -1):
            idx = -(i * self.frame_delay)
            
            if abs(idx) < len(self.buffer):
                past_person, past_mask = self.buffer[idx]
                
                if self.color_shift:
                    # Apply a 'Time-Tunnel' tint: older clones fade to blue/purple
                    tint_factor = 1.0 - (i / self.max_clones)
                    clone_to_draw = cv2.addWeighted(
                        past_person, tint_factor, 
                        np.zeros_like(past_person), 0, 0
                    )
                    # Shift color hue slightly for each step
                    # (Simplified: just reduce brightness/saturation for depth)
                else:
                    clone_to_draw = past_person

                # Paste the past person onto the black canvas
                cv2.copyTo(clone_to_draw, past_mask, canvas)

        # 5. Final Layer: The 'Present' Person
        cv2.copyTo(person_isolated, mask, canvas)
        
        return canvas

    def reset(self):
        self.buffer.clear()