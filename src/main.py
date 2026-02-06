import cv2
import time
import numpy as np
from core.optFlow import OpticalFlowEngine
from core.background import BackgroundProcessor
from effects.geometry import ArrowEffect, GridWarpEffect
from effects.fluidPaint import FluidPaintEffect
from effects.debug import ShowMaskEffect
from effects.trails import MotionTrailEffect
from utils.hud import HUD

class ExhibitionApp:
    def __init__(self):
        # 1. Hardware & Engine Setup
        self.cap = cv2.VideoCapture(0)
        self.flow_engine = OpticalFlowEngine(scale=0.5) # Ported FLOW_SCALE
        self.bg_processor = BackgroundProcessor()
        
        # 2. Effect Library (The Exhibition Playlist)
        self.effects = [
            ShowMaskEffect(),                             # The Science (Study)
            FluidPaintEffect(decay=0.985),                # The Fluid Art
            GridWarpEffect(step=30, amplitude=4.0),       # The Geometry
            MotionTrailEffect(trail_length=0.95),         # The Ghosting
            MotionTrailEffect(thick_mode=True)            # The Solid Rastro
        ]
        self.current_idx = 0
        self.effect_duration = 15.0  # Duration in seconds
        self.start_time = time.time()
        
        # 3. UI and Display State
        ret, frame = self.cap.read()
        if not ret: raise RuntimeError("Could not initialize camera.")
        h, w = frame.shape[:2]
        self.hud = HUD(w, h)
        self.window_name = "Scientific Exhibition: Optical Flow Study"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def run(self):
        """The Main Application Loop"""
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            # Pre-processing: Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # 1. Core Logic: Flow and Masking
            flow = self.flow_engine.update(frame)
            mask = self.bg_processor.get_mask(frame, flow)
            
            # 2. Timing and Auto-Rotation
            elapsed = time.time() - self.start_time
            if elapsed > self.effect_duration:
                self.next_effect()
                elapsed = 0

            # 3. Render Current Effect
            current_effect = self.effects[self.current_idx]
            try:
                output = current_effect.apply(frame, flow, mask)
            except Exception as e:
                print(f"Effect Error: {e}")
                output = frame

            # 4. HUD and Visual Feedback
            method_label = "MODE: " + ("Optical Flow" if self.bg_processor.use_flow_mask else "Static BG")
            self.hud.render(
                output, 
                current_effect.name, 
                method_label, 
                remaining_time=(self.effect_duration - elapsed)
            )

            cv2.imshow(self.window_name, output)
            
            # 5. Input Handling
            if self.handle_input():
                break

        self.cleanup()

    def next_effect(self):
        """Cleanly transition to the next visual effect."""
        self.current_idx = (self.current_idx + 1) % len(self.effects)
        self.effects[self.current_idx].reset()
        self.start_time = time.time()

    def handle_input(self):
        """Ported keyboard controls from backSubtr.py"""
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]: # Quit
            return True
        elif key == ord('n'): # Next Effect
            self.next_effect()
        elif key == ord('m'): # Toggle Study Method
            self.bg_processor.set_mode(not self.bg_processor.use_flow_mask)
        elif key == ord('b'): # Capture Static Background
            self.bg_processor.capture_static_model(self.cap)
        elif key == ord('d'): # Toggle HUD
            self.hud.toggle()
        elif key == ord('r'): # Reset Current Effect
            self.effects[self.current_idx].reset()
        return False

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = ExhibitionApp()
    app.run()