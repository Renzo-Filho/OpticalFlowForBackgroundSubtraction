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
        # 1. Initialize Camera
        self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()
        if not ret: raise RuntimeError("Could not initialize camera.")
        
        # 2. Engine Setup
        self.flow_engine = OpticalFlowEngine(scale=0.5)
        self.bg_processor = BackgroundProcessor()
        
        # 3. Effects Playlist
        self.effects = [
            ShowMaskEffect(),
            FluidPaintEffect(decay=0.985),
            GridWarpEffect(step=40, amplitude=3.0),
            MotionTrailEffect(trail_length=0.95),
            ArrowEffect(step=30)
        ]
        self.current_idx = 0
        self.effect_duration = 20.0
        self.start_time = time.time()
        
        # 4. Window & HUD Setup
        h, w = frame.shape[:2]
        self.hud = HUD(w, h)
        self.window_name = "Exhibition"
        
        # Use WINDOW_NORMAL to keep window controls (Minimize/Resize/Close)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Resize to a large standard resolution to fill the screen automatically
        # You can change (1280, 720) to (1920, 1080) if you have a Full HD screen
        cv2.resizeWindow(self.window_name, 1280, 720)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1) # Mirror view
            
            # --- Processing ---
            flow = self.flow_engine.update(frame)
            mask = self.bg_processor.get_mask(frame, flow)
            
            # --- Auto Rotation ---
            elapsed = time.time() - self.start_time
            if elapsed > self.effect_duration:
                self.next_effect()
                elapsed = 0

            # --- Render Effect ---
            current_effect = self.effects[self.current_idx]
            try:
                output = current_effect.apply(frame, flow, mask)
            except Exception as e:
                print(f"Error in {current_effect.name}: {e}")
                output = frame

            # --- HUD & Status Logic ---
            status_msg = ""
            if not self.bg_processor.use_flow_mask and self.bg_processor.bg_base is None:
                status_msg = "No BG captured (Press 'b')"

            method_label = "MODE: " + ("Motion (Flow)" if self.bg_processor.use_flow_mask else "Static (Subtr)")
            
            self.hud.render(
                output, 
                current_effect.name, 
                method_label, 
                remaining_time=(self.effect_duration - elapsed),
                extra_info=status_msg
            )

            cv2.imshow(self.window_name, output)
            
            if self.handle_input():
                break

        self.cleanup()

    def next_effect(self):
        self.current_idx = (self.current_idx + 1) % len(self.effects)
        self.effects[self.current_idx].reset()
        self.start_time = time.time()

    def handle_input(self):
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]: return True
        elif key == ord('n'): self.next_effect()
        elif key == ord('m'): self.bg_processor.set_mode(not self.bg_processor.use_flow_mask)
        elif key == ord('b'): self.bg_processor.capture_static_model(self.cap)
        elif key == ord('d'): self.hud.toggle()
        elif key == ord('r'): self.effects[self.current_idx].reset()
        return False

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = ExhibitionApp()
    app.run()