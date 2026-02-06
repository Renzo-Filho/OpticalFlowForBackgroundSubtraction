import cv2
import time
from core.optFlow import OpticalFlowEngine
from core.background import BackgroundProcessor
from effects.arrows import ArrowEffect
from effects.fluidPaint import FluidPaintEffect
from utils.hud import HUD

class ExhibitionApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.flow_engine = OpticalFlowEngine()
        self.bg_processor = BackgroundProcessor()
        
        # Load effects into a list
        self.effects = [
            FluidPaintEffect(),
            ArrowEffect()
        ]
        self.current_idx = 0
        self.effect_duration = 20 # seconds per effect
        self.start_time = time.time()
        
        # UI
        ret, frame = self.cap.read()
        h, w = frame.shape[:2]
        self.hud = HUD(w, h)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            # 1. Processing Pipeline
            flow = self.flow_engine.update(frame)
            mask = self.bg_processor.get_mask(frame, flow)
            
            # 2. Effect Rotation Logic
            elapsed = time.time() - self.start_time
            if elapsed > self.effect_duration:
                self.next_effect()

            # 3. Render
            current_effect = self.effects[self.current_idx]
            output = current_effect.apply(frame, flow, mask)
            
            # 4. UI Overlay
            method_name = "MODE: " + ("Motion" if self.bg_processor.use_motion_mode else "Static")
            self.hud.render(output, current_effect.name, method_name, 
                            remaining_time=(self.effect_duration - elapsed))

            cv2.imshow("Exhibition", output)
            if self.handle_input(): break

        self.cap.release()
        cv2.destroyAllWindows()

    def next_effect(self):
        self.current_idx = (self.current_idx + 1) % len(self.effects)
        self.effects[self.current_idx].reset()
        self.start_time = time.time()

    def handle_input(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): return True
        if key == ord('n'): self.next_effect()
        if key == ord('m'): self.bg_processor.set_mode(not self.bg_processor.use_motion_mode)
        if key == ord('b'): self.bg_processor.capture_static_model(self.cap)
        if key == ord('d'): self.hud.toggle()
        return False

if __name__ == "__main__":
    app = ExhibitionApp()
    app.run()