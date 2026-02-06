import cv2

class HUD:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.active = True  # Allows toggling with 'd' key
        
        # Style configurations (centralized here)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.primary_color = (0, 255, 255)  # Yellow for titles
        self.secondary_color = (200, 200, 200) # Grey for info
        self.alert_color = (0, 0, 255)
        
    def toggle(self):
        self.active = not self.active

    def render(self, frame, effect_name, method_name, fps=None, remaining_time=None):
        """
        Draws the standard interface on the frame.
        """
        if not self.active:
            return

        # 1. Top Section: Status
        self._draw_text(frame, f"EFEITO: {effect_name}", (20, 35), 0.8, self.primary_color)
        
        if remaining_time is not None:
            self._draw_text(frame, f"Prox. em: {remaining_time:.1f}s", (20, 65), 0.6, self.secondary_color)
            
        self._draw_text(frame, method_name, (20, 95), 0.5, self.secondary_color)

        # 2. Bottom Section: Controls
        help_text = "n/p:Mudar | d:HUD | r:Reset | b:Fundo | q:Sair"
        self._draw_text(frame, help_text, (20, self.height - 20), 0.5, self.secondary_color)

    def _draw_text(self, img, text, pos, scale, color):
        """Internal helper to wrap cv2.putText with anti-aliasing"""
        cv2.putText(
            img, text, pos, 
            self.font, scale, color, 
            1, cv2.LINE_AA # Thickness 1 looks cleaner
        )