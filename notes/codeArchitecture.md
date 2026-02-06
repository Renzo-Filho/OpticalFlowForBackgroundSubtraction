### 1. High-Level Architecture

We will separate the code into four distinct layers. This ensures that if you want to change the camera resolution, you don't break the fluid simulation code.

1. **Input Layer:** Handles the webcam and raw frame acquisition.
2. **Core Processing Layer:** The "Brain." It calculates Optical Flow and generates the Background/Foreground Masks.
3. **Visual Effects Layer:** The "Art." A system where each effect is a self-contained object (Plugin architecture).
4. **Application Layer:** The "Manager." Handles the main loop, user input, and effect switching.

---

### 2. Proposed Directory Structure

Instead of one giant file, we will organize it like this:

```text
exhibition_flow/
│
├── main.py                # Entry point (The Application Loop)
├── config.py              # Constants (FLOW_SCALE, COLORS, TIMERS)
│
├── core/
│   ├── __init__.py
│   ├── camera.py          # Class VideoInput
│   ├── motion.py          # Class OpticalFlowEngine
│   └── background.py      # Classes for StaticBG and MotionBG
│
├── effects/
│   ├── __init__.py
│   ├── base_effect.py     # Abstract Base Class (The Blueprint)
│   ├── fluid.py           # Class FluidEffect
│   ├── geometry.py        # Class GridWarp, Class Arrows
│   └── trails.py          # Class GhostTrail
│
└── utils/
    ├── __init__.py
    └── hud.py             # Class DisplayManager (Text overlays)

```

---

### 3. Class Design & Responsibilities

Here is how we translate your current functions into Classes (POO).

#### A. The Core Logic (`core/`)

**1. `OpticalFlowEngine**`

* **Responsibility:** It encapsulates the `calculate_smooth_flow`.
* **Why:** It needs to store the `prev_gray` frame internally so the main loop doesn't have to worry about it.
* **Methods:** `update(current_frame) -> flow_vectors`

**2. `BackgroundProcessor**`

* **Responsibility:** Generates the binary mask (0 = BG, 255 = FG).
* **Strategy Pattern:** We can switch between "Static Mode" and "Motion Mode" on the fly.
* **Attributes:** Stores the `bg_model` (the average image).
* **Methods:**
* `capture_background(camera)`
* `get_mask(frame, flow_vectors) -> mask`



#### B. The Effects System (`effects/`)

This is the most important part for your exhibition. Currently, you pass a huge `state` dictionary around. In OOP, **each effect becomes an object that manages its own memory.**

**1. `BaseEffect` (Abstract Class)**
Every effect must follow this contract:

* `__init__()`: Allocate memory (e.g., the black canvas for fluid).
* `apply(frame, flow, mask) -> image`: Returns the processed frame.
* `reset()`: Clears its internal canvas.

**2. Concrete Effects (e.g., `FluidPaintEffect`)**

* Instead of `if "fluid_canvas" not in state...`, the canvas is just `self.canvas` initialized in the constructor.
* This fixes the issue where switching effects might leak memory or visual artifacts.

#### C. The Manager (`main.py`)

**1. `ExhibitionApp**`

* Holds the list of effects: `self.effects = [FluidEffect(), GridEffect(), ...]`
* Manages the timer: `self.timer` to auto-rotate effects every 10 seconds.
* Handles Input: `self.handle_keys()`

---

### 4. Comparison: Script vs. OOP

| Feature | Current Script (`backSubtr.py`) | Proposed OOP System |
| --- | --- | --- |
| **State** | Shared `state = {}` dict passed everywhere. | `self.variable` inside specific classes. |
| **Adding Effects** | Add a function, update `main()`, check `state`. | Create a class inheriting `BaseEffect`, add to list. |
| **Robustness** | If one effect fails, the script might crash. | App catches errors in `effect.apply()` and skips frame. |
| **Exhibition** | Hard to tweak individual parameters. | Each effect has its own `self.params`. |

---

### 5. Step-by-Step Implementation Plan

To avoid overwhelming you, I suggest we build this in **3 Stages**:

* **Stage 1: The Foundation.** Create the `BaseEffect` class and the `OpticalFlowEngine`. Move the math out of the main loop.
* **Stage 2: The Logic.** Implement the `BackgroundProcessor` to cleanly swap between your "Static" and "Flow" masking methods.
* **Stage 3: The Polish.** Port the "Fluid" and "Trails" effects into the new class structure and build the final `main.py`.
