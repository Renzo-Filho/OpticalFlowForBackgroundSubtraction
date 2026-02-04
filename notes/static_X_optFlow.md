Here is the breakdown of the **process** for both modes, based directly on the functions in your file.

### 1. Static Background Subtraction (Mode A)

**The "Memory" Method**
*Code Function:* `make_foreground_mask`

This method works by comparing what the camera sees *now* against a stored memory of what the empty room looks like.

* **Step 1: The "Ground Truth" (Capture Phase)**
* When you press `b`, the function `capture_background_average` runs.
* It takes **150 frames** of the empty room and calculates the average color for every pixel.
* **Why Average?** To remove camera grain and slight light flickering. This creates a "clean" reference image called `bg_base`.


* **Step 2: The Comparison (Diff)**
* Every loop, the code converts the current frame and the `bg_base` into a color space called **YCrCb** (Luminance + Chroma). This is often better than RGB for handling shadows.
* It calculates the absolute difference: .
* If the difference is high (e.g., > 25), that pixel is considered "Foreground" (You).


* **Step 3: Cleanup (Morphology)**
* The raw difference is noisy. The code applies **Morphological Operations**:
* **Open:** Removes tiny white dots (noise).
* **Close:** Fills small black holes inside the white blobs (e.g., if part of your shirt matches the wall).

### 2. Optical Flow Subtraction (Mode B)

**The "Motion" Method**
*Code Function:* `make_mask_from_flow_robust` (The one we added)

This method ignores the "past" (the empty room) and only cares about the "change" (movement).

* **Step 1: The Vector Field**
* The function `calculate_smooth_flow` compares the *Current Frame* vs. the *Previous Frame* (milliseconds ago).
* It generates a **Vector Field**: A grid where every pixel has a direction  and a speed.


* **Step 2: Magnitude (Speed)**
* The code converts the vector  into a single number: **Magnitude**.
* 
* If the magnitude is > `NOISE_THRESHOLD` (2.0), that pixel is "moving."


* **Step 3: The "Memory" Fix (Accumulator)**
* **The Problem:** If you stand still, your magnitude becomes 0, and you disappear (The "Statue Problem").
* **The Solution:** The code uses a decay formula:
`Accumulator = Max(Current_Speed, Old_Accumulator * 0.90)`
* This creates a "fading trail" in the math. Even if you stop moving, the "heat" of your movement lingers for a few seconds, keeping you visible in the mask.



### Summary Comparison Table

| Feature | Static Mode (Mode A) | Optical Flow Mode (Mode B) |
| --- | --- | --- |
| **Core Logic** | "Difference from Memory" | "Difference from Previous Moment" |
| **Requires** | Empty Room Setup (Press 'b') | Constant Movement |
| **Best For** | Standing still, detailed outlines. | Moving cameras, dynamic scenes. |
| **Weakness** | Lighting changes (clouds, lamps). | "Aperture Problem" (solid colors don't show motion well). |