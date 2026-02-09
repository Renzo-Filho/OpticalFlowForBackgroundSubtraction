## **1. Understanding the Metrics**

| Metric | What it tells the Researcher | Exhibition Context |
| --- | --- | --- |
| **Magnitude Mean (Energy)** | Measures the average "strength" of the vectors. Higher values mean the method is sensitive to small movements (like breathing or cloth shifting). | Does the fluid move vigorously or sluggishly? |
| **Flow Sparsity (%)** | The percentage of the image where motion is detected. In a static room, a high % indicates **noise** (pixels "shimmering" for no reason). | How "clean" is the background? High sparsity means the method is messy. |
| **Latency (ms)** | The time the CPU takes to solve the equations. | Is the effect laggy? This determines if the method is "Exhibition Ready." |

---

### 2. How to interpret the results for your project

When you look at the generated graphs, here is what you should look for to support your academic arguments:

* **Latency vs. Real-time:** If a method’s latency is above 33ms, it is dropping below 30 FPS. If it's above 100ms (like TV-L1 often is), you can mathematically prove it is unsuitable for "high-speed interactive" exhibitions but perhaps better for "slow-motion art."
* **The Energy/Sparsity Trade-off:** * A "Good" method should have high **Energy** when someone is moving, but very low **Sparsity** when the room is empty.
* If **Farneback** has high Sparsity even when no one is moving, you have mathematical proof that it is "noisier" than **DIS**.


* **Consistency:** The **Standard Deviation (`std`)** in the summary table tells you how stable the method is. A high `std` in latency means the method is unpredictable, which can cause "stuttering" in the visual effects.

---