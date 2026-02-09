# Optical Flow & Background Subtraction: Scientific Exhibition System

An interactive computer vision system designed for two purposes: an academic study of different **Optical Flow** algorithms and a creative application that applies real-time fluid dynamics and geometric effects to human motion.

## 📋 Table of Contents

* [Overview](https://www.google.com/search?q=%23overview)
* [Scientific Objectives](https://www.google.com/search?q=%23scientific-objectives)
* [System Architecture](https://www.google.com/search?q=%23system-architecture)
* [Optical Flow Methods](https://www.google.com/search?q=%23optical-flow-methods)
* [Visual Effects](https://www.google.com/search?q=%23visual-effects)
* [Getting Started](https://www.google.com/search?q=%23getting-started)
* [Controls](https://www.google.com/search?q=%23controls)
* [Data Collection & Analysis](https://www.google.com/search?q=%23data-collection--analysis)

---

## 🔍 Overview

This project explores the separation of moving foregrounds (people) from static backgrounds using various Optical Flow techniques. By moving from simple background subtraction to dense motion estimation, the system achieves robust detection even in challenging exhibition environments with varying lighting.

## 🎓 Scientific Objectives

1. **Algorithm Comparison:** Evaluate the efficiency and accuracy of **Farneback**, **DIS**, and **Dual TV-L1** algorithms.
2. **Robustness Analysis:** Study the "Statue Problem" (disappearance of subjects during pauses) and solve it using temporal decay accumulators.
3. **Performance Benchmarking:** Quantify the trade-off between computational latency (ms) and motion sensitivity (Energy).

---

## 🏗 System Architecture

The project is built using a **Modular Object-Oriented (OOP)** design to ensure stability during long-running exhibitions.

```text
├── main.py                # Application Controller & UI Loop
├── core/
│   ├── motion.py          # Optical Flow Engine (Strategy Pattern)
│   └── background.py      # Masking Logic (Static vs. Motion)
├── effects/
│   ├── baseEffect.py      # Abstract Blueprint for all effects
│   ├── fluid.py           # Advection-based Fluid Simulation
│   ├── geometry.py        # Grid Warp and Arrow Vectors
|   ├── debug.py           # Binary foreground mask.
│   └── trails.py          # Ghosting and Motion Trails   
├── utils/
│   ├── hud.py             # User Interface Overlay
│   └── benchmarker.py     # CSV Logging & Math Metrics
└── analyze_results.py     # Data Analysis & Graphing Tool

```

---

## ⚙ Optical Flow Methods

The system supports hot-swapping between three primary engines:

* **DIS (Dense Inverse Search):** Fast, modern CPU-based method. High temporal stability.
* **Farneback:** Classical dense flow. Reliable and fast for standard hardware.
* **Dual TV-L1:** High-quality variational method. Exceptional smoothness, ideal for artistic effects (high CPU cost).

---

## 🎨 Visual Effects

* **Fluid Paint:** Real-time color advection. Motion "paints" the background with swirling colors.
* **Grid Warp:** A virtual wireframe grid that deforms in response to physical motion.
* **Motion Trails:** Persistent "ghosting" effects that track the path of the subject.
* **Show Mask:** A debug view illustrating the raw background subtraction logic for academic transparency.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* OpenCV (`opencv-contrib-python` required for advanced flow methods)
* Pandas & Matplotlib (for data analysis)

### Installation

```bash
git clone https://github.com/yourusername/optical-flow-exhibition.git
cd optical-flow-exhibition
pip install -r requirements.txt

```

### Running the System

```bash
python main.py

```

---

## ⌨ Controls

| Key | Action |
| --- | --- |
| `n` | **Next Effect:** Cycle through visual styles. |
| `m` | **Toggle Mask:** Switch between Static Subtraction and Motion Masking. |
| `o` | **Swap Engine:** Cycle between DIS, TV-L1, and Farneback engines. |
| `b` | **Capture BG:** Capture a new static background model (Static mode only). |
| `d` | **HUD:** Toggle on-screen information overlay. |
| `q` | **Quit:** Safely close the application. |

---

## 📊 Data Collection & Analysis

The system automatically logs performance data to `exhibition_data.csv` every 0.5 seconds.

**Metrics Collected:**

* **Latency (ms):** Processing time per frame.
* **Energy:** Mean magnitude of motion vectors (Sensitivity).
* **Sparsity (%):** Percentage of moving pixels (Noise detection).

To generate your academic report graphs, run:

```bash
python analyze_results.py

```

---
