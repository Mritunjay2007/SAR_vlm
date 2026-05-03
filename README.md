# SAR_vlm
A sar model using vlm and drone. mainly to optimise battery and mark hotspots to find subject faster and efficiently

---

```markdown
# 🚁 Vision-Language Guided Search & Rescue (SAR) Drone Simulation

## 📌 Overview

This project presents a **Vision-Language Model (VLM) driven Search & Rescue system** for detecting and locating humans in complex environments such as:

- Dense jungle
- Disaster zones
- Highways
- Open terrain

The system simulates a **drone-based search operation** using:

- 🧠 AI-based perception (YOLOE)
- 📊 Probabilistic reasoning (grid-based belief)
- 🧭 Intelligent path planning
- 🎯 Target-focused search strategy

---

## 🎯 Objective

To design a system that:

- Detects humans in **any posture or condition**
- Works in **unstructured environments**
- Minimizes search time
- Avoids redundant scanning
- Demonstrates **real-world deployability**

---

## 🧠 Core Idea

We model the search space as a **grid-based probability map**, where:

Each cell represents:
- likelihood of human presence
- updated dynamically using detections

---

## ⚙️ System Pipeline

### 1. Image Acquisition
- Input: single aerial image (drone view)
- Image is divided into `N × N` grid

---

### 2. Perception (YOLOE - Prompt-based Detection)

We use **YOLOE (Ultralytics)** for detection.

Unlike traditional YOLO:
- supports **text prompts**
- detects humans in multiple contexts

### Prompts used:
```

person
human
human body
person lying on ground
person sitting
person standing
person partially hidden
injured person
lost person
survivor

````

---

### 3. Probability Map Construction

Each detection is converted into a grid probability:

\[
P(c) = \frac{\text{confidence}}{\max(\text{confidence})}
\]

Where:
- \( c \) = grid cell

---

### 4. Search Algorithm (Custom Designed)

We define a scoring function:

\[
Score = \alpha P - \beta V - \gamma D
\]

Where:

| Term | Meaning |
|------|--------|
| \( P \) | Detection probability |
| \( V \) | Visit count (penalty) |
| \( D \) | Distance from drone |

Constants:
- \( \alpha = 2.5 \)
- \( \beta = 1.5 \)
- \( \gamma = 0.5 \)

---

### 5. Drone Movement

Drone follows:

- **Greedy + penalty-based navigation**
- Moves one step at a time
- Avoids revisiting same cells
- Prioritizes high-probability regions

---

### 6. Termination Condition

Search stops when:

- High confidence region found (`P > 0.8`)
- OR maximum steps reached

---

## 📊 Visualization

The system displays:

- 🔥 Heatmap of probability
- 📍 Drone path (cyan line)
- 🔵 Current drone position
- 🟥 Detected regions overlay

---

## 🧪 Features

- Real-time simulation
- Works on any uploaded image
- Efficient search strategy
- Avoids redundant traversal
- Explainable AI pipeline

---

## 🧰 Tech Stack

- Python
- Streamlit (UI)
- Ultralytics YOLOE
- NumPy
- Matplotlib
- PyTorch

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install ultralytics streamlit torch torchvision pillow numpy pandas matplotlib
````

---

### 2. Run app

```bash
streamlit run app.py
```

---

### 3. Upload image

* Use aerial / jungle / drone images
* Click **Run**

---

## 📁 Project Structure

```
SAR_vlm/
│
├── app.py
├── dataset/          # (ignored)
├── data_dump/        # (ignored)
├── utils/
├── planner/
├── perception/
└── README.md
```

---

## ⚠️ Notes

* Dataset is **not included** (large size)
* Model weights downloaded automatically
* Works best with:

  * 640px images
  * clear aerial perspective

---

## 🔬 Research Contributions

This project introduces:

* Hybrid **VLM + Probabilistic Search**
* Efficient grid-based exploration
* Prompt-based human detection
* Adaptive search strategy

---

## 🚀 Future Improvements

* Multi-drone coordination
* Temporal tracking (video input)
* Reinforcement Learning planner
* Thermal + RGB fusion
* Real-world drone integration

---

## 👨‍💻 Author

Mritunjay Kumar

---
