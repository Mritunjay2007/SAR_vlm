# SAR_vlm
A sar model using vlm and drone. mainly to optimise battery and mark hotspots to find subject faster and efficiently

```markdown
# 🚁 Uncertainty-Aware Vision-Language Search for UAV-Based SAR

## 📌 Overview

This project simulates an intelligent drone system for **Search and Rescue (SAR)** using:

- Probabilistic reasoning (Bayesian belief map)
- Vision-Language inspired perception
- Real aerial datasets (SARD + VisDrone)
- Energy-aware and spatially-aware search planning

The system models how a drone explores a jungle environment to locate a missing person under uncertainty.

---

## 🧠 Key Concepts

- **Belief Map**: Probability distribution over possible victim locations
- **Bayesian Update**: Updates belief based on observations
- **Entropy**: Measures uncertainty
- **Spatial Smoothing**: Creates hotspot tracking behavior
- **Planner**: Chooses next move based on probability and distance
- **Camera Footprint**: Drone observes multiple cells per step

---

## 📂 Project Structure

```

SAR_VLM/
│
├── main.py
├── config.py
│
├── environment/
├── agent/
├── belief/
├── perception/
├── planner/
├── utils/
│
├── data_loader/
│   ├── build_dataset.py
│   └── grid_mapper.py
│
├── dataset/        # (ignored in git)
├── data_dump/      # (ignored in git)

````

---

## 📊 Datasets Used

- **SARD Dataset** → human detection
- **VisDrone Dataset** → aerial imagery
- (Optional) C2A dataset → robustness

These are combined to create a **hybrid SAR dataset**.

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone <your-repo-url>
cd SAR_VLM
````

---

### 2️⃣ Install Dependencies

```bash
pip install numpy matplotlib pillow
```

(Optional for later phases)

```bash
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
```

---

### 3️⃣ Add Datasets

Place downloaded datasets inside:

```
data_dump/
```

Example:

```
data_dump/
├── sard/
├── VisDrone2019-DET-train/
```

---

### 4️⃣ Build Dataset

```bash
python data_loader/build_dataset.py
```

---

### 5️⃣ Create Grid Mapping

```bash
python data_loader/grid_mapper.py
```

This generates:

```
dataset/grid/
    0_0.jpg
    0_1.jpg
    ...
```

---

### 6️⃣ Run Simulation

```bash
python main.py
```

---

## 🎥 Output

### 1. Heatmap Visualization

* Red = high probability
* Blue = low probability

### 2. Image Grid Visualization

* Each cell = real image
* Drone movement tracked

---

## 📐 Mathematical Model

### Bayesian Update:

[
P(c|o) \propto P(o|c) \cdot P(c)
]

---

### Entropy:

[
H = -\sum P \log P
]

---

### Planner:

[
Score = \frac{P(victim)}{distance + 1}
]

---

## 🚀 Features Implemented

* Probabilistic belief update
* Multi-cell observation (camera footprint)
* Spatial smoothing (hotspot detection)
* Energy-aware planning
* Real dataset integration

---

## 🔬 Future Work

* CLIP-based Vision-Language perception
* Multi-evidence fusion (footprints, cloth, etc.)
* Multi-drone coordination
* Real-time deployment

---

## ⚠️ Note

Datasets are not included in the repository due to size constraints.
Please download them separately.

---

## 👨‍💻 Author

Mritunjay Kumar

````

---

# 🧠 3. FINAL CHECKLIST BEFORE RUN

---

## ✅ Make sure:

- `dataset/grid/` exists  
- images like `0_0.jpg` present  
- no missing folders  

---

## ✅ Run:

```bash
python main.py
````

