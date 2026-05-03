import random
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import torch

# ============================
# SETTINGS
# ============================
GRID_DEFAULT = 12
MAX_STEPS = 80

PROMPTS = [
    "person",
    "human",
    "human body",
    "person lying on ground",
    "person sitting",
    "person standing",
    "person partially hidden",
    "injured person",
    "lost person",
    "survivor",
]

# ============================
# LOAD YOLOE
# ============================
@st.cache_resource
def load_model(model_name):
    from ultralytics import YOLO
    model = YOLO(model_name)
    return model

# ============================
# GRID SPLIT
# ============================
def split_grid(image, grid):
    w, h = image.size
    cells = []
    for r in range(grid):
        for c in range(grid):
            x1 = int(c * w / grid)
            y1 = int(r * h / grid)
            x2 = int((c+1) * w / grid)
            y2 = int((r+1) * h / grid)

            crop = image.crop((x1, y1, x2, y2))
            cells.append({
                "row": r,
                "col": c,
                "crop": crop,
                "box": (x1,y1,x2,y2)
            })
    return cells

# ============================
# DETECTION
# ============================
def detect(model, image, prompts, threshold):
    model.set_classes(prompts)

    results = model.predict(
        source=image,
        conf=threshold,
        imgsz=640,
        verbose=False
    )

    if not results or results[0].boxes is None:
        return []

    r = results[0]
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy()

    detections = []
    for b, s, c in zip(boxes, scores, cls):
        label = prompts[int(c)]
        detections.append({
            "box": b,
            "score": float(s),
            "label": label
        })

    return detections

# ============================
# MAP BUILDING
# ============================
def build_map(image, grid, model, prompts, threshold):
    cells = split_grid(image, grid)
    dets = detect(model, image, prompts, threshold)

    score_map = np.zeros((grid, grid))

    for d in dets:
        x1,y1,x2,y2 = d["box"]
        cx = int((x1+x2)/2 / image.size[0] * grid)
        cy = int((y1+y2)/2 / image.size[1] * grid)

        cx = min(grid-1, max(0, cx))
        cy = min(grid-1, max(0, cy))

        score_map[cy][cx] += d["score"]

    if score_map.max() > 0:
        score_map /= score_map.max()

    return cells, score_map, dets

# ============================
# PLANNER (FIXED)
# ============================
class Planner:
    def __init__(self, grid):
        self.grid = grid
        self.visits = np.zeros((grid,grid))

    def next(self, score_map, pos):
        best = pos
        best_score = -999

        for r in range(self.grid):
            for c in range(self.grid):

                prob = score_map[r][c]
                visit_penalty = self.visits[r][c]
                dist = abs(pos[0]-r) + abs(pos[1]-c)

                score = 2.5*prob - 1.5*visit_penalty - 0.5*dist

                if score > best_score:
                    best_score = score
                    best = (r,c)

        return best

# ============================
# MOVE STEP
# ============================
def move(pos, target):
    r,c = pos
    tr,tc = target

    if r < tr: r+=1
    elif r > tr: r-=1
    elif c < tc: c+=1
    elif c > tc: c-=1

    return (r,c)

# ============================
# VISUALIZATION
# ============================
def draw(image, score_map, grid, drone, path):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(image)

    w,h = image.size

    for r in range(grid):
        for c in range(grid):
            val = score_map[r][c]
            if val > 0:
                x = c*w/grid
                y = r*h/grid
                rect = plt.Rectangle((x,y), w/grid, h/grid,
                                     color=(1,0,0,val*0.5))
                ax.add_patch(rect)

    if path:
        xs = [p[1]*w/grid for p in path]
        ys = [p[0]*h/grid for p in path]
        ax.plot(xs, ys, 'cyan')

    dr,dc = drone
    ax.scatter([dc*w/grid],[dr*h/grid], c='blue', s=100)

    ax.axis("off")
    return fig

# ============================
# MAIN
# ============================
st.title("SAR Drone with YOLOE")

uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

model_name = st.selectbox("Model", ["yoloe-11s-seg.pt","yoloe-11m-seg.pt"])
grid = st.slider("Grid Size", 8,20,GRID_DEFAULT)
steps = st.slider("Steps", 20,200,MAX_STEPS)
threshold = st.slider("Detection Threshold",0.1,0.9,0.3)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image)

    model = load_model(model_name)

    if st.button("Run"):
        cells, score_map, dets = build_map(image, grid, model, PROMPTS, threshold)

        planner = Planner(grid)
        pos = (0,0)
        path = [pos]

        for i in range(steps):
            planner.visits[pos]+=1

            target = planner.next(score_map, pos)
            pos = move(pos, target)

            path.append(pos)

            if score_map[pos] > 0.8:
                break

        fig = draw(image, score_map, grid, pos, path)
        st.pyplot(fig)

        st.write("Detections:", dets[:10])