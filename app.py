import random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import torch

# =========================================================
# SETTINGS
# =========================================================
GRID_DEFAULT = 12
MAX_STEPS_DEFAULT = 80

PROMPTS_DEFAULT = [
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
    "backpack",
    "clothes",
    "shoe",
    "footprints",
]

# direct human cues are weighted highest
LABEL_WEIGHTS = {
    "person": 1.00,
    "human": 1.00,
    "human body": 1.00,
    "person lying on ground": 0.98,
    "person sitting": 0.95,
    "person standing": 0.95,
    "person partially hidden": 0.98,
    "injured person": 0.98,
    "lost person": 0.98,
    "survivor": 0.98,
    "backpack": 0.65,
    "clothes": 0.60,
    "shoe": 0.55,
    "footprints": 0.55,
}

# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource
def load_model(model_name: str):
    from ultralytics import YOLO
    return YOLO(model_name)

def is_promptable_model(model_name: str) -> bool:
    name = model_name.lower()
    return "yoloe" in name or "world" in name

def parse_prompts(text: str):
    prompts = [line.strip() for line in text.splitlines() if line.strip()]
    return prompts if prompts else PROMPTS_DEFAULT

# =========================================================
# IMAGE / GRID UTILITIES
# =========================================================
def split_grid(image: Image.Image, grid: int):
    w, h = image.size
    cells = []

    for r in range(grid):
        for c in range(grid):
            x1 = int(c * w / grid)
            y1 = int(r * h / grid)
            x2 = int((c + 1) * w / grid)
            y2 = int((r + 1) * h / grid)

            crop = image.crop((x1, y1, x2, y2))
            cells.append(
                {
                    "row": r,
                    "col": c,
                    "crop": crop,
                    "box": (x1, y1, x2, y2),
                }
            )
    return cells

def split_subgrid(crop: Image.Image, subgrid: int):
    w, h = crop.size
    subs = []

    for r in range(subgrid):
        for c in range(subgrid):
            x1 = int(c * w / subgrid)
            y1 = int(r * h / subgrid)
            x2 = int((c + 1) * w / subgrid)
            y2 = int((r + 1) * h / subgrid)

            sub = crop.crop((x1, y1, x2, y2))
            subs.append(
                {
                    "row": r,
                    "col": c,
                    "crop": sub,
                    "box": (x1, y1, x2, y2),
                }
            )
    return subs

def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

# =========================================================
# DETECTION
# =========================================================
def detect_objects(image: Image.Image, prompts, model, model_name: str, threshold: float):
    """
    YOLOE mode: promptable open-vocabulary detection.
    YOLO11 mode: fixed person detector fallback.
    """
    if model is None:
        return []

    promptable = is_promptable_model(model_name)

    try:
        if promptable and hasattr(model, "set_classes"):
            try:
                model.set_classes(prompts)
            except Exception:
                pass

        predict_kwargs = dict(
            source=image.convert("RGB"),
            conf=threshold,
            imgsz=640,
            verbose=False,
            max_det=25,
            iou=0.5,
        )

        # Fallback model: person-only detection on COCO class 0
        if not promptable:
            predict_kwargs["classes"] = [0]

        results = model.predict(**predict_kwargs)
    except Exception:
        return []

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return []

    r = results[0]
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy()

    detections = []
    names = getattr(model, "names", {})

    for b, s, c in zip(boxes, scores, cls):
        c = int(c)

        if promptable:
            label = prompts[c] if 0 <= c < len(prompts) else "person"
        else:
            if isinstance(names, dict):
                label = names.get(c, "person")
            else:
                label = names[c] if 0 <= c < len(names) else "person"

        weight = LABEL_WEIGHTS.get(label.lower(), 0.25)
        evidence = float(s) * weight

        detections.append(
            {
                "box": [float(v) for v in b.tolist()],
                "score": float(s),
                "label": label,
                "evidence": evidence,
            }
        )

    return detections

# =========================================================
# MAP BUILDING
# =========================================================
def rasterize_to_grid(detections, image_size, grid):
    w, h = image_size
    xs = np.linspace(0, w, grid + 1)
    ys = np.linspace(0, h, grid + 1)

    score_map = np.zeros((grid, grid), dtype=np.float32)
    cell_label = np.full((grid, grid), "", dtype=object)

    rows = []
    for det in detections:
        box = det["box"]
        evidence = det["evidence"]
        label = det["label"]

        for r in range(grid):
            for c in range(grid):
                cell_box = (xs[c], ys[r], xs[c + 1], ys[r + 1])
                inter = intersection_area(box, cell_box)
                if inter <= 0:
                    continue

                frac = inter / max(box_area(cell_box), 1e-9)
                add = evidence * frac

                # accumulate evidence across all overlapping cells
                score_map[r, c] += add
                if add > 0 and (cell_label[r, c] == "" or add > score_map[r, c]):
                    cell_label[r, c] = label

        rows.append(
            {
                "label": label,
                "score": det["score"],
                "evidence": evidence,
                "box": det["box"],
            }
        )

    if score_map.max() > 0:
        score_map = score_map / score_map.max()

    cell_rows = []
    for r in range(grid):
        for c in range(grid):
            cell_rows.append(
                {
                    "row": r,
                    "col": c,
                    "score": float(score_map[r, c]),
                    "label": str(cell_label[r, c]) if cell_label[r, c] else "background",
                }
            )

    det_df = pd.DataFrame(rows)
    cell_df = pd.DataFrame(cell_rows).sort_values("score", ascending=False)

    return score_map, det_df, cell_df

def heuristic_map(image: Image.Image, grid: int):
    cells = split_grid(image, grid)
    score_map = np.zeros((grid, grid), dtype=np.float32)

    for cell in cells:
        arr = np.asarray(cell["crop"].convert("RGB")).astype(np.float32) / 255.0
        gray = arr.mean(axis=2)
        texture = float(gray.std())
        darkness = float(1.0 - gray.mean())
        green = float(arr[:, :, 1].mean())

        score_map[cell["row"], cell["col"]] = 0.45 * texture + 0.35 * darkness + 0.20 * green

    if score_map.max() > 0:
        score_map = score_map / score_map.max()

    cell_rows = [
        {"row": r, "col": c, "score": float(score_map[r, c]), "label": "heuristic"}
        for r in range(grid)
        for c in range(grid)
    ]
    return score_map, pd.DataFrame(), pd.DataFrame(cell_rows).sort_values("score", ascending=False), cells

def build_map(image, grid, model, model_name, prompts, threshold):
    cells = split_grid(image, grid)
    detections = detect_objects(image, prompts, model, model_name, threshold)

    if len(detections) == 0:
        # detector fallback if nothing is found
        score_map, det_df, cell_df, _ = heuristic_map(image, grid)
        return cells, score_map, det_df, cell_df

    score_map, det_df, cell_df = rasterize_to_grid(detections, image.size, grid)
    return cells, score_map, det_df, cell_df

def refine_top_k_cells(image, cells, score_map, prompts, model, model_name, top_k=4, subgrid=3, threshold=0.08):
    flat = [((r, c), float(score_map[r, c])) for r in range(score_map.shape[0]) for c in range(score_map.shape[1])]
    flat.sort(key=lambda x: x[1], reverse=True)
    chosen = [rc for rc, _ in flat[:top_k]]

    refine_rows = []

    for (r, c) in chosen:
        cell = cells[r * score_map.shape[1] + c]
        subs = split_subgrid(cell["crop"], subgrid)

        local_best = 0.0
        local_label = "background"

        for sub in subs:
            dets = detect_objects(sub["crop"], prompts, model, model_name, threshold)
            if len(dets) == 0:
                arr = np.asarray(sub["crop"].convert("RGB")).astype(np.float32) / 255.0
                gray = arr.mean(axis=2)
                texture = float(gray.std())
                darkness = float(1.0 - gray.mean())
                green = float(arr[:, :, 1].mean())
                local_score = 0.45 * texture + 0.35 * darkness + 0.20 * green
                local_label = "heuristic_refine"
            else:
                best = max(dets, key=lambda x: x["evidence"])
                local_score = best["evidence"]
                local_label = best["label"]

            local_best = max(local_best, float(local_score))

        score_map[r, c] = 0.60 * float(score_map[r, c]) + 0.40 * local_best
        refine_rows.append(
            {
                "row": r,
                "col": c,
                "refined_score": float(local_best),
                "refined_label": local_label,
            }
        )

    if score_map.max() > 0:
        score_map = score_map / score_map.max()

    refine_df = pd.DataFrame(refine_rows).sort_values("refined_score", ascending=False)
    return score_map, refine_df

# =========================================================
# PLANNER
# =========================================================
class Planner:
    def __init__(self, grid):
        self.grid = grid
        self.visits = np.zeros((grid, grid), dtype=np.int32)
        self.last_visit = np.full((grid, grid), -10_000, dtype=np.int32)
        self.step_idx = 0

    def tick(self):
        self.step_idx += 1

    def update_visit(self, pos):
        self.visits[pos] += 1
        self.last_visit[pos] = self.step_idx

    def neighbors(self, r, c):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid and 0 <= nc < self.grid:
                yield nr, nc

    def frontier_cells(self, explored):
        frontiers = []
        for r in range(self.grid):
            for c in range(self.grid):
                if explored[r, c] == 0:
                    continue
                if any(explored[nr, nc] == 0 for nr, nc in self.neighbors(r, c)):
                    frontiers.append((r, c))
        return frontiers

    def score_goal(self, cell, score_map, explored, current_pos, battery):
        r, c = cell

        if self.visits[r, c] > 3:
            return -1e9

        p = float(score_map[r, c])
        dist = abs(current_pos[0] - r) + abs(current_pos[1] - c)
        novelty = 1.0 if explored[r, c] == 0 else 0.0
        visits = float(self.visits[r, c])

        age = self.step_idx - int(self.last_visit[r, c])
        recency = np.exp(-max(age, 0) / 5.0)
        urgency = 1.0 + (1.0 - battery)

        return (
            2.8 * p +
            1.3 * novelty +
            0.5 * urgency -
            0.9 * dist -
            2.5 * visits -
            0.8 * recency
        )

    def choose_goal(self, score_map, explored, current_pos, battery, lock_threshold):
        max_cell = tuple(np.unravel_index(np.argmax(score_map), score_map.shape))
        max_score = float(score_map[max_cell])

        # if very confident, lock on and head straight there
        if max_score >= lock_threshold:
            return max_cell, True, max_score

        candidates = self.frontier_cells(explored)
        if not candidates:
            unexplored = [tuple(x) for x in zip(*np.where(explored == 0))]
            candidates = unexplored if unexplored else [current_pos]

        best_goal = current_pos
        best_score = -1e18

        for cell in candidates:
            s = self.score_goal(cell, score_map, explored, current_pos, battery)
            if s > best_score:
                best_score = s
                best_goal = cell

        return best_goal, False, max_score

    def step_toward(self, current, goal):
        r, c = current
        gr, gc = goal

        nr, nc = r, c
        if r < gr:
            nr += 1
        elif r > gr:
            nr -= 1
        elif c < gc:
            nc += 1
        elif c > gc:
            nc -= 1

        return (nr, nc)

    def escape_move(self, explored, current):
        r, c = current
        best = current
        best_score = -1e18

        for nr, nc in self.neighbors(r, c):
            bonus = 1.0 if explored[nr, nc] == 0 else 0.0
            penalty = 0.5 * self.visits[nr, nc]
            score = bonus - penalty
            if score > best_score:
                best_score = score
                best = (nr, nc)

        return best

# =========================================================
# VISUALIZATION
# =========================================================
def overlay_on_original(image, score_map, grid, drone=None, path=None, alpha=0.38, show_scores=True):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    w, h = image.size
    xs = np.linspace(0, w, grid + 1)
    ys = np.linspace(0, h, grid + 1)
    cmap = plt.cm.inferno
    vmax = max(float(score_map.max()), 1e-9)

    for r in range(grid):
        for c in range(grid):
            val = float(score_map[r, c]) / vmax
            color = cmap(val)
            L, R = xs[c], xs[c + 1]
            T, B = ys[r], ys[r + 1]

            rect = plt.Rectangle(
                (L, T),
                R - L,
                B - T,
                facecolor=(color[0], color[1], color[2], alpha),
                edgecolor=(1, 1, 1, 0.10),
                linewidth=0.35,
            )
            ax.add_patch(rect)

            if show_scores and grid <= 12:
                ax.text(
                    (L + R) / 2,
                    (T + B) / 2,
                    f"{score_map[r, c]:.2f}",
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=6,
                )

    if path:
        xs_path = []
        ys_path = []
        for rr, cc in path:
            L, R = xs[cc], xs[cc + 1]
            T, B = ys[rr], ys[rr + 1]
            xs_path.append((L + R) / 2)
            ys_path.append((T + B) / 2)
        ax.plot(xs_path, ys_path, color="cyan", linewidth=2)

    if drone is not None:
        rr, cc = drone
        L, R = xs[cc], xs[cc + 1]
        T, B = ys[rr], ys[rr + 1]
        ax.scatter([(L + R) / 2], [(T + B) / 2], c="deepskyblue", s=120, edgecolors="black", linewidths=1.0)

    ax.set_axis_off()
    fig.tight_layout()
    return fig

def plot_heatmap(score_map, drone=None, path=None, title="Confidence heatmap"):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    im = ax.imshow(score_map, cmap="inferno", vmin=0, vmax=1)

    if path:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        ax.plot(xs, ys, color="cyan", linewidth=2)

    if drone is not None:
        ax.scatter([drone[1]], [drone[0]], c="deepskyblue", s=120, edgecolors="black", linewidths=1.0)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig

# =========================================================
# SEARCH LOOP
# =========================================================
def run_search(
    image,
    grid,
    camera_radius,
    max_steps,
    prompts,
    model_name,
    top_k,
    refine_grid,
    threshold,
    lock_threshold,
    battery_start,
):
    model = load_model(model_name)

    cells, score_map, det_df, cell_df = build_map(image, grid, model, model_name, prompts, threshold)

    # coarse-to-fine refinement on the strongest cells
    score_map, refine_df = refine_top_k_cells(
        image=image,
        cells=cells,
        score_map=score_map,
        prompts=prompts,
        model=model,
        model_name=model_name,
        top_k=top_k,
        subgrid=refine_grid,
        threshold=max(0.04, threshold * 0.75),
    )

    if len(refine_df) > 0:
        cell_df = pd.concat(
            [
                cell_df,
                refine_df.rename(columns={"refined_score": "score", "refined_label": "label"}),
            ],
            ignore_index=True,
        )

    planner = Planner(grid)
    explored = np.zeros((grid, grid), dtype=np.float32)

    current = (0, 0)
    path = [current]
    history = []
    drone_history = []
    battery_history = []

    battery = battery_start
    locked = False
    lock_cell = None

    for _ in range(max_steps):
        planner.tick()
        planner.update_visit(current)

        r, c = current

        # camera footprint marks explored cells and reduces repeated re-scanning
        for rr in range(r - camera_radius, r + camera_radius + 1):
            for cc in range(c - camera_radius, c + camera_radius + 1):
                if 0 <= rr < grid and 0 <= cc < grid:
                    explored[rr, cc] = 1.0
                    score_map[rr, cc] *= 0.35

        if score_map.max() > 0:
            score_map = score_map / score_map.max()

        max_cell = tuple(np.unravel_index(np.argmax(score_map), score_map.shape))
        max_score = float(score_map[max_cell])

        if max_score >= lock_threshold:
            locked = True
            lock_cell = max_cell

        history.append(score_map.copy())
        drone_history.append(current)
        battery_history.append(battery)

        battery = max(0.0, battery - (1.0 / max_steps))
        if battery <= 0.02:
            break

        if locked:
            goal = lock_cell
            nxt = planner.step_toward(current, goal)
        else:
            goal, _, _ = planner.choose_goal(score_map, explored, current, battery, lock_threshold)
            nxt = planner.step_toward(current, goal)

        if planner.visits[nxt] > 3:
            nxt = planner.escape_move(explored, current)

        if nxt == current:
            break

        current = nxt
        path.append(current)

        if locked and current == lock_cell:
            break

        if explored.sum() >= grid * grid:
            break

    return {
        "score_map": score_map,
        "det_df": det_df,
        "cell_df": cell_df,
        "path": path,
        "history": history,
        "drone_history": drone_history,
        "battery_history": battery_history,
        "locked": locked,
        "lock_cell": lock_cell,
    }

# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="UAV SAR Search", layout="wide")
st.title("UAV SAR — Coarse-to-Fine Search")

st.write(
    "This version uses YOLOE when promptable detection is available, or YOLO11 as a fast fallback. "
    "It builds the heatmap from box overlap, not just the center point, and the planner avoids revisits."
)

mode = st.sidebar.radio("Input mode", ["Random from database", "Upload image"])
model_name = st.sidebar.selectbox(
    "Model",
    ["yoloe-11s-seg.pt", "yolo11m.pt"],
    index=0,
)

grid = st.sidebar.slider("Grid Size", 8, 20, GRID_DEFAULT)
steps = st.sidebar.slider("Steps", 20, 200, MAX_STEPS_DEFAULT)
threshold = st.sidebar.slider("Detection Threshold", 0.05, 0.90, 0.25, 0.05)
lock_threshold = st.sidebar.slider("Lock Threshold", 0.10, 1.00, 0.60, 0.05)
camera_radius = st.sidebar.slider("Camera Radius", 1, 3, 1)
top_k = st.sidebar.slider("Top-K refinement cells", 1, 10, 4)
refine_grid = st.sidebar.slider("Local refinement grid", 2, 6, 3)
db_dir = st.sidebar.text_input("Random database folder", "data/jungle_db")
prompt_text = st.sidebar.text_area(
    "Prompts (one per line)",
    value="\n".join(PROMPTS_DEFAULT),
    height=220,
)

uploaded = None
selected_image = None

if mode == "Random from database":
    images = []
    db_path = Path(db_dir)
    if db_path.exists():
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        images = sorted([p for p in db_path.rglob("*") if p.suffix.lower() in exts])

    if len(images) == 0:
        st.warning(f"No images found in `{db_dir}`.")
    else:
        if "rand_img" not in st.session_state:
            st.session_state["rand_img"] = random.choice(images)
        if st.sidebar.button("Pick another random image"):
            st.session_state["rand_img"] = random.choice(images)
        selected_image = Image.open(st.session_state["rand_img"]).convert("RGB")
        st.caption(f"Random image: {st.session_state['rand_img']}")
else:
    uploaded = st.file_uploader("Upload a jungle / SAR aerial image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is not None:
        selected_image = Image.open(uploaded).convert("RGB")

if selected_image is None:
    st.info("Choose a random image or upload one.")
    st.stop()

st.image(selected_image, caption="Input image", use_container_width=True)

run_button = st.button("Run search")

if run_button or "result" in st.session_state:
    prompts = parse_prompts(prompt_text)

    with st.spinner("Running detector + coarse-to-fine search..."):
        result = run_search(
            image=selected_image,
            grid=grid,
            camera_radius=camera_radius,
            max_steps=steps,
            prompts=prompts,
            model_name=model_name,
            top_k=top_k,
            refine_grid=refine_grid,
            threshold=threshold,
            lock_threshold=lock_threshold,
            battery_start=1.0,
        )
        st.session_state["result"] = result

if "result" in st.session_state:
    result = st.session_state["result"]
    history = result["history"]
    path = result["path"]
    drone_history = result["drone_history"]
    battery_history = result["battery_history"]
    det_df = result["det_df"]
    cell_df = result["cell_df"]

    step_idx = st.slider("Time step", 0, max(0, len(history) - 1), max(0, len(history) - 1))
    score_t = history[step_idx] if len(history) else result["score_map"]
    drone_t = drone_history[min(step_idx, len(drone_history) - 1)] if len(drone_history) else (0, 0)
    path_t = path[: min(step_idx + 1, len(path))]

    col1, col2 = st.columns(2)

    with col1:
        fig1 = overlay_on_original(
            image=selected_image,
            score_map=score_t,
            grid=grid,
            drone=drone_t,
            path=path_t,
            alpha=0.38,
            show_scores=(grid <= 12),
        )
        st.pyplot(fig1, clear_figure=True)

    with col2:
        fig2 = plot_heatmap(
            score_map=score_t,
            drone=drone_t,
            path=path_t,
            title="Confidence heatmap",
        )
        st.pyplot(fig2, clear_figure=True)

    st.subheader("Detector evidence")
    if len(det_df) > 0:
        st.dataframe(det_df.head(50), use_container_width=True)
    else:
        st.write("No detections; the heuristic fallback was used.")

    st.subheader("Per-cell scores")
    st.dataframe(cell_df.head(50), use_container_width=True)

    st.subheader("Tracking over time")
    if len(history) > 0 and len(cell_df) > 0:
        tracked = [tuple(x) for x in cell_df[["row", "col"]].drop_duplicates().head(20).to_numpy()]
        if not tracked:
            tracked = [(0, 0)]
        track_cell = st.selectbox("Cell to track", tracked)
        rr, cc = track_cell
        series = [h[rr, cc] for h in history]
        st.line_chart(pd.DataFrame({"score": series}))

    if len(battery_history) > 0:
        st.subheader("Battery over time")
        st.line_chart(pd.DataFrame({"battery": battery_history}))

    st.write(f"Lock-on: {result['locked']}")
    st.write(f"Target hotspot: {result['lock_cell']}")