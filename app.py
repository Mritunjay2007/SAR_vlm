import random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import torch

# ----------------------------
# Settings
# ----------------------------
DEFAULT_GRID_SIZE = 17
DEFAULT_CAMERA_RADIUS = 1
DEFAULT_MAX_STEPS = 120
DEFAULT_TOPK_REFINE = 4
DEFAULT_REFINEMENT_GRID = 3
DEFAULT_DB_DIR = "data/jungle_db"

DEFAULT_PROMPTS = [
    "person",
    "human",
    "person lying on ground",
    "person sitting in forest",
    "person partially hidden in jungle",
    "injured person",
    "lost person",
    "backpack",
    "clothes",
    "shoe",
    "footprints",
]

# ----------------------------
# Model loading
# ----------------------------
@st.cache_resource
def load_detector(model_id: str):
    """
    Grounding DINO via Hugging Face zero-shot object detection pipeline.
    Returns a pipeline object or None if loading fails.
    """
    try:
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        detector = pipeline(
            "zero-shot-object-detection",
            model=model_id,
            device=device,
        )
        return detector, True
    except Exception:
        return None, False


def parse_prompts(text: str):
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(line)
    return items if items else DEFAULT_PROMPTS


def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])


# ----------------------------
# Scene tiling
# ----------------------------
def split_image_into_grid(image: Image.Image, grid_size: int):
    """
    Split one original image into grid_size x grid_size cells.
    Each cell keeps original location via bbox.
    """
    image = image.convert("RGB")
    w, h = image.size
    xs = np.linspace(0, w, grid_size + 1).astype(int)
    ys = np.linspace(0, h, grid_size + 1).astype(int)

    cells = []
    for r in range(grid_size):
        for c in range(grid_size):
            box = (xs[c], ys[r], xs[c + 1], ys[r + 1])
            crop = image.crop(box)
            crop_224 = crop.resize((224, 224))
            cells.append(
                {
                    "row": r,
                    "col": c,
                    "box": box,
                    "crop": crop,
                    "crop_224": crop_224,
                }
            )
    return cells, (w, h)


def split_crop_into_subgrid(crop: Image.Image, subgrid: int):
    crop = crop.convert("RGB")
    w, h = crop.size
    xs = np.linspace(0, w, subgrid + 1).astype(int)
    ys = np.linspace(0, h, subgrid + 1).astype(int)

    subs = []
    for r in range(subgrid):
        for c in range(subgrid):
            box = (xs[c], ys[r], xs[c + 1], ys[r + 1])
            sub = crop.crop(box).resize((224, 224))
            subs.append({"row": r, "col": c, "box": box, "crop_224": sub})
    return subs


# ----------------------------
# SAR prompt weighting
# ----------------------------
def label_weight(label: str):
    """
    Human-like and rescue-like cues get higher weight.
    """
    t = label.lower()
    if "empty" in t:
        return 0.0
    if any(k in t for k in ["person", "human", "victim", "man", "woman", "body"]):
        return 1.0
    if any(k in t for k in ["injured", "lying", "sitting", "crouching", "hidden", "partially hidden"]):
        return 0.95
    if any(k in t for k in ["footprint", "footprints", "clothes", "shoe", "backpack"]):
        return 0.60
    return 0.15


# ----------------------------
# Detection
# ----------------------------
def detect_on_image(image: Image.Image, prompts, detector, box_threshold=0.10):
    """
    Returns a list of dicts:
      {'box': [x1,y1,x2,y2], 'score': float, 'label': str, 'evidence': float}
    """
    if detector is None:
        return []

    try:
        preds = detector(
            image.convert("RGB"),
            candidate_labels=prompts,
            threshold=box_threshold,
        )
    except Exception:
        return []

    dets = []
    for p in preds:
        box = p.get("box", {})
        if isinstance(box, dict):
            x1 = float(box.get("xmin", 0.0))
            y1 = float(box.get("ymin", 0.0))
            x2 = float(box.get("xmax", 0.0))
            y2 = float(box.get("ymax", 0.0))
        else:
            x1, y1, x2, y2 = [float(v) for v in box]

        score = float(p.get("score", 0.0))
        label = str(p.get("label", "unknown"))
        evidence = score * label_weight(label)

        dets.append(
            {
                "box": [x1, y1, x2, y2],
                "score": score,
                "label": label,
                "evidence": evidence,
            }
        )
    return dets


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


def rasterize_detections_to_grid(dets, image_size, grid_size):
    """
    Convert detector boxes into a per-cell evidence map.
    """
    w, h = image_size
    xs = np.linspace(0, w, grid_size + 1)
    ys = np.linspace(0, h, grid_size + 1)

    score_map = np.zeros((grid_size, grid_size), dtype=np.float32)
    cell_label = np.full((grid_size, grid_size), "", dtype=object)

    rows = []

    for d in dets:
        box = d["box"]
        ev = d["evidence"]
        label = d["label"]

        for r in range(grid_size):
            for c in range(grid_size):
                cell_box = (xs[c], ys[r], xs[c + 1], ys[r + 1])
                inter = intersection_area(box, cell_box)
                if inter <= 0:
                    continue
                frac = inter / max(box_area(cell_box), 1e-9)
                add = ev * frac
                if add > score_map[r, c]:
                    score_map[r, c] = add
                    cell_label[r, c] = label

        rows.append(
            {
                "label": label,
                "score": d["score"],
                "evidence": ev,
                "box": d["box"],
            }
        )

    if float(score_map.max()) > 0:
        score_map = score_map / float(score_map.max())

    cell_rows = []
    for r in range(grid_size):
        for c in range(grid_size):
            cell_rows.append(
                {
                    "row": r,
                    "col": c,
                    "score": float(score_map[r, c]),
                    "label": str(cell_label[r, c]) if cell_label[r, c] else "background",
                }
            )

    det_df = pd.DataFrame(rows).sort_values("evidence", ascending=False) if rows else pd.DataFrame(
        columns=["label", "score", "evidence", "box"]
    )
    cell_df = pd.DataFrame(cell_rows).sort_values("score", ascending=False)

    return score_map, det_df, cell_df


def build_coarse_map(image, grid_size, prompts, detector, box_threshold=0.10):
    cells, _ = split_image_into_grid(image, grid_size)
    dets = detect_on_image(image, prompts, detector, box_threshold=box_threshold)

    if len(dets) == 0:
        # fallback heuristic if detector fails
        S = np.zeros((grid_size, grid_size), dtype=np.float32)
        for cell in cells:
            arr = np.asarray(cell["crop"].convert("RGB")).astype(np.float32) / 255.0
            gray = arr.mean(axis=2)
            texture = float(gray.std())
            darkness = float(1.0 - gray.mean())
            green = float(arr[:, :, 1].mean())
            S[cell["row"], cell["col"]] = 0.45 * texture + 0.35 * darkness + 0.20 * green
        if float(S.max()) > 0:
            S = S / float(S.max())
        cell_df = pd.DataFrame(
            [{"row": r, "col": c, "score": float(S[r, c]), "label": "heuristic"} for r in range(grid_size) for c in range(grid_size)]
        ).sort_values("score", ascending=False)
        return cells, S, pd.DataFrame(), cell_df

    score_map, det_df, cell_df = rasterize_detections_to_grid(dets, image.size, grid_size)
    return cells, score_map, det_df, cell_df


def refine_top_k_cells(image, cells, score_map, prompts, detector, top_k=4, subgrid=3, box_threshold=0.06):
    """
    Coarse-to-fine refinement: run detector on the top-K cells only.
    """
    flat = [((r, c), float(score_map[r, c])) for r in range(score_map.shape[0]) for c in range(score_map.shape[1])]
    flat.sort(key=lambda x: x[1], reverse=True)
    chosen = [rc for rc, _ in flat[:top_k]]

    refine_rows = []

    for (r, c) in chosen:
        cell = cells[r * score_map.shape[1] + c]
        crop = cell["crop"]
        subs = split_crop_into_subgrid(crop, subgrid)

        local_best = 0.0
        local_label = "background"

        for sub in subs:
            dets = detect_on_image(sub["crop_224"], prompts, detector, box_threshold=box_threshold)
            if len(dets) == 0:
                arr = np.asarray(sub["crop_224"].convert("RGB")).astype(np.float32) / 255.0
                gray = arr.mean(axis=2)
                texture = float(gray.std())
                darkness = float(1.0 - gray.mean())
                green = float(arr[:, :, 1].mean())
                local_score = 0.45 * texture + 0.35 * darkness + 0.20 * green
                local_label = "heuristic_refine"
            else:
                local_score = max(d["evidence"] for d in dets)
                best = max(dets, key=lambda x: x["evidence"])
                local_label = best["label"]

            local_best = max(local_best, float(local_score))

        score_map[r, c] = 0.55 * float(score_map[r, c]) + 0.45 * local_best
        refine_rows.append(
            {
                "row": r,
                "col": c,
                "refined_score": float(local_best),
                "refined_label": local_label,
            }
        )

    if float(score_map.max()) > 0:
        score_map = score_map / float(score_map.max())

    refine_df = pd.DataFrame(refine_rows).sort_values("refined_score", ascending=False) if refine_rows else pd.DataFrame(
        columns=["row", "col", "refined_score", "refined_label"]
    )
    return score_map, refine_df


# ----------------------------
# Planner
# ----------------------------
class SearchPlanner:
    """
    Grounded Coarse-to-Fine Search planner:
    - frontier-guided
    - revisit-penalized
    - lock-on when hotspot is strong
    """

    def __init__(self, size):
        self.size = size
        self.visit_count = np.zeros((size, size), dtype=np.int32)
        self.last_visit = np.full((size, size), -10_000, dtype=np.int32)
        self.step_idx = 0

    def tick(self):
        self.step_idx += 1

    def update_visit(self, pos):
        self.visit_count[pos] += 1
        self.last_visit[pos] = self.step_idx

    def neighbors(self, r, c):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                yield nr, nc

    def frontier_cells(self, explored):
        frontiers = []
        for r in range(self.size):
            for c in range(self.size):
                if explored[r, c] == 0:
                    continue
                if any(explored[nr, nc] == 0 for nr, nc in self.neighbors(r, c)):
                    frontiers.append((r, c))
        return frontiers

    def score_goal(self, cell, score_map, explored, current_pos, battery):
        r, c = cell
        if self.visit_count[r, c] > 3:
            return -1e9

        p = float(score_map[r, c])
        dist = abs(current_pos[0] - r) + abs(current_pos[1] - c)
        visits = int(self.visit_count[r, c])
        novelty = 1.0 if explored[r, c] == 0 else 0.0
        age = self.step_idx - int(self.last_visit[r, c])
        recency = np.exp(-max(age, 0) / 5.0)
        urgency = 1.0 + (1.0 - battery)

        return (
            2.4 * p +
            1.2 * novelty +
            0.5 * urgency -
            0.85 * dist -
            2.8 * visits -
            0.7 * recency
        )

    def choose_goal(self, score_map, explored, current_pos, battery, lock_threshold):
        max_cell = tuple(np.unravel_index(np.argmax(score_map), score_map.shape))
        max_score = float(score_map[max_cell])

        if max_score >= lock_threshold:
            return max_cell, True, max_score

        candidates = self.frontier_cells(explored)
        if not candidates:
            candidates = [tuple(x) for x in zip(*np.where(explored == 0))]
            if not candidates:
                return current_pos, False, max_score

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

    def escape_move(self, explored, current_pos):
        r, c = current_pos
        best = current_pos
        best_score = -1e18

        for nr, nc in self.neighbors(r, c):
            s = (1.0 if explored[nr, nc] == 0 else 0.0) - 0.4 * self.visit_count[nr, nc]
            if s > best_score:
                best_score = s
                best = (nr, nc)

        return best


# ----------------------------
# Visualization
# ----------------------------
def overlay_on_original(image, score_map, grid_size, drone_pos=None, path=None, alpha=0.38, show_scores=True):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    w, h = image.size
    xs = np.linspace(0, w, grid_size + 1)
    ys = np.linspace(0, h, grid_size + 1)

    cmap = plt.cm.inferno
    vmax = max(float(score_map.max()), 1e-9)

    for r in range(grid_size):
        for c in range(grid_size):
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

            if show_scores and grid_size <= 12:
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
        cx, cy = [], []
        for rr, cc in path:
            L, R = xs[cc], xs[cc + 1]
            T, B = ys[rr], ys[rr + 1]
            cx.append((L + R) / 2)
            cy.append((T + B) / 2)
        ax.plot(cx, cy, color="cyan", linewidth=2)

    if drone_pos is not None:
        rr, cc = drone_pos
        L, R = xs[cc], xs[cc + 1]
        T, B = ys[rr], ys[rr + 1]
        ax.scatter([(L + R) / 2], [(T + B) / 2], c="deepskyblue", s=120, edgecolors="black", linewidths=1.0)

    ax.set_axis_off()
    fig.tight_layout()
    return fig


def plot_heatmap(score_map, drone_pos=None, path=None, title="Confidence heatmap"):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    im = ax.imshow(score_map, cmap="inferno", vmin=0, vmax=1)
    if path:
        ys = [p[1] for p in path]
        xs = [p[0] for p in path]
        ax.plot(ys, xs, color="cyan", linewidth=2)
    if drone_pos is not None:
        ax.scatter([drone_pos[1]], [drone_pos[0]], c="deepskyblue", s=120, edgecolors="black", linewidths=1.0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


# ----------------------------
# Search loop
# ----------------------------
def run_search(
    image,
    grid_size,
    camera_radius,
    max_steps,
    prompts,
    model_id,
    top_k,
    refine_grid,
    box_threshold,
    lock_threshold,
    battery_start,
):
    detector, ok = load_detector(model_id)
    cells, score_map, det_df, cell_df = build_coarse_map(image, grid_size, prompts, detector, box_threshold=box_threshold)

    # Coarse -> fine refinement
    score_map, refine_df = refine_top_k_cells(
        image=image,
        cells=cells,
        score_map=score_map,
        prompts=prompts,
        detector=detector,
        top_k=top_k,
        subgrid=refine_grid,
        box_threshold=max(0.04, box_threshold * 0.75),
    )

    if len(refine_df) > 0:
        cell_df = pd.concat(
            [
                cell_df,
                refine_df.rename(columns={"refined_score": "score", "refined_label": "label"}),
            ],
            ignore_index=True,
        )

    planner = SearchPlanner(grid_size)
    explored = np.zeros((grid_size, grid_size), dtype=np.float32)

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

        # simulate camera footprint
        for rr in range(r - camera_radius, r + camera_radius + 1):
            for cc in range(c - camera_radius, c + camera_radius + 1):
                if 0 <= rr < grid_size and 0 <= cc < grid_size:
                    explored[rr, cc] = 1.0
                    # dead-zone attenuation so revisits become less attractive
                    score_map[rr, cc] *= 0.35

        if float(score_map.max()) > 0:
            score_map = score_map / float(score_map.max())

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

        if planner.visit_count[nxt] > 3:
            nxt = planner.escape_move(explored, current)

        if nxt == current:
            break

        current = nxt
        path.append(current)

        if locked and current == lock_cell:
            break

        if explored.sum() >= grid_size * grid_size:
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


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="UAV SAR Search", layout="wide")
st.title("UAV SAR — Grounded Coarse-to-Fine Search")

st.write(
    "This demo uses a single jungle/aerial image, splits it into original-location cells, "
    "runs a text-conditioned detector, refines the top hotspots, and guides a simulated drone step-by-step."
)

mode = st.sidebar.radio("Input mode", ["Random from database", "Upload image"])
default_model = "IDEA-Research/grounding-dino-tiny" if not torch.cuda.is_available() else "IDEA-Research/grounding-dino-base"
model_id = st.sidebar.selectbox(
    "Detector model",
    ["IDEA-Research/grounding-dino-tiny", "IDEA-Research/grounding-dino-base"],
    index=0 if default_model.endswith("tiny") else 1,
)

grid_size = st.sidebar.slider("Grid size", 7, 25, DEFAULT_GRID_SIZE)
camera_radius = st.sidebar.slider("Camera radius", 1, 3, DEFAULT_CAMERA_RADIUS)
max_steps = st.sidebar.slider("Max steps", 20, 400, DEFAULT_MAX_STEPS)
top_k = st.sidebar.slider("Top-K refine", 1, 10, DEFAULT_TOPK_REFINE)
refine_grid = st.sidebar.slider("Local refinement grid", 2, 6, DEFAULT_REFINEMENT_GRID)
box_threshold = st.sidebar.slider("Detector threshold", 0.01, 0.50, 0.08, 0.01)
lock_threshold = st.sidebar.slider("Lock threshold", 0.10, 1.00, 0.55, 0.01)
battery_start = st.sidebar.slider("Initial battery", 0.1, 1.0, 1.0, 0.05)
db_dir = st.sidebar.text_input("Random database folder", DEFAULT_DB_DIR)

prompt_text = st.sidebar.text_area(
    "Prompts (one per line)",
    value="\n".join(DEFAULT_PROMPTS),
    height=240,
)

run_button = st.sidebar.button("Run search")

selected_image = None

if mode == "Random from database":
    images = list_images(db_dir)
    if len(images) == 0:
        st.warning(f"No images found in `{db_dir}`. Upload one or add jungle images there.")
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
    st.info("Choose a random image from your folder or upload one.")
    st.stop()

st.image(selected_image, caption="Input image", use_container_width=True)

if run_button or "result" in st.session_state:
    prompts = parse_prompts(prompt_text)
    with st.spinner("Running detector + coarse-to-fine search..."):
        result = run_search(
            image=selected_image,
            grid_size=grid_size,
            camera_radius=camera_radius,
            max_steps=max_steps,
            prompts=prompts,
            model_id=model_id,
            top_k=top_k,
            refine_grid=refine_grid,
            box_threshold=box_threshold,
            lock_threshold=lock_threshold,
            battery_start=battery_start,
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

    c1, c2 = st.columns(2)

    with c1:
        fig1 = overlay_on_original(
            image=selected_image,
            score_map=score_t,
            grid_size=grid_size,
            drone_pos=drone_t,
            path=path_t,
            alpha=0.38,
            show_scores=(grid_size <= 12),
        )
        st.pyplot(fig1, clear_figure=True)

    with c2:
        fig2 = plot_heatmap(
            score_t,
            drone_pos=drone_t,
            path=path_t,
            title="Confidence heatmap",
        )
        st.pyplot(fig2, clear_figure=True)

    st.subheader("Detector evidence")
    if len(det_df) > 0:
        st.dataframe(det_df.head(50), use_container_width=True)
    else:
        st.write("No detections from the detector; using fallback heuristic.")

    st.subheader("Per-cell scores")
    st.dataframe(cell_df.head(50), use_container_width=True)

    st.subheader("Tracking over time")
    if len(cell_df) > 0 and len(history) > 0:
        tracked_options = [tuple(x) for x in cell_df[["row", "col"]].head(20).drop_duplicates().to_numpy()]
        if len(tracked_options) == 0:
            tracked_options = [(0, 0)]
        track_cell = st.selectbox("Cell to track", tracked_options)
        rr, cc = track_cell
        series = [h[rr, cc] for h in history]
        st.line_chart(pd.DataFrame({"score": series}))

    if len(battery_history) > 0:
        st.subheader("Battery over time")
        st.line_chart(pd.DataFrame({"battery": battery_history}))

    st.write(f"Lock-on: {result['locked']}")
    st.write(f"Target hotspot: {result['lock_cell']}")