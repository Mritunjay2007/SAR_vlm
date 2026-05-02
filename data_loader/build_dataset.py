import os
import shutil
import random

# ---------- PATHS ----------
SARD_PATH = "data_dump/sard/search-and-rescue/train/images"
VISDRONE_PATH = "data_dump/VisDrone2019-DET-train/VisDrone2019-DET-train/images"

OUTPUT_BASE = "dataset"

FOREST_OUT = os.path.join(OUTPUT_BASE, "forest")
HUMAN_OUT = os.path.join(OUTPUT_BASE, "human")
MIXED_OUT = os.path.join(OUTPUT_BASE, "mixed")

# ---------- CREATE FOLDERS ----------
os.makedirs(FOREST_OUT, exist_ok=True)
os.makedirs(HUMAN_OUT, exist_ok=True)
os.makedirs(MIXED_OUT, exist_ok=True)


# ---------- COPY HUMAN IMAGES (SARD) ----------
def extract_human_images():
    print("Extracting human images from SARD...")

    files = os.listdir(SARD_PATH)

    for f in files[:500]:  # limit for now
        src = os.path.join(SARD_PATH, f)
        dst = os.path.join(HUMAN_OUT, f)

        if os.path.isfile(src):
            shutil.copy(src, dst)

    print("Done: Human images")


# ---------- COPY FOREST IMAGES (VISDRONE) ----------
def extract_forest_images():
    print("Extracting aerial images from VisDrone...")

    files = os.listdir(VISDRONE_PATH)

    for f in files[:500]:
        src = os.path.join(VISDRONE_PATH, f)
        dst = os.path.join(FOREST_OUT, f)

        if os.path.isfile(src):
            shutil.copy(src, dst)

    print("Done: Forest images")


# ---------- CREATE MIXED GRID POOL ----------
def create_mixed_pool():
    print("Creating mixed dataset...")

    human_imgs = os.listdir(HUMAN_OUT)
    forest_imgs = os.listdir(FOREST_OUT)

    all_imgs = human_imgs + forest_imgs

    for i, f in enumerate(all_imgs):
        src = (
            os.path.join(HUMAN_OUT, f)
            if f in human_imgs
            else os.path.join(FOREST_OUT, f)
        )

        dst = os.path.join(MIXED_OUT, f"{i}.jpg")

        if os.path.isfile(src):
            shutil.copy(src, dst)

    print("Done: Mixed dataset")


if __name__ == "__main__":
    extract_human_images()
    extract_forest_images()
    create_mixed_pool()