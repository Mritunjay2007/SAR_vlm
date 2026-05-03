# perception/vlm_clip.py
import numpy as np
import torch
from PIL import Image

try:
    from transformers import CLIPModel, CLIPProcessor
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False


class CLIPVLM:
    def __init__(self, prompts=None):
        self.prompts = prompts or [
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
            "empty forest background",
        ]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enabled = False
        self.model = None
        self.processor = None

        if TRANSFORMERS_AVAILABLE:
            try:
                name = "openai/clip-vit-base-patch32"
                self.model = CLIPModel.from_pretrained(name).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(name)
                self.model.eval()
                self.enabled = True
            except Exception:
                self.enabled = False

    def predict(self, image):
        if not self.enabled:
            return self._mock_predict(image)

        with torch.no_grad():
            inputs = self.processor(
                text=self.prompts,
                images=image.convert("RGB"),
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=-1).squeeze(0).cpu().numpy()
            return probs

    def _mock_predict(self, image):
        arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
        gray = arr.mean(axis=2)
        texture = float(gray.std())
        green = float(arr[:, :, 1].mean())
        brownish = float(0.4 * arr[:, :, 0].mean() + 0.4 * arr[:, :, 1].mean() + 0.2 * arr[:, :, 2].mean())

        raw = np.array([
            0.35 * texture + 0.20 * green,
            0.30 * texture + 0.10 * green,
            0.10 * texture + 0.25 * brownish,
            0.08 * texture + 0.18 * brownish,
            0.55 * (1.0 - texture) + 0.10 * (1.0 - green),
        ], dtype=np.float32)

        raw = np.maximum(raw, 1e-6)
        return raw / raw.sum()