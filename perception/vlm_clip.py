import torch
import clip
from PIL import Image

class CLIPVLM:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.prompts = [
            "a person in forest",
            "footprints on ground",
            "clothes on ground",
            "empty forest"
        ]

        self.text_tokens = clip.tokenize(self.prompts).to(self.device)

    def predict(self, image_path):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_feat = self.model.encode_image(image)
            txt_feat = self.model.encode_text(self.text_tokens)

            similarity = (img_feat @ txt_feat.T).softmax(dim=-1)

        return similarity.cpu().numpy()[0]