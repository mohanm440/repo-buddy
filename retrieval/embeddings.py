import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def encode(self, texts, batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[i:i + batch_size], padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**batch).last_hidden_state
            emb = hidden[:, 0]  # CLS pooling
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out.append(emb.cpu().numpy())
        return np.vstack(out).astype("float32")
