"""Image encoders for vision heads, in-process with ``transformers``.

One implementation serves both training (``train/embed_images.py``) and the engine, so an
image is embedded the same way in both:

* ``QwenVL``: a user turn holding the image and an instruction, rendered with the chat
  template (no generation prompt); the final-norm hidden state of the last token.
* ``Siglip``: the pooled image embedding of a SigLIP-family model (MedSigLIP).

``ImageEncoders`` loads them lazily for the engine, keyed by the recipe a head's
``cfg["state_embedding"]`` records (encoder, instruction, max_pixels, dtype).
"""
from __future__ import annotations

import os
import threading

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402

DTYPES = ("fp16", "bf16", "fp32")


def image_device() -> str:
    """The image encoder is the large model, so it takes the accelerator: cuda, then mps, then cpu."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _dtype(name: str):
    import torch
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


class QwenVL:
    """Last-token final hidden state of a Qwen3-VL chat turn (image + instruction)."""

    def __init__(self, name: str, device: str, dtype, max_pixels: int, instruction: str):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(name)
        self.proc.image_processor.size = {"longest_edge": max_pixels,
                                          "shortest_edge": self.proc.image_processor.size["shortest_edge"]}
        self.proc.tokenizer.padding_side = "right"
        model = AutoModelForImageTextToText.from_pretrained(name, dtype=dtype, device_map=device)
        self.model = model.model.eval()                 # backbone only: no lm_head over the vocab
        self.hidden = model.config.text_config.hidden_size
        self.device, self.instruction = device, instruction

    def __call__(self, images):
        import torch
        msgs = [[{"role": "user", "content": [{"type": "image", "image": im},
                                              {"type": "text", "text": self.instruction}]}] for im in images]
        with torch.no_grad():
            x = self.proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt",
                                              add_generation_prompt=False,
                                              processor_kwargs={"padding": True}).to(self.device)
            h = self.model(**x).last_hidden_state
            last = x["attention_mask"].sum(1) - 1              # right padding: last real token
            return h[torch.arange(len(images), device=h.device), last].float()


class Siglip:
    """Pooled image embedding of a SigLIP-family model (MedSigLIP)."""

    def __init__(self, name: str, device: str, dtype):
        from transformers import AutoImageProcessor, AutoModel
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, dtype=dtype).to(device).eval()
        cfg = self.model.config
        self.hidden = getattr(cfg, "projection_dim", None) or cfg.vision_config.hidden_size
        self.device, self.dtype = device, dtype

    def __call__(self, images):
        import torch
        with torch.no_grad():
            px = self.proc(images=images, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
            out = self.model.get_image_features(pixel_values=px)
            return (out if torch.is_tensor(out) else out.pooler_output).float()


def load_encoder(name: str, device: str, dtype: str = "fp16", max_pixels: int = 1024 * 1024,
                 instruction: str = ""):
    """SigLIP-family ids get ``Siglip``, everything else ``QwenVL``."""
    if "siglip" in name.lower():
        return Siglip(name, device, _dtype(dtype))
    return QwenVL(name, device, _dtype(dtype), max_pixels, instruction)


class ImageEncoders:
    """Image encoders loaded on first use and kept; ``embed`` -> L2-normalised [n, hidden] float32."""

    def __init__(self, device: str):
        self.device = device
        self._encoders: dict[tuple, object] = {}
        self._lock = threading.Lock()

    @staticmethod
    def recipe(cfg: dict) -> tuple:
        e = cfg.get("state_embedding") or {}
        return (cfg["state_encoder"], e.get("dtype", "fp16"), e.get("max_pixels"), e.get("instruction"))

    def embed(self, cfg: dict, images) -> np.ndarray:
        import torch
        key = self.recipe(cfg)
        with self._lock:          # one forward pass at a time: the encoder is the big allocation
            enc = self._encoders.get(key)
            if enc is None:
                name, dtype, max_pixels, instruction = key
                enc = self._encoders[key] = load_encoder(name, self.device, dtype, max_pixels or 1024 * 1024,
                                                         instruction or "")
            if enc.hidden != cfg["hidden_size"]:
                raise ValueError(f"{key[0]} gives {enc.hidden}-d embeddings, the head expects {cfg['hidden_size']}")
            z = torch.nn.functional.normalize(enc(images), dim=-1)
            if str(self.device).startswith("mps"):
                torch.mps.empty_cache()
            return z.cpu().numpy()
