"""Model registry: every model exposes `embed(pil_images) -> np.ndarray [N, D]` (L2-normalised).

Backends
    openclip : open_clip models (OpenAI CLIP, SigLIP, Marqo fashion models via hf-hub)
    hf       : Hugging Face transformers (FashionCLIP, DINOv2, SigLIP2)
    random   : deterministic random projection of pixels, for pipeline tests only

Set RRB_OFFLINE_TEST=1 to build tiny randomly-initialised versions of each backend without
any download. This exercises the exact preprocessing / forward / pooling code paths and is
used by tests/ only. Results from offline-test models are meaningless.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

OFFLINE_TEST = os.environ.get("RRB_OFFLINE_TEST", "0") == "1"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str          # openclip | hf | random
    arch: str             # open_clip model name or HF repo id
    pretrained: str = ""  # open_clip pretrained tag (ignored for hf)
    family: str = ""      # general | fashion | vision-only
    notes: str = ""


REGISTRY: dict[str, ModelSpec] = {
    # General-purpose vision-language models
    "clip-vit-b32": ModelSpec("clip-vit-b32", "openclip", "ViT-B-32", "openai", "general", "OpenAI CLIP ViT-B/32"),
    "clip-vit-l14": ModelSpec("clip-vit-l14", "openclip", "ViT-L-14", "openai", "general", "OpenAI CLIP ViT-L/14"),
    "siglip-b16": ModelSpec("siglip-b16", "openclip", "ViT-B-16-SigLIP", "webli", "general", "Google SigLIP ViT-B/16 (WebLI)"),
    "siglip2-b16": ModelSpec("siglip2-b16", "hf", "google/siglip2-base-patch16-224", "", "general", "Google SigLIP 2 base"),
    # Fashion-specialised vision-language models
    "fashionclip": ModelSpec("fashionclip", "hf", "patrickjohncyh/fashion-clip", "", "fashion", "FashionCLIP (Chia et al. 2022)"),
    "marqo-fashionclip": ModelSpec("marqo-fashionclip", "openclip", "hf-hub:Marqo/marqo-fashionCLIP", "", "fashion", "Marqo-FashionCLIP"),
    "marqo-fashionsiglip": ModelSpec("marqo-fashionsiglip", "openclip", "hf-hub:Marqo/marqo-fashionSigLIP", "", "fashion", "Marqo-FashionSigLIP"),
    # Vision-only baseline
    "dinov2-b": ModelSpec("dinov2-b", "hf", "facebook/dinov2-base", "", "vision-only", "DINOv2 ViT-B/14 (CLS token)"),
    # Test-only
    "random": ModelSpec("random", "random", "random", "", "test", "Random projection of pixels; sanity check only"),
}

DEFAULT_MODELS = [
    "clip-vit-b32", "clip-vit-l14", "siglip-b16",
    "fashionclip", "marqo-fashionclip", "marqo-fashionsiglip",
    "dinov2-b",
]


def pick_device(requested: str = "auto") -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    """Wraps a loaded model with a `preprocess(PIL) -> tensor` and `forward(batch) -> features`."""

    def __init__(self, spec: ModelSpec, preprocess: Callable, forward: Callable, device: str, dim: int | None):
        self.spec = spec
        self.preprocess = preprocess
        self._forward = forward
        self.device = device
        self.dim = dim

    def embed_batch(self, batch) -> np.ndarray:
        import torch
        with torch.no_grad():
            feats = self._forward(batch.to(self.device))
            feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        return feats.cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------------------
# open_clip backend
# --------------------------------------------------------------------------------------

def _load_openclip(spec: ModelSpec, device: str) -> Embedder:
    import open_clip
    import torch

    if OFFLINE_TEST:
        # Tiny random-weight ViT-B-32 with the standard preprocessing; no download.
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
    elif spec.arch.startswith("hf-hub:"):
        model, _, preprocess = open_clip.create_model_and_transforms(spec.arch)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(spec.arch, pretrained=spec.pretrained)
    model = model.to(device).eval()
    if device == "cuda":
        model = model.half()

    def forward(x):
        if device == "cuda":
            x = x.half()
        return model.encode_image(x)

    dim = getattr(model, "embed_dim", None)
    return Embedder(spec, preprocess, forward, device, dim)


# --------------------------------------------------------------------------------------
# Hugging Face backend
# --------------------------------------------------------------------------------------

def _hf_preprocess_from_processor(processor):
    def preprocess(img):
        out = processor(images=img, return_tensors="pt")
        return out["pixel_values"][0]
    return preprocess


def _load_hf(spec: ModelSpec, device: str) -> Embedder:
    import torch
    from transformers import AutoModel, AutoImageProcessor

    repo = spec.arch
    if OFFLINE_TEST:
        # Tiny random-weight models with default processors; no download.
        if "dinov2" in repo:
            from transformers import Dinov2Config, Dinov2Model, BitImageProcessor
            model = Dinov2Model(Dinov2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64, image_size=224, patch_size=16))
            processor = BitImageProcessor(size={"shortest_edge": 224}, crop_size={"height": 224, "width": 224})
        else:
            from transformers import CLIPConfig, CLIPModel, CLIPImageProcessor, CLIPTextConfig, CLIPVisionConfig
            cfg = CLIPConfig(text_config=CLIPTextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2).to_dict(),
                             vision_config=CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, image_size=224, patch_size=32).to_dict(),
                             projection_dim=16)
            model = CLIPModel(cfg)
            processor = CLIPImageProcessor()
    else:
        model = AutoModel.from_pretrained(repo)
        processor = AutoImageProcessor.from_pretrained(repo)

    model = model.to(device).eval()
    if device == "cuda":
        model = model.half()

    is_clip_like = hasattr(model, "get_image_features")

    def _to_tensor(out):
        """transformers >= 5 returns output objects from get_image_features; older versions return tensors."""
        if torch.is_tensor(out):
            return out
        for key in ("image_embeds", "pooler_output"):
            v = getattr(out, key, None)
            if v is not None:
                return v
        return out.last_hidden_state[:, 0]   # CLS token fallback

    def forward(x):
        if device == "cuda":
            x = x.half()
        if is_clip_like:
            return _to_tensor(model.get_image_features(pixel_values=x))
        return _to_tensor(model(pixel_values=x))   # DINOv2: pooler_output = layer-normed CLS token

    dim = None
    cfg = model.config
    if hasattr(cfg, "projection_dim") and is_clip_like:
        dim = cfg.projection_dim
    elif hasattr(cfg, "hidden_size"):
        dim = cfg.hidden_size
    return Embedder(spec, _hf_preprocess_from_processor(processor), forward, device, dim)


# --------------------------------------------------------------------------------------
# Random backend (tests)
# --------------------------------------------------------------------------------------

def _load_random(spec: ModelSpec, device: str) -> Embedder:
    import torch
    from torchvision import transforms

    rng = np.random.default_rng(0)
    proj = torch.tensor(rng.standard_normal((3 * 16 * 16, 64)).astype(np.float32))
    preprocess = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])

    def forward(x):
        flat = x.reshape(x.shape[0], -1)
        return flat @ proj.to(x.device)

    return Embedder(spec, preprocess, forward, "cpu", 64)


LOADERS = {"openclip": _load_openclip, "hf": _load_hf, "random": _load_random}


def load_model(name: str, device: str = "auto") -> Embedder:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(REGISTRY)}")
    spec = REGISTRY[name]
    dev = "cpu" if spec.backend == "random" else pick_device(device)
    return LOADERS[spec.backend](spec, dev)
