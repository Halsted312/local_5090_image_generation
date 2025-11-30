"""
CPU-only CLIP and aesthetic scoring models.

These models run on CPU to avoid GPU contention with the main inference pipeline.
Thread count is configurable via METRICS_CPU_THREADS environment variable.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

# Limit PyTorch CPU threads to avoid contention
# Only set if not already configured (avoids RuntimeError on reimport)
NUM_CPU_THREADS = int(os.environ.get("METRICS_CPU_THREADS", "26"))
try:
    torch.set_num_threads(NUM_CPU_THREADS)
except RuntimeError:
    pass  # Already set

try:
    torch.set_num_interop_threads(2)
except RuntimeError:
    pass  # Already set


@dataclass
class MetricsConfig:
    """Configuration for metrics models."""
    models_dir: Path = None
    device: str = "cpu"

    def __post_init__(self):
        if self.models_dir is None:
            # Default to HF cache
            self.models_dir = Path.home() / ".cache" / "huggingface" / "hub"
        self.models_dir = Path(self.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)


class ClipScorer:
    """
    CLIP ViT-L/14 scorer for image-text similarity.

    Uses OpenCLIP with OpenAI's pretrained weights.
    All inputs are resized to 224x224 internally.
    """

    def __init__(self, config: MetricsConfig):
        import open_clip

        self.device = config.device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai', device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
        self.model.eval()

    @torch.no_grad()
    def score(self, image: Image.Image, prompt: str) -> float:
        """
        Compute CLIP similarity score between image and text.

        Args:
            image: PIL Image (any size, will be preprocessed)
            prompt: Text prompt to compare against

        Returns:
            Cosine similarity score (typically 0.15-0.35 for good matches)
        """
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        text_tokens = self.tokenizer([prompt]).to(self.device)

        img_features = self.model.encode_image(img_tensor)
        text_features = self.model.encode_text(text_tokens)

        # L2 normalize
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Cosine similarity
        return (img_features @ text_features.T).item()

    def unload(self):
        """Unload model and free memory."""
        import gc
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                self.model.to("cpu")
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass
        del self.model
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class AestheticScorer:
    """
    Aesthetic predictor using LAION's aesthetic model.

    This is an MLP trained on CLIP ViT-L/14 embeddings to predict
    aesthetic scores from 1-10.
    """

    def __init__(self, config: MetricsConfig):
        import open_clip

        self.device = config.device

        # Load CLIP model for embeddings
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai', device=self.device
        )
        self.clip_model.eval()

        # Load aesthetic predictor MLP
        self.aesthetic_mlp = self._load_aesthetic_model(config.models_dir)

    def _load_aesthetic_model(self, models_dir: Path) -> torch.nn.Module:
        """Load the LAION aesthetic predictor MLP."""
        import torch.nn as nn

        # Define MLP architecture (matches LAION aesthetic predictor v2)
        class AestheticMLP(nn.Module):
            def __init__(self, input_size: int = 768):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, 1024),
                    nn.Dropout(0.2),
                    nn.Linear(1024, 128),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.Dropout(0.1),
                    nn.Linear(64, 16),
                    nn.Linear(16, 1),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.layers(x)

        mlp = AestheticMLP(input_size=768)

        # Try to load pretrained weights
        weights_path = models_dir / "sac+logos+ava1-l14-linearMSE.pth"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            mlp.load_state_dict(state_dict)
        else:
            # Download from GitHub (LAION improved aesthetic predictor)
            import urllib.request

            url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
            download_path = models_dir / "sac+logos+ava1-l14-linearMSE.pth"

            try:
                models_dir.mkdir(parents=True, exist_ok=True)
                print(f"Downloading aesthetic model from GitHub...")
                urllib.request.urlretrieve(url, download_path)
                state_dict = torch.load(download_path, map_location=self.device, weights_only=True)
                mlp.load_state_dict(state_dict)
            except Exception as e:
                print(f"Warning: Could not load aesthetic model weights: {e}")
                print("Using randomly initialized weights (scores will be meaningless)")

        mlp.to(self.device)
        mlp.eval()
        return mlp

    @torch.no_grad()
    def score(self, image: Image.Image) -> float:
        """
        Predict aesthetic score for an image.

        Args:
            image: PIL Image (any size, will be preprocessed)

        Returns:
            Aesthetic score from 1-10 (higher = more aesthetic)
        """
        # Get CLIP image embedding
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        img_features = self.clip_model.encode_image(img_tensor)

        # L2 normalize (same as CLIP training)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        # Predict aesthetic score
        score = self.aesthetic_mlp(img_features.float())
        return score.item()

    def unload(self):
        """Unload models and free memory."""
        import gc
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                self.clip_model.to("cpu")
                self.aesthetic_mlp.to("cpu")
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass
        del self.clip_model
        del self.aesthetic_mlp
        self.clip_model = None
        self.aesthetic_mlp = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MetricsScorer:
    """
    Combined scorer for CLIP and aesthetic metrics.

    Shares the CLIP model between ClipScorer and AestheticScorer
    to reduce memory usage and initialization time.
    """

    def __init__(self, config: Optional[MetricsConfig] = None):
        if config is None:
            config = MetricsConfig(models_dir=Path("/models"))

        self.config = config
        self._clip_scorer: Optional[ClipScorer] = None
        self._aesthetic_scorer: Optional[AestheticScorer] = None

    @property
    def clip_scorer(self) -> ClipScorer:
        if self._clip_scorer is None:
            self._clip_scorer = ClipScorer(self.config)
        return self._clip_scorer

    @property
    def aesthetic_scorer(self) -> AestheticScorer:
        if self._aesthetic_scorer is None:
            self._aesthetic_scorer = AestheticScorer(self.config)
        return self._aesthetic_scorer

    def score_image(self, image: Image.Image, prompt: str) -> dict:
        """
        Score an image for both CLIP similarity and aesthetic quality.

        Args:
            image: PIL Image
            prompt: Text prompt for CLIP scoring

        Returns:
            Dictionary with 'clip_score' and 'aesthetic_score'
        """
        return {
            "clip_score": self.clip_scorer.score(image, prompt),
            "aesthetic_score": self.aesthetic_scorer.score(image),
        }
