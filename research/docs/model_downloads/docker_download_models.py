#!/usr/bin/env python3
"""
Docker-friendly model download script for benchmark container.
Run this inside the container to pre-download all models.
"""

import os
import sys
import logging
from huggingface_hub import snapshot_download, login
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Fixed model list with correct names
MODELS = [
    "stabilityai/sdxl-turbo",                          # Public - SDXL Turbo
    "black-forest-labs/FLUX.1-dev",                    # GATED - Accept at https://huggingface.co/black-forest-labs/FLUX.1-dev
    "SG161222/RealVisXL_V4.0",                        # Public - RealVisXL V4
    "stabilityai/stable-diffusion-3-medium",           # GATED - Accept at https://huggingface.co/stabilityai/stable-diffusion-3-medium  
    "stabilityai/stable-diffusion-xl-base-1.0",        # Public - SDXL Base
    "HiDream-ai/HiDream-I1-Full",                     # Public - HiDream
    "DeepFloyd/IF-I-XL-v1.0",                         # GATED - Accept at https://huggingface.co/DeepFloyd/IF-I-XL-v1.0
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",       # Public - For CLIP scores
]

GATED_MODELS = {
    "black-forest-labs/FLUX.1-dev": "https://huggingface.co/black-forest-labs/FLUX.1-dev",
    "stabilityai/stable-diffusion-3-medium": "https://huggingface.co/stabilityai/stable-diffusion-3-medium",
    "DeepFloyd/IF-I-XL-v1.0": "https://huggingface.co/DeepFloyd/IF-I-XL-v1.0",
}

# Common model file patterns
ALLOW_PATTERNS = [
    "*.safetensors", "*.bin", "*.pt", "*.pth",
    "*.json", "*.txt", "*.yaml", "*.yml",
    "*model_index.json", "*config*.json",
    "*tokenizer*", "*scheduler*", "*vae*", "*text_encoder*"
]

def setup_auth():
    """Setup HuggingFace authentication."""
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    
    if token:
        try:
            login(token=token, add_to_git_credential=False)
            logger.info(f"✅ Authenticated with HuggingFace (token: {token[:8]}...)")
            return True
        except Exception as e:
            logger.error(f"Failed to authenticate: {e}")
            return False
    else:
        logger.warning("No HuggingFace token found in environment")
        logger.warning("Set HUGGINGFACE_HUB_TOKEN to download gated models")
        return False


def download_models():
    """Download all models with proper error handling."""
    cache_dir = os.environ.get("HF_HOME", "/workspace/hf_cache")
    logger.info(f"Cache directory: {cache_dir}")
    
    success = []
    failed = []
    
    for model_id in MODELS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading: {model_id}")
        
        try:
            path = snapshot_download(
                model_id,
                cache_dir=cache_dir,
                allow_patterns=ALLOW_PATTERNS,
                ignore_patterns=["*.md", "*.gitattributes"],
                token=os.environ.get("HUGGINGFACE_HUB_TOKEN"),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            logger.info(f"✅ Success: {path}")
            success.append(model_id)
            
        except GatedRepoError:
            logger.error(f"❌ GATED MODEL: {model_id}")
            if model_id in GATED_MODELS:
                logger.error(f"   Accept license at: {GATED_MODELS[model_id]}")
            logger.error("   Then wait 5-10 minutes and retry")
            failed.append(model_id)
            
        except RepositoryNotFoundError:
            logger.error(f"❌ NOT FOUND: {model_id}")
            failed.append(model_id)
            
        except Exception as e:
            logger.error(f"❌ ERROR: {e}")
            failed.append(model_id)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("DOWNLOAD SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Success: {len(success)}/{len(MODELS)}")
    for m in success:
        logger.info(f"   • {m}")
    
    if failed:
        logger.info(f"\n❌ Failed: {len(failed)}")
        for m in failed:
            logger.info(f"   • {m}")
            if m in GATED_MODELS:
                logger.info(f"     → Accept license: {GATED_MODELS[m]}")
    
    return len(failed) == 0


def main():
    logger.info("🚀 Docker Model Download Script")
    logger.info("="*60)
    
    # Check if running in Docker
    if os.path.exists("/.dockerenv"):
        logger.info("✅ Running in Docker container")
    else:
        logger.warning("⚠️  Not running in Docker - this script is optimized for containers")
    
    # Setup authentication
    auth_ok = setup_auth()
    
    if not auth_ok:
        logger.warning("\n⚠️  No authentication - only public models will download")
        logger.warning("\nGATED MODELS requiring license acceptance:")
        for model, url in GATED_MODELS.items():
            logger.warning(f"  • {model}")
            logger.warning(f"    Accept at: {url}")
        
        response = input("\nContinue without auth? (y/n): ")
        if response.lower() != 'y':
            logger.info("Exiting. Please set HUGGINGFACE_HUB_TOKEN and retry.")
            return 1
    
    # Download models
    all_success = download_models()
    
    if all_success:
        logger.info("\n✨ All models downloaded successfully!")
        logger.info("You can now run the benchmark.")
        return 0
    else:
        logger.warning("\n⚠️  Some models failed to download")
        logger.warning("Please accept licenses for gated models and retry")
        return 1


if __name__ == "__main__":
    sys.exit(main())
