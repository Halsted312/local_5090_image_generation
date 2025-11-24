#!/usr/bin/env python3
"""
Setup script for downloading and caching all benchmark models.
Handles gated models that require license acceptance.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, login
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError


# Models configuration with gating status
MODELS = {
    "stabilityai/sdxl-turbo": {
        "name": "SDXL-Turbo",
        "gated": False,
        "license_url": None
    },
    "black-forest-labs/FLUX.1-dev": {
        "name": "FLUX.1-dev", 
        "gated": True,
        "license_url": "https://huggingface.co/black-forest-labs/FLUX.1-dev"
    },
    "SG161222/RealVisXL_V4.0": {  # Fixed model name
        "name": "RealVisXL V4.0",
        "gated": False,
        "license_url": None
    },
    "stabilityai/stable-diffusion-3-medium": {
        "name": "SD3 Medium",
        "gated": True,
        "license_url": "https://huggingface.co/stabilityai/stable-diffusion-3-medium"
    },
    "stabilityai/stable-diffusion-xl-base-1.0": {
        "name": "SDXL 1.0",
        "gated": False,
        "license_url": None
    },
    "HiDream-ai/HiDream-I1-Full": {
        "name": "HiDream-I1",
        "gated": False,  # Currently public
        "license_url": None
    },
    "DeepFloyd/IF-I-XL-v1.0": {
        "name": "DeepFloyd IF",
        "gated": True,
        "license_url": "https://huggingface.co/DeepFloyd/IF-I-XL-v1.0"
    },
    # Additional models for metrics
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k": {
        "name": "CLIP ViT-bigG-14",
        "gated": False,
        "license_url": None
    }
}


def check_auth():
    """Check if HuggingFace authentication is set up."""
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    
    if not token:
        print("\n⚠️  WARNING: No HuggingFace token found!")
        print("\nTo download gated models, you need to:")
        print("1. Create a HuggingFace account at https://huggingface.co/join")
        print("2. Create an access token at https://huggingface.co/settings/tokens")
        print("3. Set your token as an environment variable:")
        print("   export HUGGINGFACE_HUB_TOKEN='your_token_here'")
        print("\nAlternatively, you can login interactively:")
        print("   huggingface-cli login")
        return False
    
    # Try to login with the token
    try:
        login(token=token)
        print(f"✅ Authenticated with HuggingFace (token: {token[:8]}...)")
        return True
    except Exception as e:
        print(f"❌ Failed to authenticate: {e}")
        return False


def check_gated_models():
    """Show which models require license acceptance."""
    gated = [(repo, info) for repo, info in MODELS.items() if info["gated"]]
    
    if not gated:
        print("✅ No gated models in the list!")
        return
    
    print("\n🔒 GATED MODELS REQUIRING LICENSE ACCEPTANCE:")
    print("=" * 60)
    for repo, info in gated:
        print(f"\n📦 {info['name']} ({repo})")
        print(f"   Accept license at: {info['license_url']}")
        print(f"   1. Go to the URL above")
        print(f"   2. Click 'Agree and access repository' button")
        print(f"   3. Make sure you're logged in with the same account as your token")
    print("\n" + "=" * 60)


def download_model(repo_id, cache_dir=None):
    """Download a single model with proper error handling."""
    model_info = MODELS.get(repo_id, {"name": repo_id, "gated": False})
    model_name = model_info["name"]
    
    print(f"\n📥 Downloading {model_name} ({repo_id})...")
    
    # Files to download (common model files)
    allow_patterns = [
        "*.safetensors", "*.bin", "*.pt", "*.pth",
        "*.json", "*.txt", "*.yaml", "*.yml",
        "*model_index.json", "*config*.json",
        "*tokenizer*", "*scheduler*", "*vae*"
    ]
    
    try:
        path = snapshot_download(
            repo_id,
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=["*.md", "*.gitattributes"],
            token=os.environ.get("HUGGINGFACE_HUB_TOKEN"),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"✅ Downloaded to: {path}")
        return True
        
    except GatedRepoError as e:
        print(f"\n❌ GATED MODEL ERROR for {model_name}!")
        print(f"   This model requires license acceptance.")
        if model_info.get("license_url"):
            print(f"   👉 Accept the license at: {model_info['license_url']}")
        print(f"   Then wait a few minutes for access to be granted.")
        print(f"\n   Error details: {str(e)}")
        return False
        
    except RepositoryNotFoundError:
        print(f"\n❌ MODEL NOT FOUND: {repo_id}")
        print(f"   Please check if the model ID is correct.")
        # Suggest alternatives if known
        if "Realistic_Vision" in repo_id:
            print(f"   💡 Try: SG161222/RealVisXL_V4.0")
        return False
        
    except Exception as e:
        print(f"\n❌ Failed to download {model_name}: {str(e)}")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("🚀 BENCHMARK MODELS SETUP")
    print("=" * 60)
    
    # Check authentication
    auth_ok = check_auth()
    
    # Show gated models
    check_gated_models()
    
    if not auth_ok:
        response = input("\nContinue without authentication? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Please set up authentication and try again.")
            sys.exit(1)
        print("\n⚠️  Continuing without auth - only public models will download.")
    
    # Set cache directory
    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"\n📁 Using cache directory: {cache_dir}")
    
    # Download models
    print("\n" + "=" * 60)
    print("DOWNLOADING MODELS")
    print("=" * 60)
    
    success = []
    failed = []
    
    for repo_id in MODELS.keys():
        if download_model(repo_id, cache_dir):
            success.append(repo_id)
        else:
            failed.append(repo_id)
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"\n✅ Successfully downloaded: {len(success)}/{len(MODELS)}")
    for repo in success:
        print(f"   • {MODELS[repo]['name']} ({repo})")
    
    if failed:
        print(f"\n❌ Failed to download: {len(failed)}")
        for repo in failed:
            model_info = MODELS[repo]
            print(f"   • {model_info['name']} ({repo})")
            if model_info["gated"]:
                print(f"     → Accept license at: {model_info['license_url']}")
    
    # Docker instructions
    if failed:
        print("\n" + "=" * 60)
        print("📝 NEXT STEPS FOR GATED MODELS:")
        print("=" * 60)
        print("\n1. Accept licenses for gated models at their URLs above")
        print("2. Wait 5-10 minutes for access to be granted")
        print("3. Run this script again to download the remaining models")
        print("\n4. For Docker usage, mount the cache directory:")
        print(f"   docker run -v {cache_dir}:/workspace/hf_cache ...")
        print("   And set HF_HOME=/workspace/hf_cache in the container")
    
    print("\n✨ Setup complete!")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
