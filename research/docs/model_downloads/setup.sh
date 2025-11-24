#!/bin/bash
# Quick setup script for RTX 5090 benchmark

set -e

echo "================================"
echo "RTX 5090 Benchmark Setup Helper"
echo "================================"

# Check for HuggingFace token
if [ -z "$HUGGINGFACE_HUB_TOKEN" ]; then
    echo ""
    echo "⚠️  WARNING: No HuggingFace token found!"
    echo ""
    echo "Please set your token first:"
    echo "  export HUGGINGFACE_HUB_TOKEN='hf_xxxxxxxxxxxxxxxxxxxx'"
    echo ""
    echo "Get your token at: https://huggingface.co/settings/tokens"
    echo ""
    echo "Also make sure you've accepted licenses for:"
    echo "  • FLUX.1-dev: https://huggingface.co/black-forest-labs/FLUX.1-dev"
    echo "  • SD3 Medium: https://huggingface.co/stabilityai/stable-diffusion-3-medium"
    echo "  • DeepFloyd IF: https://huggingface.co/DeepFloyd/IF-I-XL-v1.0"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ HuggingFace token found: ${HUGGINGFACE_HUB_TOKEN:0:8}..."
fi

# Menu
echo ""
echo "What would you like to do?"
echo ""
echo "1) Download all models (run first)"
echo "2) Run smoke test (verify models work)"
echo "3) Run quick SDXL benchmark"
echo "4) Run full benchmark (15 hours)"
echo "5) Build Docker image"
echo "6) Download models in Docker"
echo ""
read -p "Choose option (1-6): " choice

case $choice in
    1)
        echo "Downloading all models..."
        python3 setup_models.py
        ;;
    2)
        echo "Running smoke test..."
        python3 research/benchmarks/smoke_test_models.py
        ;;
    3)
        echo "Running quick SDXL benchmark..."
        BENCHMARK_MODE=quick_sdxl_lightning python3 research/benchmarks/efficient_benchmark_runner.py
        ;;
    4)
        echo "Running full benchmark (this will take ~15 hours)..."
        read -p "Are you sure? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python3 research/benchmarks/efficient_benchmark_runner.py
        fi
        ;;
    5)
        echo "Building Docker image..."
        docker compose build benchmark
        ;;
    6)
        echo "Downloading models in Docker..."
        docker compose run --rm \
            -e HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN} \
            benchmark python docker_download_models.py
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "✨ Done!"
