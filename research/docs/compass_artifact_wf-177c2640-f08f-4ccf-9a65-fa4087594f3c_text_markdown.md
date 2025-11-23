# RTX 5090 PyTorch Installation Guide for sm_120

The NVIDIA RTX 5090 with **compute capability sm_120** requires **CUDA 12.8 or higher** and is only officially supported by **PyTorch 2.7.0+ or nightly builds** as of November 2025. Stable PyTorch versions prior to 2.7.0 will fail with "CUDA error: no kernel image is available for execution" messages. The fastest path to working diffusion models is installing PyTorch nightly builds with cu128 support, though building from source provides the most control for custom extensions.

The RTX 5090, launched January 30, 2025, uses NVIDIA's Blackwell architecture with the new sm_120 designation, creating an initial compatibility gap across the deep learning software ecosystem. While this gap has largely closed by late 2025 with PyTorch 2.7.0's release, users still encounter issues with older frameworks and custom CUDA extensions that require recompilation with sm_120 flags.

## Hardware specifications and requirements

The RTX 5090 features **21,760 CUDA cores**, **32GB GDDR7 memory** with 1,792 GB/s bandwidth, and **680 fifth-generation Tensor Cores** supporting FP4, FP8, FP16, and BF16 precision. The GPU delivers 209.5 TFLOPS of FP16 performance, representing a 27% improvement over the RTX 4090, with 3,352 AI TOPS for inference workloads. This makes it particularly well-suited for diffusion models and large language models that can leverage the expanded 32GB memory capacity.

Driver requirements are strict: **Linux systems need driver version 570.86.16 or higher** (with 572+ strongly recommended for stability), while **Windows requires driver 571.86+**. The GPU draws up to 575W under full load, necessitating a 1000W power supply with PCIe 5.0 12V-2x6 connector support. Standard Ubuntu driver installation tools do not work properly with RTX 50 series cards—users must download and run NVIDIA's official .run installer directly from nvidia.com.

## Installing PyTorch with sm_120 support

PyTorch nightly builds represent the easiest installation path for RTX 5090 users. These builds have included sm_120 support since January 31, 2025 for Linux and February 20, 2025 for Windows. The installation command is straightforward:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

This command installs the latest nightly build compiled with CUDA 12.8 support, including pre-compiled kernels for sm_120 architecture. The nightly builds support Python versions 3.9 through 3.13, though **Python 3.10 or later is recommended** for optimal compatibility. After installation, users can verify sm_120 support by checking the architecture list:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Supported architectures: {torch.cuda.get_arch_list()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
```

The output should show sm_120 in the architecture list and compute capability (12, 0). If CUDA operations fail at this point, the issue typically stems from incorrect driver versions rather than PyTorch itself.

For users requiring stable releases, **PyTorch 2.7.0 and later include official sm_120 support** when built with CUDA 12.8. The stable installation follows a similar pattern:

```bash
pip install torch==2.7.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Specific wheel URLs for different Python versions and platforms are available at `https://download.pytorch.org/whl/nightly/cu128/torch/`. For reproducibility in production environments, pinning to a specific nightly build date prevents unexpected API changes.

## Building PyTorch from source with sm_120

Building from source provides maximum control and is essential for users developing custom CUDA extensions. The process requires CUDA Toolkit 12.8 or 12.9, GCC 7+ or Clang 5+, CMake 3.12+, and approximately 50GB of free disk space. Build times range from 30-60 minutes depending on CPU core count.

The critical step is setting the **TORCH_CUDA_ARCH_LIST environment variable to specify sm_120**:

```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git submodule sync
git submodule update --init --recursive

export TORCH_CUDA_ARCH_LIST="12.0"
export FORCE_CUDA=1
export CUDA_HOME=/usr/local/cuda
export CUDACXX=$CUDA_HOME/bin/nvcc

pip install pyyaml numpy ninja cmake
python setup.py develop
```

The TORCH_CUDA_ARCH_LIST variable accepts either "12.0" or "sm_120" format—both produce identical results. The FORCE_CUDA flag ensures CUDA compilation proceeds even if autodetection fails, while CUDACXX explicitly points to the NVCC compiler to prevent build system confusion with system compilers.

For debugging build issues, additional flags prove helpful:

```bash
export TORCH_USE_CUDA_DSA=1        # Enable device-side assertions
export CUDA_LAUNCH_BLOCKING=1      # Synchronous CUDA calls for clearer error messages
export USE_SYSTEM_NCCL=ON          # Use system-installed NCCL library
```

Users can verify the build included sm_120 by inspecting the build artifacts:

```bash
cat build/temp.linux-x86_64-cpython-312/build.ninja | grep sm_120
```

This should display lines containing `-gencode arch=compute_120,code=sm_120`, confirming the compiler generated kernels for the Blackwell architecture.

## Docker container approach

The NVIDIA PyTorch container provides a controlled environment with pre-configured dependencies, making it the recommended approach for advanced users and production deployments:

```bash
docker pull nvcr.io/nvidia/pytorch:25.04-py3
docker run -d --name pytorch-rtx5090 --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=16g \
  -p 8888:8888 nvcr.io/nvidia/pytorch:25.04-py3
```

Inside the container, CUDA 12.9 installation completes the setup:

```bash
docker exec -it pytorch-rtx5090 bash
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install -y cuda-toolkit-12-9
```

The container approach isolates the RTX 5090 environment from system libraries, preventing conflicts with other projects using different GPU architectures. This proves particularly valuable for teams supporting multiple GPU generations simultaneously.

## Known issues with diffusion models

Diffusion model frameworks initially struggled with RTX 5090 support, displaying the persistent error "RuntimeError: CUDA error: no kernel image is available for execution on the device." This error stems from frameworks shipping with PyTorch versions that lack sm_120 kernel compilation. By late 2025, most major frameworks have resolved these issues, though specific configurations remain necessary.

**Automatic1111 Stable Diffusion WebUI** requires the development branch for RTX 5090 support. The stable releases still ship with PyTorch versions predating sm_120 support. Users should clone the dev branch and force PyTorch reinstallation:

```bash
git clone -b dev https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui

# Linux
./webui.sh --reinstall-torch

# Windows (edit webui-user.bat)
set COMMANDLINE_ARGS=--reinstall-torch
```

The xformers library requires version **0.0.30 or later** for RTX 5090 compatibility. Earlier versions either fail to load or provide no performance benefit over vanilla attention mechanisms. Proper xformers configuration reduces SDXL generation time from 37 seconds on RTX 4090 to 25 seconds on RTX 5090 at 1024×1024 resolution.

**ComfyUI** works reliably with correct PyTorch versions. Users should either download the portable version with CUDA 12.8 pre-configured or manually install PyTorch 2.7+:

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
python -m venv venv
source venv/bin/activate  # Linux
# or venv\Scripts\activate on Windows

pip uninstall torch torchvision torchaudio
pip install torch==2.7.1+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python main.py
```

Performance on properly configured systems shows significant improvements: **SDXL 1024×1024 generates in 15 seconds** (down from 22 seconds on RTX 4090), while **SD 1.5 at 512×512 completes in 5-8 seconds**. These benchmarks assume proper thermal management, as the 575W TDP can cause thermal throttling in poorly ventilated cases.

**Fooocus** continues experiencing compatibility issues even with correct PyTorch versions, with GitHub issue #3862 tracking persistent CUDA errors during model initialization. **InvokeAI** requires PyTorch 2.7+ but otherwise functions normally after this upgrade. The broader **Diffusers library** from HuggingFace works without modification when using compatible PyTorch versions.

## Building custom CUDA extensions

Custom CUDA extensions and specialized libraries often require manual recompilation with sm_120 flags. Popular extensions like causal-conv1d, mamba-ssm, and VMamba fail on RTX 5090 until their setup.py files receive modifications. The general pattern involves adding sm_120 to the compute capability list:

```python
# In setup.py, find sections like:
if bare_metal_version >= Version("12.8"):
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_100,code=sm_100")
    
# Add sm_120 support:
if bare_metal_version >= Version("12.8"):
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_120,code=sm_120")
```

After modifying setup.py, build with the standard environment variables:

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
export FORCE_CUDA=1
export CUDA_HOME=/usr/local/cuda
export CUDACXX=$CUDA_HOME/bin/nvcc

pip install . --no-build-isolation -vv
```

The `--no-build-isolation` flag ensures pip uses the environment's CUDA configuration rather than creating a temporary isolated environment that might lack proper CUDA paths. The `-vv` flag provides verbose output for diagnosing compilation failures.

For extensions installed via pip, the same approach works:

```bash
TORCH_CUDA_ARCH_LIST="12.0" pip install --upgrade --no-cache-dir <extension-name>
```

The `--no-cache-dir` flag forces recompilation rather than using cached wheels built for different architectures.

## Hardware and driver workarounds

Several hardware-specific issues affect RTX 5090 stability beyond software configuration. Users report black screen crashes during intensive generation tasks, particularly with driver versions 570-571. **Upgrading to driver version 572 or higher** resolves most stability issues, showing the early driver releases lacked proper power management for the 575W TDP.

PCIe 5.0 compatibility problems manifest as display connection losses or system hangs. The workaround involves switching motherboard BIOS settings to **PCIe 4.0 mode**, which incurs less than 1% performance loss at 4K resolution but dramatically improves stability. This suggests PCIe 5.0 signal integrity issues on some motherboard designs, particularly with longer PCIe riser cables.

Multi-GPU configurations present additional challenges. Dual RTX 5090 setups drawing 1150W+ require exceptional power delivery and thermal management. **AMD EPYC platforms demonstrate better multi-GPU stability than Intel i9 systems**, likely due to superior PCIe lane distribution. Users report VRM temperatures exceeding 107°C in poorly cooled configurations, potentially triggering protection circuits that manifest as mysterious system resets during intensive workloads.

Thermal throttling occurs when ambient temperatures exceed 25°C without adequate case airflow. The 575W TDP produces substantial heat, and NVIDIA's reference cooler design operates at performance limits. Monitoring GPU temperatures during diffusion model generation shows sustained temperatures near the 90°C maximum, indicating users should prioritize case ventilation and consider aftermarket cooling solutions for sustained workloads.

## Recommended setup paths by platform

Linux users enjoy the most straightforward path to RTX 5090 functionality. A complete setup from fresh Ubuntu 22.04 installation follows this sequence:

```bash
# Install latest NVIDIA drivers (manual installation required)
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/570.144/NVIDIA-Linux-x86_64-570.144.run
sudo ./NVIDIA-Linux-x86_64-570.144.run

# Install CUDA 12.9
wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda_12.9.0_555.42.02_linux.run
sudo sh cuda_12.9.0_555.42.02_linux.run

# Create conda environment
conda create -n rtx5090 python=3.11
conda activate rtx5090

# Install PyTorch
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install diffusion framework (ComfyUI example)
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
python main.py
```

Windows users face additional driver complexity but can achieve stable results with similar steps. The critical difference involves downloading drivers directly from NVIDIA's website rather than using Windows Update, which often provides outdated versions incompatible with sm_120 requirements.

For production deployments, the Docker container approach provides reproducibility and isolation:

```bash
docker run --gpus all -it --rm \
  -v ~/models:/workspace/models \
  -p 7860:7860 \
  nvcr.io/nvidia/pytorch:25.04-py3
```

Inside the container, install the specific diffusion framework needed. This approach enables consistent environments across development and production systems, preventing "works on my machine" scenarios common with complex CUDA configurations.

## Performance expectations and benchmarks

Real-world diffusion model performance on RTX 5090 shows significant improvements over RTX 4090 when properly configured. SDXL generation at 1024×1024 resolution drops from 22 seconds to 15 seconds, representing a **32% speedup**. This improvement stems from the combination of increased CUDA core count (21,760 vs 16,384), higher memory bandwidth (1,792 GB/s vs 1,008 GB/s), and enhanced Tensor Core capabilities.

SD 1.5 generation at 512×512 completes in 5-8 seconds compared to 8-10 seconds on RTX 4090. Batch generation shows even larger improvements due to the 32GB VRAM capacity enabling larger batch sizes without memory overflow. A batch of four SDXL images completes in approximately 60 seconds total on RTX 5090, whereas RTX 4090's 24GB VRAM often requires splitting such batches into multiple passes.

The **32GB VRAM capacity** proves transformative for advanced workflows involving multiple models loaded simultaneously. Users can keep SDXL base, refiner, and multiple ControlNet models resident in VRAM, eliminating model swap overhead that dominated RTX 4090 workflows. This enables complex multi-stage generation pipelines without the constant model loading that consumed 30-40% of generation time on previous hardware.

Some users initially reported RTX 5090 performing slower than RTX 4090, invariably traced to incorrect PyTorch versions or driver issues. With PyTorch 2.7+ and driver 572+, performance consistently exceeds RTX 4090 by the expected 30-40% margin across diffusion workloads.

## Conclusion and current state

RTX 5090 support has matured substantially from the challenging early adoption phase in January-February 2025 to the relatively stable state in November 2025. **PyTorch 2.7.0's official sm_120 support** represents the turning point where RTX 5090 transitioned from bleeding-edge liability to viable production GPU. Users can now achieve stable operation through straightforward PyTorch nightly installation or by using PyTorch 2.7.0 stable releases.

The key requirements remain non-negotiable: **CUDA 12.8 or higher**, **driver version 572+ for stability**, and **PyTorch 2.7.0 or nightly builds**. Custom extensions require recompilation with TORCH_CUDA_ARCH_LIST="12.0", but this process is well-documented and straightforward for developers familiar with CUDA compilation.

For immediate use without compilation, install PyTorch nightly builds with cu128 and use development branches of diffusion frameworks. For custom extension work, build PyTorch from source or modify extension setup.py files to include sm_120 compilation flags. For production deployments, NVIDIA PyTorch containers provide stable, reproducible environments.

The 32GB VRAM, enhanced Tensor Cores with FP4 support, and substantial memory bandwidth make RTX 5090 particularly well-suited for diffusion models and large language models that previously struggled with VRAM constraints. While availability remains limited and pricing elevated, the performance improvements justify the upgrade for users running VRAM-constrained workloads on RTX 4090 or earlier hardware.