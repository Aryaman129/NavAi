# GPU Configuration Guide

## Current Status (October 10, 2025)

### Hardware
- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)
- **Driver**: 572.83

### Windows Environment (navai-gtsam conda)

#### Installed Packages
- Python 3.10.18
- PyTorch 2.7.1+cu118 (CUDA 11.8 support)
- TensorFlow 2.13.1 (CPU-only)
- scikit-learn 1.7.2
- NumPy 1.24.3

#### GPU Status
- **PyTorch CUDA**: ❌ False (CUDA runtime not installed)
- **TensorFlow GPU**: ❌ 0 devices (Windows GPU support ended at TF 2.10)

---

## Why No GPU?

### 1. PyTorch CUDA: False

**Reason**: PyTorch has CUDA support built-in (`+cu118`), but Windows needs CUDA Toolkit installed separately.

**Solution Options**:

**Option A: Install CUDA Toolkit 11.8 on Windows**
```powershell
# Download from: https://developer.nvidia.com/cuda-11-8-0-download-archive
# Install CUDA 11.8 (requires ~6GB disk space)
# Add to PATH: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
```

**Option B: Use PyTorch CPU-only** (current setup works fine)
- PyTorch is only used for model loading in export script
- GPU acceleration not critical for this step
- Saves disk space (C: drive only has 2.6GB free)

**Recommendation**: Keep PyTorch CPU for now, focus on TensorFlow GPU via WSL2

---

### 2. TensorFlow GPU: 0 Devices

**Reason**: Microsoft and NVIDIA officially dropped native Windows GPU support after TensorFlow 2.10 (2022)

**The ONLY Working Solution**: WSL2 Ubuntu with TensorFlow 2.17+

**Why Native Windows Doesn't Work**:
- TensorFlow 2.11+ requires DirectML plugin (discontinued in 2025)
- TensorFlow 2.10 is outdated and has security vulnerabilities
- TensorFlow-GPU Windows wheels no longer published

**Official Microsoft Recommendation**: Use WSL2
- Source: https://www.tensorflow.org/install/pip#windows-native
- Quote: "Starting with TensorFlow 2.11, GPU support on native-Windows is only available via tensorflow-directml-plugin"

---

## Permanent D: Drive Configuration ✅

All package downloads now use D: drive to avoid C: drive space issues.

### Created Directories
```
D:\libs\
├── pip_cache\      # Pip downloads and wheels
├── conda_pkgs\     # Conda package cache
└── temp\           # Temporary files during installation
```

### Environment Variables (Permanent)
- `PIP_CACHE_DIR` = `D:\libs\pip_cache`
- `CONDA_PKGS_DIRS` = `D:\libs\conda_pkgs`
- `TEMP` = `D:\libs\temp`
- `TMP` = `D:\libs\temp`

### Configuration Files
- **Pip Config**: `%APPDATA%\pip\pip.ini`
  ```ini
  [global]
  cache-dir = D:\libs\pip_cache
  ```

- **Conda Config**: `~/.condarc`
  ```yaml
  pkgs_dirs:
    - D:\libs\conda_pkgs
  ```

### Verification
```powershell
# Check environment variables
[System.Environment]::GetEnvironmentVariable('PIP_CACHE_DIR', 'User')
[System.Environment]::GetEnvironmentVariable('CONDA_PKGS_DIRS', 'User')

# Test pip installation
conda run -n navai-gtsam pip install --dry-run some-package
# Should show: "Using cache: D:\libs\pip_cache"

# Test conda installation
conda install --dry-run numpy
# Should show: "pkgs_dirs: D:\libs\conda_pkgs"
```

---

## WSL2 GPU Setup Plan (Methodical Approach)

### Prerequisites
1. ✅ WSL2 installed (Ubuntu-22.04)
2. ✅ NVIDIA GPU detected in Windows
3. ⏳ NVIDIA Driver in WSL2 (automatic with Windows driver 572.83)

### Installation Order (CRITICAL)

**Phase 1: Clean Environment**
```bash
cd /mnt/d/NavAi
rm -rf .venv_wsl
python3 -m venv .venv_wsl
source .venv_wsl/bin/activate
pip install --upgrade pip
```

**Phase 2: Install TensorFlow FIRST** (sets CUDA/cuDNN baseline)
```bash
pip install tensorflow[and-cuda]
# This installs:
# - TensorFlow 2.17.0
# - CUDA 12.3 libraries
# - cuDNN 8.9 libraries
```

**Phase 3: Verify TensorFlow GPU**
```bash
python3 -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
# Expected: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

**Phase 4: Install PyTorch** (adapts to existing CUDA)
```bash
# Install PyTorch that works with TensorFlow's CUDA libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Phase 5: Install Other Dependencies**
```bash
pip install pandas tqdm scikit-learn
```

**Phase 6: Test Export with GPU**
```bash
python3 ml/export_phase1_tflite.py
# Should show: TensorFlow using GPU, PyTorch using CPU (acceptable)
```

---

## Current Workaround (CPU Export)

**Status**: Export running successfully on Windows CPU
- Terminal ID: 596bfb5f-c03d-4860-8d73-04e19ef7b7bc
- Estimated time: 20-30 minutes
- Output: `ml/outputs/phase1_model.tflite`

**Why This Works**:
- All packages now compatible with numpy 1.24.3
- PyTorch CPU is sufficient for model loading
- TensorFlow CPU can complete export (just slower)

**After Export Completes**:
1. Verify TFLite model works
2. Benchmark inference time
3. Then set up WSL2 for future GPU-accelerated exports

---

## Summary

### Windows (Current)
- ✅ All packages installed and working
- ✅ D: drive configuration permanent
- ❌ No GPU acceleration (by design)
- ✅ CPU export running successfully

### WSL2 (Next Step)
- Clean installation following proper order
- TensorFlow GPU acceleration available
- PyTorch can be CPU (not critical for export)
- Estimated setup time: 15-20 minutes

### Recommendation
1. ✅ Keep current Windows setup for CPU work
2. ✅ D: drive configuration prevents space issues
3. ⏭️ Set up WSL2 when GPU acceleration needed
4. ⏭️ Use WSL2 for all future TensorFlow GPU work

---

## References

- [TensorFlow Windows GPU Support Deprecation](https://www.tensorflow.org/install/pip#windows-native)
- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)
- [WSL2 GPU Support](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-tensorflow-wsl)
- [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
