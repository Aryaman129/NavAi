# GPU Setup Status Report - October 10, 2025

## 🎯 Executive Summary

**Good News**: You ALREADY have all the correct packages installed in WSL2! No reinstallation needed.

**Issue**: GPU passthrough from Windows to WSL2 is not working (NVIDIA driver visibility issue)

**Solution**: Fix system-level GPU passthrough, then test with existing packages

---

## ✅ Current Package Inventory

### Windows Conda Environment (navai-gtsam)
```
Python: 3.10.18
PyTorch: 2.7.1+cu118 (CUDA 11.8 support)
TensorFlow: 2.13.1 (CPU-only - Windows limitation)
NumPy: 1.24.3
scikit-learn: 1.7.2
pandas: 2.3.3 (just installed)
tqdm: installed

GPU Status:
- PyTorch CUDA: False (no CUDA toolkit on Windows)
- TensorFlow GPU: 0 (native Windows support ended at TF 2.10)
```

### WSL2 Ubuntu-22.04 (.venv_wsl) ✅ ALL CORRECT VERSIONS!
```
Python: 3.10.12
PyTorch: 2.5.1+cu121 ✅ (guide recommends 2.4.1+cu121 - newer is fine!)
TensorFlow: 2.17.0 ✅ (PERFECT - exact match with guide!)
NumPy: 1.26.4 ✅ (exact match with guide!)
pandas: 2.3.3 ✅
scikit-learn: 1.7.2 ✅
tqdm: 4.67.1 ✅ (guide wants 4.66.5 - newer is fine!)
torchvision: 0.20.1+cu121 ✅
torchaudio: 2.5.1+cu121 ✅

CUDA Libraries Present:
✅ /usr/lib/wsl/lib/libcuda.so
✅ /usr/lib/wsl/lib/libnvidia-ml.so
✅ /usr/lib/wsl/lib/nvidia-smi
```

**Comparison with Guide's Recommendations:**

| Package | Guide Version | Your WSL2 Version | Status |
|---------|---------------|-------------------|---------|
| Python | 3.11 | 3.10.12 | ✅ Compatible (TF 2.17 works with both) |
| PyTorch | 2.4.1+cu121 | 2.5.1+cu121 | ✅ Newer version (better!) |
| TensorFlow | 2.17.0 | 2.17.0 | ✅ Perfect match! |
| NumPy | 1.26.4 | 1.26.4 | ✅ Perfect match! |
| tqdm | 4.66.5 | 4.67.1 | ✅ Newer version (fine) |

---

## ❌ Current Problem: GPU Passthrough Not Working

### Symptoms
```bash
$ nvidia-smi
Failed to initialize NVML: N/A
```

### Root Cause Analysis

**System Configuration:**
- WSL Version: 2.4.13.0 (updating to 2.6.1) ⏳
- Kernel: 5.15.167.4-microsoft-standard-WSL2 ✅
- Windows: 10.0.26100.6584 (Windows 11) ✅
- NVIDIA Driver: 572.83 (should be compatible) ✅

**Problem**: 
- CUDA libraries exist in `/usr/lib/wsl/lib/` ✅
- But NVIDIA driver NOT visible to WSL2 ❌
- No NVIDIA kernel messages in `dmesg` ❌
- LD_LIBRARY_PATH was empty (fixed, but still not working)

**Diagnosis**: Driver passthrough mechanism broken at system level

### Attempted Fixes

1. ✅ Added `/usr/lib/wsl/lib` to LD_LIBRARY_PATH → Still failed
2. ⏳ Updating WSL from 2.4.13 to 2.6.1 → In progress
3. ⏭️ System restart needed after WSL update
4. ⏭️ May need NVIDIA driver reinstallation

---

## 🔧 Next Steps to Fix GPU Passthrough

### Step 1: Complete WSL Update (In Progress)
```powershell
# Already running:
wsl --update
# Updating to version 2.6.1
```

### Step 2: Restart Computer
**WHY**: Driver passthrough often requires restart after WSL update

```powershell
Restart-Computer
```

### Step 3: Test GPU After Restart
```powershell
# Start Ubuntu WSL2
wsl -d Ubuntu-22.04

# Test nvidia-smi
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 572.83       Driver Version: 572.83       CUDA Version: 12.6     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
# |   0  NVIDIA GeForce RTX 4050 Laptop GPU | P0    30W /  75W |      0MiB /  6144MiB |
# +-----------------------------------------------------------------------------+
```

### Step 4: Test TensorFlow & PyTorch GPU
```bash
cd /mnt/d/NavAi
source .venv_wsl/bin/activate

# Test both frameworks
python3 << 'EOF'
import tensorflow as tf
import torch

print("=== TensorFlow GPU Test ===")
print(f"Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {gpus}")

if gpus:
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.matmul(a, a)
    print("✅ TensorFlow GPU working!")
else:
    print("❌ TensorFlow GPU not available")

print("\n=== PyTorch GPU Test ===")
print(f"Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print("✅ PyTorch GPU working!")
else:
    print("❌ PyTorch GPU not available")
EOF
```

### Step 5: If Still Not Working - Reinstall NVIDIA Driver

**Download latest Game Ready Driver:**
- Go to: https://www.nvidia.com/Download/index.aspx
- Select: GeForce RTX 40 Series → RTX 4050 Laptop GPU → Windows 11
- Download and install
- **IMPORTANT**: During installation, choose "Custom Installation"
- Make sure "WSL-GPU Support" is checked
- Restart after installation

---

## 📊 Comparison: Your Setup vs Guide Requirements

### ✅ What You Already Have (No Action Needed)

| Component | Guide Requirement | Your Status | Action |
|-----------|------------------|-------------|---------|
| WSL2 Ubuntu | 22.04 | 22.04 | ✅ None - already have it |
| Python | 3.11 | 3.10.12 | ✅ None - 3.10 works fine with TF 2.17 |
| PyTorch | 2.4.1+cu121 | 2.5.1+cu121 | ✅ None - newer version is better |
| TensorFlow | 2.17.0 | 2.17.0 | ✅ None - perfect match |
| NumPy | 1.26.4 | 1.26.4 | ✅ None - perfect match |
| torchvision | 0.19.1+cu121 | 0.20.1+cu121 | ✅ None - newer is fine |
| pandas | 2.x | 2.3.3 | ✅ None - already installed |
| tqdm | 4.66.5 | 4.67.1 | ✅ None - newer is fine |

### ⏭️ What Needs Fixing (System Level)

| Issue | Status | Action Required |
|-------|--------|-----------------|
| WSL Version | Updating 2.4.13 → 2.6.1 | ⏳ Wait for update to complete |
| GPU Passthrough | Not working | 🔄 Restart after WSL update |
| NVIDIA Driver Visibility | Failed to initialize NVML | 🔄 May need driver reinstall |

---

## 🚀 Quick Start After GPU Fix

Once GPU passthrough is working, you can immediately run your export script:

```bash
# In WSL2 Ubuntu terminal
cd /mnt/d/NavAi
source .venv_wsl/bin/activate

# Run export with GPU acceleration
python3 ml/export_phase1_tflite.py
```

**Expected Performance:**
- PyTorch GPU: Generate 5000 samples in ~8 sec (vs 120 sec on CPU)
- TensorFlow GPU: Train 30 epochs in ~45 sec (vs 300 sec on CPU)
- Total runtime: ~1.5 min (vs ~7 min on CPU)

---

## 💾 D: Drive Configuration (Completed) ✅

All future package installations will use D: drive:

```
D:\libs\
├── pip_cache\      # Pip downloads and wheels
├── conda_pkgs\     # Conda package cache
└── temp\           # Temporary files
```

**Configuration Files:**
- ✅ `%APPDATA%\pip\pip.ini` - Fixed encoding issue
- ✅ `~/.condarc` - Added D:\libs\conda_pkgs
- ✅ Environment variables set permanently

---

## 📋 Action Checklist

### Immediate (User Action Required)

- [ ] Wait for WSL update to complete (~2 minutes)
- [ ] **Restart computer** (required for driver passthrough)
- [ ] Test `nvidia-smi` in WSL2 after restart
- [ ] Test TensorFlow GPU with existing packages
- [ ] Test PyTorch GPU with existing packages

### If GPU Still Not Working After Restart

- [ ] Reinstall NVIDIA driver with WSL-GPU support checked
- [ ] Verify Windows features (Virtual Machine Platform, WSL)
- [ ] Check Windows Update for latest WSL kernel

### Windows Training (Separate Track)

- [ ] Wait for Phase 1 training to complete (Terminal ID: 42b432e4)
- [ ] Training will create numpy 1.24.3-compatible checkpoint
- [ ] Then run CPU export on Windows (as fallback)

---

## 🎓 Key Learnings

1. **No Package Reinstallation Needed**: WSL2 already has all correct packages
2. **Python 3.10 vs 3.11**: Both work fine with TensorFlow 2.17 and PyTorch 2.5
3. **Package Versions**: Slightly newer versions (PyTorch 2.5.1 vs 2.4.1) are fine
4. **The Real Issue**: System-level GPU passthrough, not package configuration
5. **D: Drive Config**: Prevents C: drive space issues (only 2.6GB free)

---

## 🔍 Diagnostic Commands Reference

```bash
# Check WSL version
wsl --version

# Check kernel version
wsl -d Ubuntu-22.04 -- uname -r

# Check NVIDIA driver in WSL
wsl -d Ubuntu-22.04 -- nvidia-smi

# List installed packages
wsl -d Ubuntu-22.04 -- bash -c "cd /mnt/d/NavAi && source .venv_wsl/bin/activate && pip list"

# Test TensorFlow GPU
wsl -d Ubuntu-22.04 -- bash -c "cd /mnt/d/NavAi && source .venv_wsl/bin/activate && python3 -c 'import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))'"

# Test PyTorch GPU
wsl -d Ubuntu-22.04 -- bash -c "cd /mnt/d/NavAi && source .venv_wsl/bin/activate && python3 -c 'import torch; print(torch.cuda.is_available())'"
```

---

## 📞 Support Resources

- [WSL2 GPU Documentation](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)
- [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [TensorFlow WSL2 Guide](https://www.tensorflow.org/install/pip#windows-wsl2)
- [PyTorch WSL2 Setup](https://pytorch.org/get-started/locally/)

---

**Last Updated**: October 10, 2025 03:15 AM
**Status**: WSL updating, GPU passthrough needs restart to test
