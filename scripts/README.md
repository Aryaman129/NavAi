# Scripts Organization

This directory contains utility scripts for various project tasks.

---

## 📁 Directory Structure

```
scripts/
├── android/                    # Android-related scripts
│   ├── build_android_studio.ps1
│   ├── check_android_setup.ps1
│   ├── debug_android_app.ps1
│   ├── launch_android_emulator.ps1
│   ├── setup_android_build.ps1
│   ├── setup_android_emulator.ps1
│   └── setup_gradle_d_drive.ps1
│
├── [Dataset Scripts]
│   ├── download_comma2k19.py       # Main download script
│   ├── download_datasets.py        # Multi-dataset downloader
│   ├── check_comma2k19_status.py   # Verify dataset integrity
│   └── prepare_comma2k19_dataset.py # Preprocessing
│
├── [Training Scripts]
│   ├── train_phase1_gpu_optimized.py  # Main training script (use this!)
│   ├── train_optimized_full_dataset.py # Advanced training with attention
│   └── train.sh                        # Shell wrapper for training
│
├── [Evaluation & Demo Scripts]
│   ├── enhanced_integration_demo.py # Full pipeline demo
│   ├── show_results.py              # Visualize training results
│   ├── check_model_params.py        # Model architecture inspection
│   └── debug_speed.py               # Speed prediction debugging
│
├── [GPS/Data Analysis]
│   ├── check_gps_data.py
│   ├── check_gps_structure.py
│   └── setup_environment.py
│
└── [WSL2 Scripts]
    ├── setup_wsl2_gpu.ps1 / .sh
    ├── run_export_wsl2.ps1 / .sh
    └── install_aria2_and_download.ps1
```

---

## 🚀 Quick Reference

### Android Development

**Build and Deploy:**
```powershell
# Quick debug (from root)
..\quick_debug.ps1

# Or use Android scripts
.\android\build_android_studio.ps1
.\android\debug_android_app.ps1
```

**Setup:**
```powershell
# Check Android environment
.\android\check_android_setup.ps1

# Setup Gradle
.\android\setup_gradle_d_drive.ps1

# Launch emulator
.\android\launch_android_emulator.ps1
```

---

### Dataset Management

**Download Comma2k19:**
```bash
python download_comma2k19.py
```

**Check Status:**
```bash
python check_comma2k19_status.py
```

**Prepare for Training:**
```bash
python prepare_comma2k19_dataset.py
```

---

### Model Training

**Train Phase 1 Model (Recommended):**
```bash
cd ../ml
python training/train_phase1_gpu_optimized.py
```

**Or from scripts directory:**
```bash
python train_phase1_gpu_optimized.py
```

**Advanced Training with Attention:**
```bash
python train_optimized_full_dataset.py
```

---

### Debugging & Analysis

**Check Model:**
```bash
python check_model_params.py
```

**Debug Speed Prediction:**
```bash
python debug_speed.py
```

**Show Training Results:**
```bash
python show_results.py
```

**Full Demo:**
```bash
python enhanced_integration_demo.py
```

---

## 📝 Script Descriptions

### Android Scripts (`android/`)

| Script | Description | Usage |
|--------|-------------|-------|
| `build_android_studio.ps1` | Build APK using Gradle | Build automation |
| `check_android_setup.ps1` | Verify Android environment | Troubleshooting |
| `debug_android_app.ps1` | Install and debug APK | Development |
| `launch_android_emulator.ps1` | Start Android emulator | Testing |
| `setup_android_build.ps1` | Configure build environment | Initial setup |
| `setup_android_emulator.ps1` | Setup AVD for testing | Initial setup |
| `setup_gradle_d_drive.ps1` | Move Gradle cache to D: | Save C: drive space |

### Dataset Scripts

| Script | Description | When to Use |
|--------|-------------|-------------|
| `download_comma2k19.py` | Download Comma2k19 dataset | First-time setup |
| `check_comma2k19_status.py` | Verify dataset integrity | After download |
| `prepare_comma2k19_dataset.py` | Preprocess raw data | Before training |

### Training Scripts

| Script | Description | Status |
|--------|-------------|---------|
| `train_phase1_gpu_optimized.py` | **Main training script** | ✅ Use this! |
| `train_optimized_full_dataset.py` | Advanced with attention | ⏳ Experimental |
| `train.sh` | Shell wrapper | Linux/WSL |

### Utility Scripts

| Script | Description |
|--------|-------------|
| `enhanced_integration_demo.py` | Full pipeline demonstration |
| `show_results.py` | Visualize training metrics |
| `check_model_params.py` | Inspect model architecture |
| `debug_speed.py` | Debug speed predictions |
| `check_gps_data.py` | Analyze GPS data quality |

---

## 🗑️ Removed Scripts (Cleanup)

The following obsolete scripts were removed during cleanup:

**Removed:**
- `train_fixed.py` (superseded by `train_phase1_gpu_optimized.py`)
- `train_improved.py` (merged into optimized version)
- `train_phase1_foundation.py` (old baseline)
- `train_phase1_ultra_fast.py` (unstable)
- `download_comma2k19_simple.py` (duplicate)
- `download_comma2k19_smart.py` (duplicate)
- `download_comma2k19_tar.py` (obsolete format)
- `download_comma2k19_parquet.py` (merged into main)
- `fast_comma_download.py` (duplicate)
- `download_real_comma2k19.py` (duplicate)
- `download_real_datasets.py` (duplicate)
- `examine_real_comma2k19.py` (debugging, not needed)
- `parse_real_comma2k19.py` (merged into prepare script)
- `extract_and_prepare_comma2k19.py` (duplicate)
- `comparison_demo.py` (old demo)
- `simplified_enhanced_demo.py` (use enhanced_integration_demo.py)

---

## 💡 Best Practices

1. **Use `quick_debug.ps1`** from root for fast Android builds
2. **Use `train_phase1_gpu_optimized.py`** for model training
3. **Check dataset status** before training with `check_comma2k19_status.py`
4. **Keep Android scripts** in `android/` subdirectory for organization
5. **Remove obsolete scripts** to avoid confusion

---

**Last Updated**: October 10, 2025
