# 📝 NavAI Scratchpad

**Purpose**: Temporary notes, ongoing work, ideas, and WIP items. Content here gets moved to PROJECT_DIARY.md when completed.

**Last Updated**: October 9, 2025

---

## 🚧 Currently Working On

### Documentation Consolidation (In Progress)

**Status**: Plan created, ready to execute

**What We're Doing**:
- Reducing 33 .md files to 6 core documents
- Zero data loss - all valuable info preserved
- Better organization for future updates

**Files to Create**:
- [ ] PROJECT_STATUS.md (consolidate current state)
- [ ] IMPLEMENTATION_GUIDE.md (merge implementation plans)
- [ ] DATASET_GUIDE.md (merge dataset docs)

**Files to Rename**:
- [ ] COMPLETE_TRAINING_TIMELINE.md → TRAINING_HISTORY.md
- [ ] BEGINNER_EXPLANATION.md → CONCEPTS_EXPLAINED.md

**Files to Delete** (~27 files after extracting content):
- [ ] currentstatus.md
- [ ] FINAL_STATUS_REPORT.md
- [ ] PLANNING_COMPLETE_SUMMARY.md
- [ ] ... (see DOCUMENTATION_CONSOLIDATION_PLAN.md for full list)

**Next Step**: Execute consolidation plan

---

### Bug Fixes (Planned)

**Bugs Identified**:
1. ❌ R² = 0.0 (should be ~0.037)
2. ❌ RMSE = MAE (15.356 m/s, unusual)
3. ❌ Validation R² = -16 trillion

**Root Cause**: No feature normalization

**Fix Required**:
```python
from sklearn.preprocessing import StandardScaler

# In dataset __init__:
self.scaler = StandardScaler()
self.scaler.fit(imu_features)

# In __getitem__:
normalized_features = self.scaler.transform(features)
```

**Files to Modify**:
- `ml/training/train_speed_estimation_fixed.py`
- OR use `scripts/train_optimized_full_dataset.py` (already has better architecture)

**Next Step**: Implement normalization fix

---

### Training Plan (Next)

**Phase 1: Fix & Baseline**
- Add normalization
- Use full 478k samples (not just 100k)
- Proper temporal train/test split
- Target: RMSE 10-12 m/s

**Phase 2: Integrated Features**
- Use `train_optimized_full_dataset.py`
- Enable: BiLSTM + Attention + Physics Loss
- Target: RMSE 6-8 m/s

**Phase 3: GTSAM Post-Processing**
- Use `improvements/factor_graph_navigation.py`
- Apply to predictions
- Target: RMSE < 5 m/s

**Next Step**: Start Phase 1 after doc consolidation

---

## 💡 Ideas & Notes

### Potential Improvements (To Consider Later)

**Data Augmentation**:
- Time warping
- Magnitude warping
- Rotation augmentation
- Noise injection

**Model Architecture**:
- Try TCN from `hardware_aware_tcn.py`
- Experiment with Transformer
- Test CNN-LSTM hybrid

**Training Tricks**:
- Learning rate scheduling
- Gradient clipping
- Early stopping
- Model checkpointing

**Evaluation**:
- Per-speed-range metrics (low/medium/high speed)
- Per-maneuver metrics (acceleration/braking/turning)
- Uncertainty calibration

---

## ❓ Questions / Decisions Needed

### Current Questions

**Q1**: Should we train on ALL 478k samples or use a subset for faster iteration?
- **Consideration**: Full dataset = better model, but slower training
- **Decision**: TBD (ask user or start with subset, then full)

**Q2**: Which temporal model to use first?
- Option A: BiLSTM (current, proven)
- Option B: TCN (mobile-optimized, faster)
- Option C: Hybrid (BiLSTM + TCN)
- **Decision**: Start with BiLSTM (already working), then experiment

**Q3**: How to handle GTSAM factor graph?
- Option A: Separate script (post-processing)
- Option B: Integrated into inference pipeline
- **Decision**: Separate script first, integrate later

---

## 🐛 Known Issues

### Active Bugs
1. ✅ R² = 0 bug - **IDENTIFIED** (no normalization)
2. ✅ RMSE = MAE bug - **IDENTIFIED** (model not learning)
3. ⏳ Only 21% dataset used - **TO FIX** (use all 478k)

### Resolved Issues
- ✅ TAR.GZ redundancy - **FIXED** (deleted 3.5 GB)
- ✅ Dataset confusion - **FIXED** (documented 3 formats)
- ✅ Advanced features ignored - **FIXED** (integration plan created)

---

## 📋 Quick TODO

### High Priority (This Week)
- [ ] Complete documentation consolidation
- [ ] Add feature normalization
- [ ] Train on full 478k samples
- [ ] Verify bugs fixed

### Medium Priority (Next Week)
- [ ] Integrate advanced features
- [ ] Add data augmentation
- [ ] Implement proper testing suite
- [ ] Create evaluation dashboard

### Low Priority (Future)
- [ ] Mobile deployment
- [ ] Real-time testing
- [ ] ARCore VIO integration
- [ ] Multi-user dataset

---

## 🔬 Experiments to Try

### Experiment 1: Normalization Impact
**Hypothesis**: Adding normalization will reduce RMSE from 15.3 to ~10-12 m/s  
**Status**: Planned  
**Expected Result**: 30-40% improvement

### Experiment 2: Full Dataset Training
**Hypothesis**: Using 478k (not 100k) will improve RMSE to 8-10 m/s  
**Status**: Planned  
**Expected Result**: 20% improvement over 100k

### Experiment 3: Attention Mechanism
**Hypothesis**: Adding attention will improve RMSE to 6-8 m/s  
**Status**: Planned (use train_optimized_full_dataset.py)  
**Expected Result**: 25% improvement

### Experiment 4: GTSAM Post-Processing
**Hypothesis**: GTSAM refinement will achieve RMSE < 5 m/s  
**Status**: Planned (Phase 3)  
**Expected Result**: 40% improvement over neural network alone

---

## 📊 Performance Tracking

### Training Runs

| Date | Dataset | Samples | Model | RMSE | R² | Notes |
|------|---------|---------|-------|------|----|----|
| Oct 8 | Synthetic | 30k | BiLSTM | 0.286 | ? | GPU test |
| Oct 9 | Real | 100k | BiLSTM | 8.25 | ? | Eval on train data (invalid) |
| Jan 9 | Real | ? | BiLSTM | 15.356 | 0.0 | Buggy (no normalization) |
| TBD | Real | 478k | BiLSTM+Norm | 10-12 | 0.05-0.10 | Phase 1 target |
| TBD | Real | 478k | BiLSTM+Attn+Physics | 6-8 | 0.25-0.35 | Phase 2 target |
| TBD | Real | 478k | Above+GTSAM | <5 | >0.40 | Phase 3 target |

### Dataset Status

| Format | Location | Size | Samples | Status |
|--------|----------|------|---------|--------|
| Synthetic | `processed/` | 16.4 MB | 60k | ✅ Keep for testing |
| Parquet | `processed_real/` | 3.8 GB | 478k | ✅ Production dataset |
| TAR.GZ | `raw_tar/` | 3.5 GB | 478k | ✅ Deleted (redundant) |

---

## 🎯 Success Criteria

### Phase 1 Success
- [ ] RMSE < 12 m/s
- [ ] R² > 0.05
- [ ] Trained on 478k samples
- [ ] Proper train/test split (temporal)
- [ ] Feature normalization working

### Phase 2 Success
- [ ] RMSE < 8 m/s
- [ ] R² > 0.25
- [ ] Attention mechanism working
- [ ] Physics-informed loss integrated
- [ ] Uncertainty estimation working

### Phase 3 Success
- [ ] RMSE < 5 m/s
- [ ] R² > 0.40
- [ ] GTSAM refinement working
- [ ] Real-time inference (<100ms)
- [ ] Ready for mobile deployment

---

## 🔧 Code Snippets to Remember

### Feature Normalization (To Implement)
```python
from sklearn.preprocessing import StandardScaler

class IMUDataset(Dataset):
    def __init__(self, csv_path):
        # Load data
        self.data = pd.read_csv(csv_path)
        
        # Extract features
        feature_cols = ['accel_x', 'accel_y', 'accel_z', 
                       'gyro_x', 'gyro_y', 'gyro_z']
        self.features = self.data[feature_cols].values
        
        # Normalize
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
        
        # Save scaler for inference
        joblib.dump(self.scaler, 'scaler.pkl')
```

### Temporal Train/Test Split (To Implement)
```python
# Don't shuffle! Use temporal order
total_samples = len(dataset)
train_size = int(0.7 * total_samples)
val_size = int(0.15 * total_samples)

# Early time → Train, Middle → Val, Late → Test
train_dataset = Subset(dataset, range(0, train_size))
val_dataset = Subset(dataset, range(train_size, train_size + val_size))
test_dataset = Subset(dataset, range(train_size + val_size, total_samples))
```

### Physics-Informed Loss (Already in train_optimized_full_dataset.py)
```python
def physics_informed_loss(predictions, targets, features):
    # Standard MSE
    mse = F.mse_loss(predictions, targets)
    
    # Smoothness penalty (temporal coherence)
    smoothness = torch.mean((predictions[1:] - predictions[:-1])**2)
    
    # Physical bounds (speed should be non-negative)
    bounds = torch.mean(F.relu(-predictions)**2)
    
    # Combined loss
    total_loss = mse + 0.1 * smoothness + 0.1 * bounds
    return total_loss
```

---

## 📝 Meeting Notes / Decisions

### October 9, 2025 - Documentation Discussion

**User Concern**: Too many .md files (33 total), getting messy

**Decision Made**:
- Consolidate to 6 core documents
- Create PROJECT_DIARY.md for history (append-only)
- Create SCRATCHPAD.md for WIP (this file)
- Update existing docs instead of creating new ones
- Zero data loss policy

**Action Items**:
- [x] Create PROJECT_DIARY.md
- [x] Create SCRATCHPAD.md
- [ ] Execute consolidation plan
- [ ] Update docs going forward (don't create new ones)

---

### October 9, 2025 - Architecture Discussion

**User Question**: "About training and our advanced features how would they go? Like separately or in the training itself?"

**Decision Made**: **INTEGRATED approach**

**Reasoning**:
- End-to-end optimization works better
- We already have `train_optimized_full_dataset.py`
- Simpler deployment (one model)
- GTSAM is the only thing that's post-processing

**Action Items**:
- [x] Document architecture decision
- [ ] Use train_optimized_full_dataset.py for Phase 2
- [ ] Implement GTSAM as separate post-processing step

---

## 🗂️ File References

### Training Scripts
- `ml/training/train_speed_estimation_fixed.py` - Current (buggy)
- `scripts/train_optimized_full_dataset.py` - Integrated advanced features ⭐

### Analysis Scripts
- `ml/analysis/quick_bug_check.py` - Bug diagnostic
- `ml/analysis/investigate_training_results.py` - Detailed investigation

### Advanced Features
- `improvements/factor_graph_navigation.py` - GTSAM implementation
- `improvements/hardware_aware_tcn.py` - Mobile TCN
- `improvements/enhanced_user_priors.py` - Activity classification
- `improvements/visual_inertial_navigation.py` - VIO integration

### Documentation (After Consolidation)
- `PROJECT_STATUS.md` - Current state
- `TRAINING_HISTORY.md` - Historical record
- `IMPLEMENTATION_GUIDE.md` - How-to guide
- `DATASET_GUIDE.md` - Dataset info
- `CONCEPTS_EXPLAINED.md` - Educational
- `PROJECT_DIARY.md` - Project history
- `SCRATCHPAD.md` - This file (WIP)

---

## 🎨 Visualization Ideas

### Training Dashboard (To Build)
- Real-time loss curves
- RMSE vs epoch
- R² score evolution
- Speed prediction scatter plots
- Error distribution histograms
- Per-speed-range performance

### Inference Visualization
- Live IMU signal plots
- Speed predictions over time
- Uncertainty bands
- GTSAM trajectory visualization
- Comparison: Neural network vs GTSAM-refined

---

## 🚀 Deployment Checklist (Future)

### Mobile Deployment
- [ ] Export model to TFLite
- [ ] Optimize for mobile (quantization)
- [ ] Test on Android device
- [ ] Measure inference time (<100ms target)
- [ ] Measure battery impact (<10% target)
- [ ] Integrate with Android app

### Production Readiness
- [ ] Model versioning
- [ ] A/B testing framework
- [ ] Monitoring and logging
- [ ] Error handling
- [ ] Fallback mechanisms
- [ ] User feedback collection

---

## 💾 Backup & Safety

### Data Backup
- ✅ Real dataset: 3.8 GB parquet (backed up)
- ✅ Training results: Saved in `ml/reports/`
- ✅ Model checkpoints: Saved during training

### Code Backup
- ✅ Git repository on GitHub
- ✅ All changes committed
- ⏳ Need to add comprehensive .gitignore

### Documentation Backup
- ✅ All docs in `docs/` folder
- ⏳ Will be organized after consolidation

---

## 📚 Learning Resources

### Papers to Read
- [ ] "Invariant Extended Kalman Filtering for Autonomous Vehicle Navigation"
- [ ] "Learning to Estimate the Speed of Vehicular Traffic from Synthetic Aperture Radar"
- [ ] "Deep Inertial Odometry"
- [ ] "RINS-W: Robust Inertial Navigation System on Wheels"

### Implementations to Study
- [ ] comma.ai openpilot (speed estimation)
- [ ] VINS-Mono (visual-inertial odometry)
- [ ] ORB-SLAM3 (SLAM with IMU)

---

## 🎯 Long-Term Vision

### Version 1.0 Goals
- RMSE < 5 m/s on real-world data
- Real-time mobile inference (<100ms)
- Battery efficient (<10% drain)
- Robust to sensor noise

### Version 2.0 Goals
- Multi-modal fusion (IMU + GPS + Vision)
- Map-aided navigation
- Collaborative positioning
- Sub-meter accuracy

### Version 3.0 Goals
- AR navigation overlay
- Indoor-outdoor seamless transition
- Multi-platform support (iOS, Android, Web)
- Cloud-based model updates

---

**Instructions for Using This Scratchpad**:

1. **Add notes here** as you work (don't create new .md files!)
2. **Update sections** as tasks progress
3. **Move completed items** to PROJECT_DIARY.md when done
4. **Keep it messy** - this is your working space
5. **Clean up** when a major milestone is reached
6. **Don't delete** - archive old sections at bottom if needed

---

**Last Updated**: October 9, 2025, 11:45 PM  
**Next Update**: When starting documentation consolidation

---

## 🗄️ Archive (Old Scratchpad Entries)

*Completed items will be moved here before going to PROJECT_DIARY.md*

---

*This is a living document. Feel free to be messy - it's a scratchpad!*
