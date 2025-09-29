#!/bin/bash
# NavAI End-to-End Demo Script
# Runs complete pipeline: data → training → inference → factor graph → results
# Usage: ./scripts/run_e2e_demo.sh [--gps-denied] [--dataset comma2k19|synthetic]

set -e  # Exit on any error

# Parse arguments
GPS_DENIED=false
DATASET="synthetic"
while [[ $# -gt 0 ]]; do
    case $1 in
        --gps-denied)
            GPS_DENIED=true
            shift
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--gps-denied] [--dataset comma2k19|synthetic]"
            exit 1
            ;;
    esac
done

echo "🚀 NavAI End-to-End Demo"
echo "========================"
echo "Dataset: $DATASET"
echo "GPS-denied mode: $GPS_DENIED"
echo ""

# Check environment
echo "📋 Environment Check..."
python --version
pip list | grep -E "(torch|gtsam|numpy)" || echo "Installing dependencies..."

# Step 1: Data preparation
echo ""
echo "📊 Step 1: Data Preparation"
cd ml/data
if [ "$GPS_DENIED" = true ]; then
    python data_loader.py --dataset $DATASET --ignore-gps --output ../outputs/demo_data.npz
else
    python data_loader.py --dataset $DATASET --output ../outputs/demo_data.npz
fi

# Step 2: ML Training/Inference
echo ""
echo "🧠 Step 2: ML Speed Estimation"
cd ../
python models/physics_informed_speed_estimator.py --demo --input outputs/demo_data.npz --output outputs/speed_estimates.npz

# Step 3: Factor Graph Integration
echo ""
echo "🔧 Step 3: Factor Graph Navigation"
python models/factor_graph_navigation.py --demo --speed-input outputs/speed_estimates.npz --output outputs/navigation_results.npz

# Step 4: TFLite Export
echo ""
echo "📱 Step 4: Mobile Export"
python export_tflite.py --model models/speed_estimator.py --output outputs/speed_estimator_demo.tflite

# Step 5: Results Analysis
echo ""
echo "📈 Step 5: Results Analysis"
python test_integration.py --analyze --results outputs/navigation_results.npz

echo ""
echo "✅ End-to-End Demo Complete!"
echo "📁 Results saved in ml/outputs/"
echo "📱 TFLite model: outputs/speed_estimator_demo.tflite"
echo ""
echo "🔍 Performance Summary:"
python -c "
import numpy as np
try:
    results = np.load('outputs/navigation_results.npz', allow_pickle=True)
    print(f'  • Trajectory points: {len(results[\"positions\"])}')
    print(f'  • Speed RMSE: {results[\"speed_rmse\"]:.3f} m/s')
    print(f'  • Position drift: {results[\"position_drift\"]:.2f} m')
    print(f'  • Map-matching rate: {results[\"map_match_rate\"]:.1%}')
except:
    print('  • Results analysis pending...')
"