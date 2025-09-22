#!/bin/bash

# NavAI Build and Deploy Script
# Comprehensive build, test, and deployment automation

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANDROID_PROJECT="$PROJECT_ROOT/mobile"
ML_PROJECT="$PROJECT_ROOT/ml"
DEVICE_SERIAL=""
BUILD_TYPE="debug"
SKIP_TESTS=false
SKIP_ML=false

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Android SDK
    if [ -z "$ANDROID_HOME" ]; then
        print_error "ANDROID_HOME not set. Please install Android SDK."
        exit 1
    fi
    
    # Check ADB
    if ! command -v adb &> /dev/null; then
        print_error "ADB not found. Please install Android SDK platform tools."
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found."
        exit 1
    fi
    
    # Check CUDA (optional)
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    else
        print_warning "NVIDIA GPU not detected. ML training will use CPU."
    fi
    
    print_success "Prerequisites check passed"
}

# Function to setup Python environment
setup_python_env() {
    if [ "$SKIP_ML" = true ]; then
        print_status "Skipping Python environment setup"
        return
    fi
    
    print_status "Setting up Python environment..."
    
    cd "$ML_PROJECT"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install requirements
    print_status "Installing Python requirements..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    print_success "Python environment ready"
    
    cd "$PROJECT_ROOT"
}

# Function to train ML model
train_ml_model() {
    if [ "$SKIP_ML" = true ]; then
        print_status "Skipping ML model training"
        return
    fi
    
    print_status "Training ML model..."
    
    cd "$ML_PROJECT"
    source venv/bin/activate
    
    # Check if we have training data
    if [ -d "data/navai_logs" ] && [ "$(ls -A data/navai_logs)" ]; then
        print_status "Found training data, training with real data..."
        python train_local.py --model-type cnn --batch-size 16 --num-epochs 50
    else
        print_warning "No training data found, training with synthetic data..."
        python train_local.py --model-type cnn --batch-size 16 --num-epochs 20
    fi
    
    # Export to TensorFlow Lite
    if [ -f "models/best_model.pth" ]; then
        print_status "Exporting model to TensorFlow Lite..."
        python export_tflite.py --pytorch-model models/best_model.pth --model-type cnn --quantize --validate
    else
        print_warning "No trained model found, using default model"
    fi
    
    print_success "ML model training completed"
    
    cd "$PROJECT_ROOT"
}

# Function to run tests
run_tests() {
    if [ "$SKIP_TESTS" = true ]; then
        print_status "Skipping tests"
        return
    fi
    
    print_status "Running tests..."
    
    # Python tests
    if [ "$SKIP_ML" = false ]; then
        cd "$ML_PROJECT"
        source venv/bin/activate
        
        print_status "Running Python tests..."
        python -m pytest tests/ -v || print_warning "Some Python tests failed"
        
        cd "$PROJECT_ROOT"
    fi
    
    # Android tests
    cd "$ANDROID_PROJECT"
    
    print_status "Running Android unit tests..."
    ./gradlew test || print_warning "Some Android tests failed"
    
    print_success "Tests completed"
    
    cd "$PROJECT_ROOT"
}

# Function to build Android app
build_android() {
    print_status "Building Android application..."
    
    cd "$ANDROID_PROJECT"
    
    # Clean previous build
    print_status "Cleaning previous build..."
    ./gradlew clean
    
    # Build APK
    print_status "Building $BUILD_TYPE APK..."
    if [ "$BUILD_TYPE" = "release" ]; then
        ./gradlew assembleRelease
        APK_PATH="app/build/outputs/apk/release/app-release.apk"
    else
        ./gradlew assembleDebug
        APK_PATH="app/build/outputs/apk/debug/app-debug.apk"
    fi
    
    # Check if APK was built successfully
    if [ -f "$APK_PATH" ]; then
        print_success "APK built successfully: $APK_PATH"
        
        # Get APK info
        APK_SIZE=$(du -h "$APK_PATH" | cut -f1)
        print_status "APK size: $APK_SIZE"
    else
        print_error "APK build failed"
        exit 1
    fi
    
    cd "$PROJECT_ROOT"
}

# Function to deploy to device
deploy_to_device() {
    print_status "Deploying to Android device..."
    
    # Check for connected devices
    DEVICES=$(adb devices | grep -v "List of devices" | grep "device$" | wc -l)
    
    if [ "$DEVICES" -eq 0 ]; then
        print_error "No Android devices connected. Please connect a device and enable USB debugging."
        exit 1
    elif [ "$DEVICES" -gt 1 ] && [ -z "$DEVICE_SERIAL" ]; then
        print_error "Multiple devices connected. Please specify device serial with -d option."
        adb devices
        exit 1
    fi
    
    # Set ADB target
    ADB_CMD="adb"
    if [ -n "$DEVICE_SERIAL" ]; then
        ADB_CMD="adb -s $DEVICE_SERIAL"
    fi
    
    # Get device info
    DEVICE_MODEL=$($ADB_CMD shell getprop ro.product.model)
    ANDROID_VERSION=$($ADB_CMD shell getprop ro.build.version.release)
    print_status "Target device: $DEVICE_MODEL (Android $ANDROID_VERSION)"
    
    # Install APK
    cd "$ANDROID_PROJECT"
    
    if [ "$BUILD_TYPE" = "release" ]; then
        APK_PATH="app/build/outputs/apk/release/app-release.apk"
    else
        APK_PATH="app/build/outputs/apk/debug/app-debug.apk"
    fi
    
    print_status "Installing APK..."
    $ADB_CMD install -r "$APK_PATH"
    
    # Grant permissions
    print_status "Granting permissions..."
    $ADB_CMD shell pm grant com.navai.logger android.permission.ACCESS_FINE_LOCATION
    $ADB_CMD shell pm grant com.navai.logger android.permission.ACCESS_COARSE_LOCATION
    $ADB_CMD shell pm grant com.navai.logger android.permission.HIGH_SAMPLING_RATE_SENSORS
    
    # Start the app
    print_status "Starting NavAI Logger..."
    $ADB_CMD shell am start -n com.navai.logger/.MainActivity
    
    print_success "Deployment completed successfully!"
    
    cd "$PROJECT_ROOT"
}

# Function to collect logs from device
collect_logs() {
    print_status "Collecting logs from device..."
    
    ADB_CMD="adb"
    if [ -n "$DEVICE_SERIAL" ]; then
        ADB_CMD="adb -s $DEVICE_SERIAL"
    fi
    
    # Create local logs directory
    mkdir -p "$ML_PROJECT/data/navai_logs"
    
    # Pull logs from device
    $ADB_CMD pull /sdcard/Android/data/com.navai.logger/files/logs/ "$ML_PROJECT/data/navai_logs/" 2>/dev/null || {
        print_warning "Could not pull logs. Make sure the app has created log files."
        return
    }
    
    # Count collected files
    LOG_COUNT=$(find "$ML_PROJECT/data/navai_logs" -name "*.csv" | wc -l)
    if [ "$LOG_COUNT" -gt 0 ]; then
        print_success "Collected $LOG_COUNT log files"
    else
        print_warning "No log files found"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -d, --device SERIAL Specify device serial number"
    echo "  -r, --release       Build release version (default: debug)"
    echo "  --skip-tests        Skip running tests"
    echo "  --skip-ml           Skip ML model training"
    echo "  --collect-logs      Collect logs from device after deployment"
    echo "  --ml-only           Only train ML model, skip Android build"
    echo "  --android-only      Only build Android app, skip ML training"
    echo ""
    echo "Examples:"
    echo "  $0                  # Full build and deploy (debug)"
    echo "  $0 -r               # Build and deploy release version"
    echo "  $0 --ml-only        # Only train ML model"
    echo "  $0 --android-only   # Only build Android app"
    echo "  $0 --collect-logs   # Build, deploy, and collect logs"
}

# Parse command line arguments
COLLECT_LOGS=false
ML_ONLY=false
ANDROID_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -d|--device)
            DEVICE_SERIAL="$2"
            shift 2
            ;;
        -r|--release)
            BUILD_TYPE="release"
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-ml)
            SKIP_ML=true
            shift
            ;;
        --collect-logs)
            COLLECT_LOGS=true
            shift
            ;;
        --ml-only)
            ML_ONLY=true
            ANDROID_ONLY=false
            shift
            ;;
        --android-only)
            ANDROID_ONLY=true
            SKIP_ML=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_status "Starting NavAI build and deployment process..."
    print_status "Project root: $PROJECT_ROOT"
    print_status "Build type: $BUILD_TYPE"
    
    # Check prerequisites
    check_prerequisites
    
    if [ "$ML_ONLY" = true ]; then
        # Only ML training
        setup_python_env
        train_ml_model
        print_success "ML training completed!"
        exit 0
    fi
    
    if [ "$ANDROID_ONLY" = false ]; then
        # Setup Python environment and train model
        setup_python_env
        train_ml_model
    fi
    
    # Run tests
    run_tests
    
    # Build Android app
    build_android
    
    # Deploy to device
    deploy_to_device
    
    # Collect logs if requested
    if [ "$COLLECT_LOGS" = true ]; then
        print_status "Waiting 10 seconds for app to start..."
        sleep 10
        collect_logs
    fi
    
    print_success "Build and deployment process completed successfully!"
    print_status "Next steps:"
    print_status "1. Open NavAI Logger app on your device"
    print_status "2. Start logging and take a test drive"
    print_status "3. Use --collect-logs to gather training data"
    print_status "4. Retrain models with real data for better accuracy"
}

# Run main function
main
