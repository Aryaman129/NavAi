#!/usr/bin/env python3
"""
Environment setup script for NavAI project
Optimized for RTX 4050 with CUDA support
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def check_cuda():
    """Check CUDA availability"""
    print("\n🔍 Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ CUDA available: {gpu_name} ({gpu_memory:.1f}GB)")
            return True
        else:
            print("❌ CUDA not available")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def setup_python_environment():
    """Set up Python environment"""
    print("\n🐍 Setting up Python environment...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major != 3 or python_version.minor < 8:
        print(f"❌ Python 3.8+ required, found {python_version.major}.{python_version.minor}")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Install/upgrade pip
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Install PyTorch with CUDA support
    print("\n🔥 Installing PyTorch with CUDA support...")
    torch_cmd = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    run_command(torch_cmd, "Installing PyTorch with CUDA")
    
    # Install other requirements
    requirements_path = Path("ml/requirements.txt")
    if requirements_path.exists():
        run_command(f"{sys.executable} -m pip install -r {requirements_path}", "Installing ML requirements")
    
    return True

def setup_android_environment():
    """Check Android development environment"""
    print("\n📱 Checking Android development environment...")
    
    # Check for Android SDK (including custom path)
    # Standardized CUDA/PyTorch paths
    default_cuda_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    default_pytorch_path = r'C:\ProgramData\Miniconda3\envs\NavAI\Lib\site-packages\torch'
    
    # Check for CUDA
    cuda_home = os.environ.get('CUDA_PATH') or default_cuda_path
    if os.path.exists(cuda_home):
        print(f"✅ CUDA found: {cuda_home}")
        os.environ['PATH'] += os.pathsep + os.path.join(cuda_home, 'bin')
    else:
        print("⚠️  CUDA not found at standard location")
    
    # Check PyTorch with CUDA
    try:
        import torch
        print(f"✅ PyTorch installed (CUDA: {torch.cuda.is_available()})")
        if torch.cuda.is_available():
            print(f"Using CUDA version {torch.version.cuda}")
    except ImportError:
        print("❌ PyTorch not installed")
    default_android_home = r'D:\Android Platform tool'
    android_home = os.environ.get('ANDROID_HOME') or os.environ.get('ANDROID_SDK_ROOT') or default_android_home
    if os.path.exists(android_home) and os.path.exists(os.path.join(android_home, 'platform-tools', 'adb.exe')):
        print(f"✅ Android SDK found: {android_home}")
    elif os.path.exists(default_android_home):
        print(f"⚠️  Using custom Android SDK at: {default_android_home}\n"
              "   For best results, set ANDROID_HOME environment variable")
    else:
        print("⚠️  Android SDK not found. Please install Android Studio and set ANDROID_HOME")
    
    # Check for Java
    java_result = run_command("java -version", "Checking Java")
    if java_result is not None:
        print("✅ Java available")
    else:
        print("⚠️  Java not found. Please install JDK 11 or later")

def create_project_structure():
    """Create necessary project directories"""
    print("\n📁 Creating project structure...")
    
    directories = [
        "ml/data/navai_logs",
        "ml/data/oxiod",
        "ml/data/iovnbd", 
        "ml/data/comma2k19",
        "ml/models",
        "ml/outputs",
        "mobile/app/src/main/assets",
        "docs",
        "scripts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {directory}")

def create_scripts():
    """Create utility scripts"""
    print("\n📝 Creating utility scripts...")
    
    # Training script
    train_script = """#!/bin/bash
# Quick training script for NavAI
cd ml
python train_local.py --model-type cnn --batch-size 16 --num-epochs 50
"""
    
    with open("scripts/train.sh", "w") as f:
        f.write(train_script)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod("scripts/train.sh", 0o755)
    
    # Android build script
    android_script = """#!/bin/bash
# Build Android app
cd mobile
./gradlew assembleDebug
echo "APK built: mobile/app/build/outputs/apk/debug/app-debug.apk"
"""
    
    with open("scripts/build_android.sh", "w") as f:
        f.write(android_script)
    
    if os.name != 'nt':
        os.chmod("scripts/build_android.sh", 0o755)
    
    print("✅ Created utility scripts")

def main():
    """Main setup function"""
    print("🚀 NavAI Project Setup")
    print("======================")
    
    # Create project structure
    create_project_structure()
    
    # Setup Python environment
    if not setup_python_environment():
        print("❌ Python environment setup failed")
        return
    
    # Check CUDA
    cuda_available = check_cuda()
    if not cuda_available:
        print("⚠️  CUDA not available. Training will use CPU (slower)")
    
    # Check Android environment
    setup_android_environment()
    
    # Create utility scripts
    create_scripts()
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Build Android app: cd mobile && ./gradlew assembleDebug")
    print("2. Install on device: adb install app/build/outputs/apk/debug/app-debug.apk")
    print("3. Collect sensor data using the app")
    print("4. Train model: python ml/train_local.py")
    print("5. Test the trained model")
    
    if cuda_available:
        print(f"\n💡 Your RTX 4050 is ready for training!")
    
if __name__ == "__main__":
    main()
