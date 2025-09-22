#!/bin/bash
# Build Android app
cd mobile
./gradlew assembleDebug
echo "APK built: mobile/app/build/outputs/apk/debug/app-debug.apk"
