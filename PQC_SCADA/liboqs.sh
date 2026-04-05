#!/bin/bash

set -e

echo "====== liboqs CLEAN INSTALL (Auto Path) ======"

PROJECT_DIR=$(pwd)
LIBOQS_DIR="$PROJECT_DIR/liboqs"

echo "📂 Using project directory: $PROJECT_DIR"

# -------- STEP 1: Dependencies --------
echo "[1/7] Installing dependencies..."
sudo apt update
sudo apt install -y build-essential cmake ninja-build git libssl-dev python3-dev libc-bin

echo "✅ Dependencies OK"

# -------- STEP 2: Prepare liboqs --------
echo "[2/7] Preparing liboqs..."

if [ -d "$LIBOQS_DIR" ]; then
    echo "Existing liboqs found → cleaning..."
    cd "$LIBOQS_DIR"
    rm -rf build
else
    echo "Cloning liboqs..."
    git clone --depth=1 https://github.com/open-quantum-safe/liboqs.git
    cd liboqs
fi

echo "✅ Ready"

# -------- STEP 3: Configure --------
echo "[3/7] Configuring build..."

mkdir build
cd build

cmake -GNinja \
-DBUILD_SHARED_LIBS=ON \
-DOQS_BUILD_ONLY_LIB=ON \
.. || {
    echo "❌ CMake failed"
    exit 1
}

echo "✅ CMake OK"

# -------- STEP 4: Build --------
echo "[4/7] Building..."

ninja || {
    echo "❌ Build failed"
    exit 1
}

echo "✅ Build complete"

# -------- STEP 5: Verify --------
echo "[5/7] Checking for liboqs.so..."

LIB_FOUND=$(find . -name "liboqs.so")

if [ -z "$LIB_FOUND" ]; then
    echo "❌ ERROR: liboqs.so not found"
    exit 1
else
    echo "✅ Found: $LIB_FOUND"
fi

# -------- STEP 6: Install --------
echo "[6/7] Installing..."

sudo ninja install
sudo ldconfig

echo "✅ Installed"

# -------- STEP 7: Final Check --------
echo "[7/7] Verifying..."

CHECK=$(ldconfig -p | grep oqs)

if [ -z "$CHECK" ]; then
    echo "⚠️ Not found → copying manually..."

    sudo cp ./lib/liboqs.so* /usr/local/lib/
    sudo ldconfig
fi

FINAL=$(ldconfig -p | grep oqs)

if [ -z "$FINAL" ]; then
    echo "❌ STILL NOT WORKING"
    exit 1
else
    echo "🎉 SUCCESS"
    echo "$FINAL"
fi

echo "====== DONE ======"
