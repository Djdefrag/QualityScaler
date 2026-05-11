#!/usr/bin/env bash
# Builds QualityScaler as a Linux AppImage.
# Requirements: python3.11, pip, appimagetool (in PATH or ./appimagetool-x86_64.AppImage)
# GPU: onnxruntime-gpu 1.18.x requires CUDA 11 libraries at runtime (bundled automatically).

set -euo pipefail

APPNAME="QualityScaler"
VERSION="2026.3"
APPDIR="AppDir"

# ── Bundle static ffmpeg ──────────────────────────────────────────────────────
FFMPEG_DEST="Assets/ffmpeg"
if [ ! -f "$FFMPEG_DEST" ]; then
    echo "==> Downloading static ffmpeg..."
    FFMPEG_URL="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    TMP_DIR=$(mktemp -d)
    wget -q --show-progress -O "$TMP_DIR/ffmpeg.tar.xz" "$FFMPEG_URL"
    tar -xf "$TMP_DIR/ffmpeg.tar.xz" -C "$TMP_DIR" --strip-components=1
    cp "$TMP_DIR/ffmpeg" "$FFMPEG_DEST"
    chmod +x "$FFMPEG_DEST"
    rm -rf "$TMP_DIR"
    echo "    ffmpeg downloaded: $(${FFMPEG_DEST} -version 2>&1 | head -1)"
else
    echo "==> ffmpeg already present, skipping download."
fi

# ── Bundle static exiftool ────────────────────────────────────────────────────
EXIFTOOL_DEST="Assets/exiftool"
if [ ! -f "$EXIFTOOL_DEST" ]; then
    echo "==> Downloading static exiftool..."
    EXIFTOOL_VER=$(curl -s https://exiftool.org/ver.txt)
    EXIFTOOL_URL="https://exiftool.org/Image-ExifTool-${EXIFTOOL_VER}.tar.gz"
    TMP_DIR=$(mktemp -d)
    wget -q --show-progress -O "$TMP_DIR/exiftool.tar.gz" "$EXIFTOOL_URL"
    tar -xf "$TMP_DIR/exiftool.tar.gz" -C "$TMP_DIR" --strip-components=1
    cp "$TMP_DIR/exiftool" "$EXIFTOOL_DEST"
    chmod +x "$EXIFTOOL_DEST"
    # Copy the lib directory exiftool needs alongside it
    cp -r "$TMP_DIR/lib" "Assets/exiftool_lib"
    rm -rf "$TMP_DIR"
    echo "    exiftool downloaded."
else
    echo "==> exiftool already present, skipping download."
fi

echo "==> Installing Python dependencies..."
pip install --quiet -r requirements.txt
pip install --quiet pyinstaller

echo "==> Running PyInstaller..."
pyinstaller --clean --noconfirm QualityScaler.spec

echo "==> Creating AppDir structure..."
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin"
mkdir -p "$APPDIR/usr/share/applications"
mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"

cp -r "dist/$APPNAME/." "$APPDIR/usr/bin/"

# Bundle CUDA libs from nvidia-* pip packages so the AppImage is self-contained.
echo "==> Bundling CUDA libraries from nvidia pip packages..."
mkdir -p "$APPDIR/usr/lib/cuda"
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
for pkg_lib_dir in "$SITE_PACKAGES"/nvidia/*/lib; do
    if [ -d "$pkg_lib_dir" ]; then
        find "$pkg_lib_dir" -maxdepth 1 -name "*.so*" -exec cp -n {} "$APPDIR/usr/lib/cuda/" \;
    fi
done
echo "    Bundled $(ls "$APPDIR/usr/lib/cuda/" | wc -l) CUDA library files."

# Desktop entry
cat > "$APPDIR/usr/share/applications/$APPNAME.desktop" <<EOF
[Desktop Entry]
Name=$APPNAME
Exec=$APPNAME
Icon=$APPNAME
Type=Application
Categories=Graphics;
EOF

# Icon
if [ -f "Assets/logo.png" ]; then
    cp "Assets/logo.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/$APPNAME.png"
fi

# AppDir root symlinks required by AppImage spec
cp "$APPDIR/usr/share/applications/$APPNAME.desktop" "$APPDIR/$APPNAME.desktop"
cp "$APPDIR/usr/share/icons/hicolor/256x256/apps/$APPNAME.png" "$APPDIR/$APPNAME.png" 2>/dev/null || true

# AppRun entry point
cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/bash
SELF=$(readlink -f "$0")
HERE=$(dirname "$SELF")
export PATH="$HERE/usr/bin:$PATH"

# Prepend bundled CUDA libs (copied from nvidia-* pip packages at build time).
export LD_LIBRARY_PATH="$HERE/usr/lib/cuda${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$HERE/usr/bin/QualityScaler" "$@"
EOF
chmod +x "$APPDIR/AppRun"

echo "==> Packaging AppImage..."
rm -f "${APPNAME}-${VERSION}-x86_64.AppImage"
APPIMAGETOOL=""
if command -v appimagetool &>/dev/null; then
    APPIMAGETOOL="appimagetool"
elif [ -f "./appimagetool-x86_64.AppImage" ]; then
    APPIMAGETOOL="./appimagetool-x86_64.AppImage"
else
    echo ""
    echo "appimagetool not found. Download it and re-run:"
    echo "  wget -q 'https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage'"
    echo "  chmod +x appimagetool-x86_64.AppImage"
    echo ""
    echo "PyInstaller bundle is ready in dist/$APPNAME/ — you can run it directly:"
    echo "  ./dist/$APPNAME/$APPNAME"
    exit 0
fi

ARCH=x86_64 $APPIMAGETOOL "$APPDIR" "${APPNAME}-${VERSION}-x86_64.AppImage"
echo ""
echo "Done: ${APPNAME}-${VERSION}-x86_64.AppImage"
