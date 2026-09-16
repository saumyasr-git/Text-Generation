#!/bin/bash
set -e

# Pick the Miniconda installer matching this machine's OS and CPU architecture.
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Linux)  PLATFORM="Linux" ;;
  Darwin) PLATFORM="MacOSX" ;;
  *)
    echo "Unsupported operating system: $OS"
    echo "Install Miniconda manually: https://docs.anaconda.com/miniconda/"
    exit 1
    ;;
esac

case "$ARCH" in
  x86_64|amd64)
    CPU="x86_64"
    ;;
  arm64|aarch64)
    if [ "$PLATFORM" = "MacOSX" ]; then CPU="arm64"; else CPU="aarch64"; fi
    ;;
  *)
    echo "Unsupported CPU architecture: $ARCH"
    echo "Install Miniconda manually: https://docs.anaconda.com/miniconda/"
    exit 1
    ;;
esac

INSTALLER="Miniconda3-latest-${PLATFORM}-${CPU}.sh"
echo "Downloading ${INSTALLER}"

mkdir -p ~/miniconda3
curl -fsSL "https://repo.anaconda.com/miniconda/${INSTALLER}" -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
# macOS defaults to zsh, so initialize it too when present.
if [ -n "$(command -v zsh)" ]; then
  ~/miniconda3/bin/conda init zsh
fi

echo "Done. Restart your shell (or 'source ~/.bashrc') before creating the environment."
