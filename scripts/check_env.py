"""Kiểm tra môi trường: phiên bản Python và các thư viện cần thiết."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src  # noqa: E402,F401  (cấu hình console UTF-8)

PACKAGES = [
    ("cv2", "opencv-python"), ("numpy", "numpy"), ("scipy", "scipy"),
    ("pywt", "PyWavelets"), ("skimage", "scikit-image"), ("yaml", "PyYAML"),
    ("matplotlib", "matplotlib"), ("pandas", "pandas"), ("tqdm", "tqdm"),
    ("streamlit", "streamlit"), ("pytest", "pytest"),
]


def main() -> int:
    print(f"Python {sys.version.split()[0]}  ({sys.executable})")
    missing = []
    for module, pip_name in PACKAGES:
        try:
            mod = importlib.import_module(module)
            print(f"  [OK]  {pip_name:<15} {getattr(mod, '__version__', '')}")
        except ImportError:
            print(f"  [!!]  {pip_name:<15} CHƯA CÀI")
            missing.append(pip_name)
    if missing:
        print(f"\nThiếu: pip install {' '.join(missing)}")
        return 1
    print("\nMôi trường sẵn sàng.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
