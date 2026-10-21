"""
Nạp file cấu hình YAML tập trung cho toàn bộ pipeline.

Hỗ trợ truy cập cả hai kiểu:
- Attribute : ``cfg.preprocessing.resize_width``
- Mapping   : ``cfg["preprocessing"]["resize_width"]``

Cho phép ``get`` với giá trị mặc định ở mọi cấp, nên các module con
có thể viết an toàn như ``cfg.get("preprocessing", {}).get("blur_ksize", 5)``.

Toàn bộ ngưỡng và tham số được quản lý ở ``configs/default.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import yaml

__all__ = ["ConfigDict", "load_config", "load_default_config", "DEFAULT_CONFIG_PATH"]


# Đường dẫn mặc định tới file cấu hình chính của dự án.
# ``Path(__file__)`` -> .../src/config.py  => parent.parent = thư mục gốc dự án.
DEFAULT_CONFIG_PATH: Path = (
    Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
)


class ConfigDict(dict):
    """
    Dict cho phép truy cập key như attribute, đệ quy xuống mọi nested dict.

    >>> cfg = ConfigDict({"a": {"b": 1}})
    >>> cfg.a.b
    1
    >>> cfg["a"]["b"]
    1
    >>> cfg.get("x", 0)
    0
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Đệ quy chuyển mọi dict con thành ConfigDict để chain attribute được.
        for key, value in list(self.items()):
            self[key] = self._wrap(value)

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            return ConfigDict(value)
        if isinstance(value, list):
            return [ConfigDict._wrap(v) for v in value]
        return value

    # ------------------------------------------------------------------ #
    # Truy cập attribute
    # ------------------------------------------------------------------ #
    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(
                f"Không có khóa cấu hình: '{item}'"
            ) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = self._wrap(value)

    def __delattr__(self, item: str) -> None:
        try:
            del self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


def load_config(path: Union[str, Path]) -> ConfigDict:
    """
    Nạp file YAML và trả về :class:`ConfigDict`.

    Parameters
    ----------
    path : str | Path
        Đường dẫn tới file YAML.

    Raises
    ------
    FileNotFoundError
        Nếu file không tồn tại.
    yaml.YAMLError
        Nếu nội dung YAML sai cú pháp.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(
            f"File cấu hình phải là mapping (dict), nhưng nhận được: {type(data).__name__}"
        )
    return ConfigDict(data)


def load_default_config() -> ConfigDict:
    """Nạp cấu hình mặc định ``configs/default.yaml``."""
    return load_config(DEFAULT_CONFIG_PATH)
