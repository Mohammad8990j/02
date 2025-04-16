import yaml
import os

def load_config(path=None):
    if path is None:
        # مسیر پروژه اصلی (جایی که main.py قرار داره)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        path = os.path.join(base_dir, "configs", "settings.yaml")

    if not os.path.exists(path):
        raise FileNotFoundError(f"فایل تنظیمات یافت نشد: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
