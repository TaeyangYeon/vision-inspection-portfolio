import yaml
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent.parent

def get_image_list(category: str, split: str):
    img_dir = BASE_PATH / f"data/processed/{category}/images/{split}"
    if not img_dir.exists():
        return []
    return sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

def get_label_path(category: str, split: str, img_stem: str):
    return BASE_PATH / f"data/processed/{category}/labels/{split}/{img_stem}.txt"

def load_class_names(category: str):
    yaml_path = BASE_PATH / f"data/processed/{category}/dataset.yaml"
    if not yaml_path.exists():
        return {}
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("names", {})

def get_available_categories():
    processed_dir = BASE_PATH / "data/processed"
    if not processed_dir.exists():
        return []
    return [d.name for d in processed_dir.iterdir() if d.is_dir()]

def get_available_runs():
    outputs_dir = BASE_PATH / "outputs"
    if not outputs_dir.exists():
        return []
    return [d.name for d in outputs_dir.iterdir() if d.is_dir()]