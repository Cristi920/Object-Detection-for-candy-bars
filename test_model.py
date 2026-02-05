# ---------------------------------------------------------
# Importuri
# ---------------------------------------------------------
from ultralytics import YOLO
from pathlib import Path
import re

# ---------------------------------------------------------
# Director proiect
# ---------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Users\clisu\Desktop\New folder")
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"

# ---------------------------------------------------------
# Model + imagini test
# ---------------------------------------------------------
MODEL_PATH = PROJECT_ROOT / "runs/detect/train4/weights/best.pt"
TEST_IMAGES = PROJECT_ROOT / "data/validation/images"

assert MODEL_PATH.exists(), f"❌ Model not found: {MODEL_PATH}"
assert TEST_IMAGES.exists(), f"❌ Images folder not found: {TEST_IMAGES}"

# ---------------------------------------------------------
# Găsim automat următorul folder predict liber
# ---------------------------------------------------------
def next_predict_folder(base_dir):
    base_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r"predict(\d*)")
    max_id = 0

    for folder in base_dir.iterdir():
        if folder.is_dir():
            match = pattern.fullmatch(folder.name)
            if match:
                num = match.group(1)
                num = int(num) if num else 1
                max_id = max(max_id, num)

    return f"predict{max_id + 1 if max_id else ''}"

PREDICT_NAME = next_predict_folder(RUNS_DIR)

# ---------------------------------------------------------
# Încărcăm modelul
# ---------------------------------------------------------
model = YOLO(str(MODEL_PATH))

# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------
model.predict(
    source=str(TEST_IMAGES),
    imgsz=640,
    conf=0.25,
    device="cpu",
    save=True,
    project=str(RUNS_DIR),
    name=PREDICT_NAME
)

# ---------------------------------------------------------
# Final
# ---------------------------------------------------------
print(f"✅ Predict salvat în: {RUNS_DIR / PREDICT_NAME}")
