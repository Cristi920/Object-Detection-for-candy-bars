# ---------------------------------------------------------
# Importăm moduele necesare
# ---------------------------------------------------------
from ultralytics import YOLO
import torch
from pathlib import Path
import shutil

# ---------------------------------------------------------
# Directorul principal al proiectului
# Conține dataset-ul și fișierul data.yaml
# ---------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Users\clisu\Desktop\New folder")


# Calea către fișierul YAML folosit la training
DATA_YAML = PROJECT_ROOT / "data" / "data.yaml"

# ---------------------------------------------------------
# Folderul unde YOLO va salva rezultatele training-ului
# ---------------------------------------------------------
RUNS_DIR = PROJECT_ROOT / "data" / "runs"
TRAIN_NAME = "train"

# ---------------------------------------------------------
# Detectăm automat dacă există GPU CUDA
# Dacă nu, training-ul rulează pe CPU
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Using device: {device}")



# ---------------------------------------------------------
# Încărcăm modelul YOLO pre-antrenat
# Poți schimba cu variante mai mari pentru acuratețe mai bună
# ---------------------------------------------------------
model = YOLO("yolo11l.pt")  # poți schimba cu yolo11s.pt pentru mai multă acuratețe

# Cu oricare din următoarele:
#YOLO("yolo11n.pt")  # small – deja ai folosit
#YOLO("yolo11s.pt")  # small – deja ai folosit
#YOLO("yolo11m.pt")  # medium – mai precis, mai lent
#YOLO("yolo11l.pt")  # large – foarte precis, VRAM mare necesar
#YOLO("yolo11x.pt")  # extra large – maxim de precizie, VRAM >8GB recomandat

# ---------------------------------------------------------
# Funcția principală de antrenare
# Configurează parametrii și pornește training-ul
# ---------------------------------------------------------
def run():
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

# Pornim procesul de training YOLO
    results = model.train(
        data=str(DATA_YAML),  # config dataset
        epochs=60,            # număr epoci
        imgsz=640,            # rezoluție imagini
        batch=2,              # batch size (mic pentru GPU slab)
        device=device,        # CPU sau CUDA
        amp=False,            # mixed precision pentru viteză
        workers=0,            # evită probleme pe Windows
    )

    print(f"✅ Training finalizat. Rezultate în: {RUNS_DIR / TRAIN_NAME}")

if __name__ == "__main__":
    run()