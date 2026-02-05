# ---------------------------------------------------------
# Importăm moduele necesare
# ---------------------------------------------------------
import yaml
from pathlib import Path


# ---------------------------------------------------------
# Directorul principal al proiectului
# Toate fișierele se vor genera relativ la această cale
# ---------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Users\clisu\Desktop\New folder")


# ---------------------------------------------------------
# Creăm un dumper YAML custom care permite liste inline
# (ex: names: [cat, dog] în loc de listă pe mai multe rânduri)
# ---------------------------------------------------------
class InlineListDumper(yaml.SafeDumper):
    pass


# ---------------------------------------------------------
# Funcție care forțează afișarea listelor în stil inline
# ---------------------------------------------------------
def inline_list_representer(dumper, data):
    return dumper.represent_sequence(
        'tag:yaml.org,2002:seq',
        data,
        flow_style=True
    )
    
    
# Înregistrăm reprezentarea custom pentru liste
InlineListDumper.add_representer(list, inline_list_representer)


# ---------------------------------------------------------
# Funcția principală care creează fișierul data.yaml
# ---------------------------------------------------------
def create_data_yaml():
    classes_txt = PROJECT_ROOT / "custom_data" / "classes.txt"

    output_yaml_root = PROJECT_ROOT / "data.yaml"
    output_yaml_data = PROJECT_ROOT / "data" / "data.yaml"

    if not classes_txt.exists():
        print(f"❌ classes.txt nu există la {classes_txt}")
        return

    with open(classes_txt, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    data = {
        "path": str(PROJECT_ROOT / "data"),
        "train": str(PROJECT_ROOT / "data" / "train" / "images"),
        "val": str(PROJECT_ROOT / "data" / "validation" / "images"),
        "nc": len(classes),
        "names": classes
    }

    # asigură existența folderului /data
    output_yaml_data.parent.mkdir(parents=True, exist_ok=True)

    for output_yaml in [output_yaml_root, output_yaml_data]:
        with open(output_yaml, "w", encoding="utf-8") as f:
            yaml.dump(
                data,
                f,
                Dumper=InlineListDumper,
                sort_keys=False,
                allow_unicode=True,
                width=1000
            )

    print("✅ data.yaml generat în ambele locații:")
    print(f"📄 {output_yaml_root}")
    print(f"📄 {output_yaml_data}\n")

    print("📄 Conținut data.yaml:\n")
    print(yaml.dump(
        data,
        Dumper=InlineListDumper,
        sort_keys=False,
        width=1000
    ))

if __name__ == "__main__":
    create_data_yaml()
