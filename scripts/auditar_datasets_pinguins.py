from pathlib import Path
from collections import Counter
import hashlib

# Coloque este script na pasta "Penguin Vision", ao lado de "datasets_originais".
ROOT = Path(__file__).resolve().parent / "datasets_originais"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def find_dataset_roots(root):
    # Cada dataset exportado deve conter data.yaml e train/valid/test.
    return sorted(p.parent for p in root.rglob("data.yaml"))

def inspect_dataset(dataset):
    print("\n" + "=" * 72)
    print(f"DATASET: {dataset.name}")
    print(f"Caminho: {dataset}")
    yaml_path = dataset / "data.yaml"
    if yaml_path.exists():
        print("\ndata.yaml:")
        print(yaml_path.read_text(encoding="utf-8", errors="replace")[:3000])

    image_count = Counter()
    label_count = Counter()
    label_shapes = Counter()
    missing_labels = []
    empty_labels = []
    bad_labels = []
    image_hashes = {}

    for split in ("train", "valid", "val", "test"):
        image_dir = dataset / split / "images"
        label_dir = dataset / split / "labels"
        images = [p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS] if image_dir.exists() else []
        labels = list(label_dir.rglob("*.txt")) if label_dir.exists() else []
        image_count[split] = len(images)
        label_count[split] = len(labels)

        labels_by_stem = {p.stem: p for p in labels}
        for image_path in images:
            # Hash exato para identificar imagens idênticas entre datasets.
            try:
                digest = sha256_file(image_path)
                image_hashes.setdefault(digest, []).append((split, image_path))
            except OSError:
                pass

            label_path = labels_by_stem.get(image_path.stem)
            if label_path is None:
                missing_labels.append(image_path)
                continue
            try:
                raw = label_path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                bad_labels.append(label_path)
                continue
            if not raw:
                empty_labels.append(label_path)
                continue

            for line_no, line in enumerate(raw.splitlines(), 1):
                parts = line.split()
                # Detecção YOLO: class x_center y_center width height (5 valores).
                # Segmentação YOLO: class + coordenadas de vários pontos (mais de 5).
                if len(parts) == 5:
                    label_shapes["linha_bbox_5_valores"] += 1
                elif len(parts) > 5 and len(parts) % 2 == 1:
                    label_shapes["linha_possivel_segmentacao"] += 1
                else:
                    label_shapes["linha_formato_incomum"] += 1
                    bad_labels.append(label_path)
                try:
                    class_id = int(float(parts[0]))
                    if class_id < 0:
                        raise ValueError("class id negativo")
                    coords = [float(x) for x in parts[1:]]
                    if not all(0 <= x <= 1 for x in coords):
                        label_shapes["coordenadas_fora_de_0_a_1"] += 1
                except (ValueError, IndexError):
                    label_shapes["linha_nao_numerica_ou_invalida"] += 1
                    bad_labels.append(label_path)

    print("\nResumo por divisão:")
    for split in ("train", "valid", "val", "test"):
        if image_count[split] or label_count[split]:
            print(f"  {split:5} | imagens: {image_count[split]:5} | arquivos .txt: {label_count[split]:5}")

    print("\nFormato das anotações:")
    if label_shapes:
        for kind, count in label_shapes.items():
            print(f"  {kind}: {count}")
    else:
        print("  Nenhuma anotação .txt identificada.")

    print(f"\nImagens sem arquivo de anotação correspondente: {len(missing_labels)}")
    for p in missing_labels[:5]:
        print(f"  - {p.relative_to(dataset)}")
    print(f"Arquivos de anotação vazios: {len(empty_labels)}")
    print(f"Possíveis anotações inválidas: {len(set(bad_labels))}")

    duplicate_groups = [paths for paths in image_hashes.values() if len(paths) > 1]
    print(f"Grupos de imagens com conteúdo idêntico dentro deste dataset: {len(duplicate_groups)}")
    for group in duplicate_groups[:5]:
        print("  Grupo:")
        for split, p in group:
            print(f"    [{split}] {p.relative_to(dataset)}")

def main():
    if not ROOT.exists():
        print(f"Não encontrei a pasta: {ROOT}")
        print('Coloque este script na pasta "Penguin Vision", ao lado de "datasets_originais".')
        input("\nPressione Enter para sair...")
        return

    datasets = find_dataset_roots(ROOT)
    if not datasets:
        print(f"Nenhum data.yaml encontrado dentro de {ROOT}")
        print("Confira se os três ZIPs foram extraídos e se as pastas estão dentro de datasets_originais.")
        input("\nPressione Enter para sair...")
        return

    print(f"Encontrei {len(datasets)} arquivo(s) data.yaml.")
    for dataset in datasets:
        inspect_dataset(dataset)

    print("\n" + "=" * 72)
    print("AUDITORIA CONCLUÍDA")
    print("Envie o texto exibido para decidir como unificar os datasets com segurança.")
    input("\nPressione Enter para sair...")

if __name__ == "__main__":
    main()
