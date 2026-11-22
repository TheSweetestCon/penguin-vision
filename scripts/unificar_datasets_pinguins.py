from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import shutil

# Estrutura esperada:
# Penguin Vision/
#   unificar_datasets_pinguins.py
#   datasets_originais/
#     pasta_dataset_1/data.yaml, train/images, train/labels, valid/images, ...
#     pasta_dataset_2/...
#     pasta_dataset_3/...
#
# O resultado será criado em dataset_combinado/.
ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "datasets_originais"
OUTPUT_ROOT = ROOT / "dataset_combinado"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_NAMES = ("train", "valid", "val", "test")
# Se uma imagem duplicada aparece em splits diferentes, damos prioridade ao teste,
# depois validação, depois treino, para evitar vazamento entre conjuntos.
SPLIT_PRIORITY = {"test": 0, "valid": 1, "val": 1, "train": 2}

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_class_names(dataset_dir):
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        return []
    # Leitura simples de `names: [...]` sem depender de biblioteca YAML.
    text = yaml_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if line.strip().startswith("names:"):
            value = line.split(":", 1)[1].strip()
            if value.startswith("[") and value.endswith("]"):
                inside = value[1:-1].strip()
                if not inside:
                    return []
                return [x.strip().strip("'\"") for x in inside.split(",")]
            if value.startswith("{"):
                return ["Penguin"]
    return []

def get_split_dir(dataset_dir, split):
    # Alguns datasets usam `valid`; outros usam `val`.
    candidates = [split]
    if split == "valid":
        candidates = ["valid", "val"]
    elif split == "val":
        candidates = ["val", "valid"]
    for candidate in candidates:
        image_dir = dataset_dir / candidate / "images"
        label_dir = dataset_dir / candidate / "labels"
        if image_dir.exists():
            return candidate, image_dir, label_dir
    return None, None, None

def normalize_label(path):
    """Retorna anotações normalizadas; classes viram classe única 0."""
    if not path.exists():
        return None, "arquivo de anotação ausente"
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return [], None  # Arquivo vazio pode significar imagem sem pinguins.
    normalized = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            return None, f"linha {line_number}: esperado formato bbox YOLO de 5 valores, recebido {len(parts)}"
        try:
            old_class = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            return None, f"linha {line_number}: valor não numérico"
        if old_class < 0 or not all(0 <= v <= 1 for v in coords):
            return None, f"linha {line_number}: classe/coordenadas inválidas"
        x, y, w, h = coords
        if w <= 0 or h <= 0:
            return None, f"linha {line_number}: largura/altura precisam ser maiores que zero"
        normalized.append("0 " + " ".join(f"{v:.6f}" for v in coords))
    # Ordenar ajuda a comparar anotações equivalentes com linhas em ordem diferente.
    return sorted(normalized), None

def main():
    if not SOURCE_ROOT.exists():
        print(f"ERRO: pasta não encontrada: {SOURCE_ROOT}")
        input("Pressione Enter para sair...")
        return

    dataset_dirs = sorted(p.parent for p in SOURCE_ROOT.rglob("data.yaml"))
    if not dataset_dirs:
        print(f"ERRO: nenhum data.yaml encontrado em {SOURCE_ROOT}")
        input("Pressione Enter para sair...")
        return

    if OUTPUT_ROOT.exists():
        print(f"A pasta de saída já existe:\n{OUTPUT_ROOT}")
        print("Para evitar sobrescrever arquivos, renomeie ou apague manualmente essa pasta antes de rodar novamente.")
        input("Pressione Enter para sair...")
        return

    print(f"Datasets encontrados: {len(dataset_dirs)}")
    records = []
    issues = []
    licenses = []
    for dataset_dir in dataset_dirs:
        names = read_class_names(dataset_dir)
        license_name = "não identificada"
        yaml_text = (dataset_dir / "data.yaml").read_text(encoding="utf-8", errors="replace")
        for line in yaml_text.splitlines():
            if "license:" in line:
                license_name = line.split("license:", 1)[1].strip()
                break
        licenses.append((dataset_dir.name, license_name))
        print(f"\nLendo: {dataset_dir.name} | classes originais: {names or 'não identificadas'} | licença: {license_name}")

        for split_request in ("train", "valid", "test"):
            actual_split, image_dir, label_dir = get_split_dir(dataset_dir, split_request)
            if image_dir is None:
                print(f"  Aviso: divisão '{split_request}' não encontrada.")
                continue
            images = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            print(f"  {actual_split}: {len(images)} imagens")
            label_map = {p.stem: p for p in label_dir.rglob("*.txt")} if label_dir.exists() else {}
            for image_path in images:
                label_path = label_map.get(image_path.stem)
                label_lines, error = normalize_label(label_path) if label_path else (None, "arquivo de anotação ausente")
                if error:
                    issues.append((dataset_dir.name, actual_split, str(image_path), error))
                    continue
                try:
                    digest = sha256_file(image_path)
                except OSError as e:
                    issues.append((dataset_dir.name, actual_split, str(image_path), f"não foi possível ler imagem: {e}"))
                    continue
                records.append({
                    "dataset": dataset_dir.name,
                    "split": "valid" if actual_split == "val" else actual_split,
                    "image": image_path,
                    "labels": label_lines,
                    "hash": digest,
                })

    # Agrupar imagens com conteúdo binário idêntico entre todos os datasets.
    by_hash = defaultdict(list)
    for record in records:
        by_hash[record["hash"]].append(record)

    kept = []
    duplicate_groups = 0
    conflict_groups = 0
    for digest, group in by_hash.items():
        splits = {r["split"] for r in group}
        annotation_versions = {tuple(r["labels"]) for r in group}
        if len(group) > 1:
            duplicate_groups += 1
        if len(annotation_versions) > 1:
            conflict_groups += 1
            issues.append((
                "DUPLICATA_COM_ANOTACOES_DIFERENTES",
                ",".join(sorted(splits)),
                digest,
                "Mesma imagem exata, mas as anotações diferem. Grupo ignorado para revisão manual."
            ))
            continue
        # Escolhe uma cópia e atribui todas as cópias idênticas à divisão mais restritiva.
        chosen_split = min(splits, key=lambda s: SPLIT_PRIORITY[s])
        chosen = sorted(group, key=lambda r: (SPLIT_PRIORITY[r["split"]], r["dataset"], str(r["image"])))[0].copy()
        chosen["split"] = chosen_split
        kept.append(chosen)

    # Se não sobrar nenhum arquivo válido, não cria uma saída vazia.
    if not kept:
        print("\nNenhuma imagem válida para combinar. Verifique os erros abaixo.")
        for issue in issues[:30]:
            print(" - ", issue)
        input("Pressione Enter para sair...")
        return

    for split in ("train", "valid", "test"):
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    counters = Counter()
    for record in kept:
        # Nome estável e único derivado do hash do conteúdo.
        stem = record["hash"][:16]
        ext = record["image"].suffix.lower()
        dest_image = OUTPUT_ROOT / "images" / record["split"] / f"{stem}{ext}"
        dest_label = OUTPUT_ROOT / "labels" / record["split"] / f"{stem}.txt"
        shutil.copy2(record["image"], dest_image)
        dest_label.write_text("\n".join(record["labels"]) + ("\n" if record["labels"] else ""), encoding="utf-8")
        counters[record["split"]] += 1

    yaml_content = f"""path: {OUTPUT_ROOT.as_posix()}
train: images/train
val: images/valid
test: images/test

nc: 1
names: ['penguin']
"""
    (OUTPUT_ROOT / "data.yaml").write_text(yaml_content, encoding="utf-8")

    readme = [
        "DATASET COMBINADO DE DETECCAO DE PINGUINS",
        "",
        "Classe única: penguin (id 0).",
        "Formato: bounding boxes YOLO (class x_center y_center width height, coordenadas normalizadas).",
        "",
        "Fontes e licenças declaradas nos data.yaml originais:",
    ]
    for name, license_name in licenses:
        readme.append(f"- {name}: {license_name}")
    readme += [
        "",
        "Confira as páginas originais e os termos de cada fonte antes de redistribuir este conjunto.",
        "Imagens binariamente idênticas foram deduplicadas e mantidas em apenas uma divisão.",
        "Grupos de imagens idênticas com anotações diferentes foram ignorados e registrados no relatório.",
        "Arquivos com anotações ausentes ou inválidas foram ignorados e registrados no relatório.",
    ]
    (OUTPUT_ROOT / "README_origens.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")

    report_lines = [
        "RELATORIO DE UNIFICACAO",
        f"Imagens lidas com anotacoes validas: {len(records)}",
        f"Grupos de imagens duplicadas: {duplicate_groups}",
        f"Grupos duplicados com anotacoes conflitantes e ignorados: {conflict_groups}",
        f"Imagens finais: {len(kept)}",
        f"Treino: {counters['train']}",
        f"Validacao: {counters['valid']}",
        f"Teste: {counters['test']}",
        f"Itens ignorados/problemas: {len(issues)}",
        "",
        "DETALHES DOS PROBLEMAS:",
    ]
    for issue in issues:
        report_lines.append(" | ".join(map(str, issue)))
    (OUTPUT_ROOT / "relatorio_unificacao.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 68)
    print("UNIFICACAO CONCLUIDA")
    print(f"Saida: {OUTPUT_ROOT}")
    print(f"Imagens finais: {len(kept)}")
    print(f"  Treino:    {counters['train']}")
    print(f"  Validacao: {counters['valid']}")
    print(f"  Teste:     {counters['test']}")
    print(f"Grupos de duplicatas identificados: {duplicate_groups}")
    print(f"Grupos com anotacoes conflitantes ignorados: {conflict_groups}")
    print(f"Itens com problemas/ignorados: {len(issues)}")
    print("\nLeia relatorio_unificacao.txt e README_origens.txt na pasta de saída.")
    input("\nPressione Enter para sair...")

if __name__ == "__main__":
    main()
