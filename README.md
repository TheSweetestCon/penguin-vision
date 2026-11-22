# Penguin Vision

Detecção de pinguins em fotografias com **YOLO11n**.
Projeto – Ciência e Tecnologia, Universidade Feevale.
Autor: Matheus Haag da Encarnação.

## Resultados (conjunto de teste: 97 imagens, 340 instâncias)

| Precision | Recall | mAP@50 | mAP@50-95 |
|-----------|--------|--------|-----------|
| 88,9% | 78,1% | 87,5% | 62,7% |

Na validação (196 imagens): mAP@50 de 72,7% e F1 máximo de 0,70 com confiança 0,262.
O resultado deve ser lido como **prova de conceito**: há falsos positivos, caixas
duplicadas ou imprecisas e dificuldade em cenas densas, com pinguins pequenos.

## Conteúdo do repositório

| Pasta | Descrição |
|-------|-----------|
| `scripts/` | Auditoria e unificação dos datasets |
| `dataset_config/` | `data.yaml` e registros da unificação |
| `weights/best.pt` | Pesos treinados (YOLO11n, 30 épocas) |
| `results/treino/` | `results.csv` e `results.png` (curvas de treinamento) |
| `results/val*/` | Matriz de confusão, curvas P/R/F1/PR e lotes de validação |
| `docs/` | Relatório final em PDF |

As imagens estão incluídas em `datasets_originais/` (como baixadas do Roboflow) e em `dataset_combinado/` (base unificada).

## Datasets (origem: Roboflow Universe, formato YOLO)

| Dataset | Licença | Link |
|---------|---------|------|
| Penguin Finder v1 | MIT | https://universe.roboflow.com/rf100-vl/penguin-finder-seg-va0wf-ygkoq-oznn/dataset/1 |
| Penguins (WingCV) v1 | Public Domain | https://universe.roboflow.com/wingcv/penguins-6hqac/dataset/1 |
| Yolo5-Penguins v2 | CC BY 4.0 | https://universe.roboflow.com/yaron-naim-5oxmj/yolo5-penguins/dataset/2 |

Base unificada: 974 imagens (681 treino, 196 validação, 97 teste), classe única `penguin`.
Atribuição: imagens e anotações pertencem aos autores dos datasets acima, conforme as licenças informadas.

## Como reproduzir

1. Os três datasets já estão em `datasets_originais/` (ou baixe-os de novo nos links acima).
2. Auditoria: `python scripts/auditar_datasets_pinguins.py`
3. Unificação: `python scripts/unificar_datasets_pinguins.py` (gera `dataset_combinado/`)
4. Treinamento (a partir de `dataset_combinado/`):

```
yolo detect train model=yolo11n.pt data=data.yaml epochs=30 imgsz=640 device=cpu
```

5. Avaliação:

```
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml split=val plots=True
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml split=test
```

## Ambiente utilizado

Windows, Python 3.14.6, Ultralytics 8.4.191, PyTorch 2.14.1 (CPU), Intel Core i7-11700K.
O treinamento levou cerca de 1 h 16 min em CPU.

## Observação

O Ultralytics YOLO é distribuído sob a licença AGPL-3.0 (com opção de licença comercial).