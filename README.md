# Parallel Classification Test Environment

Dieses Projekt fuehrt Benchmark-Laeufe fuer mehrere Klassifikationsverfahren aus und speichert alle Ergebnisse in sauber getrennten Run-Ordnern.

Standardmaessig wird fuer jedes Modell eine Cross-Validation-basierte Hyperparametersuche vor dem finalen Test-Run ausgefuehrt.

## Schnellstart

```bash
pip install -e .
python -m pcte_cli --config configs/breast_cancer.yaml
```

GPU-/Spezialverfahren erwarten, dass `torch`, `torchgpipe`, `cuml` bzw. `thundersvm` bereits passend in der Zielumgebung installiert sind. Das Projekt faengt fehlende oder inkonsistente Spezial-Runtimes nicht durch Fallbacks ab, sondern bricht den Lauf vor dem ersten Experiment mit einer klaren Preflight-Fehlermeldung ab.

Alternativ kannst du nach der Installation den Konsolenbefehl verwenden:

```bash
pcte-run --config configs/breast_cancer.yaml
```

Verfuegbare Benchmark-Configs pro Datensatz:

```bash
python -m pcte_cli --config configs/breast_cancer.yaml
python -m pcte_cli --config configs/adult.yaml
python -m pcte_cli --config configs/covtype.yaml
```

Beim groessten Datensatz `covtype_csv` werden die SVM-Varianten bewusst nur mit linearem Kernel konfiguriert.

## Container-Setup

Um die Umgebung handhabbar zu halten, ist das Projekt in drei Runtime-Container aufgeteilt:

- `core`: CPU-Verfahren sowie PyTorch-/GPipe-/FSDP-Verfahren
- `rapids`: cuML- und Multi-GPU-cuML-Verfahren
- `thunder`: ThunderSVC isoliert in einem eigenen Container

Die normalen Datensatz-Configs (`configs/breast_cancer.yaml`, `configs/adult.yaml`, `configs/covtype.yaml`) bleiben dabei bestehen. Ein kleiner Helfer filtert beim Start automatisch nur die Algorithmen, die zur jeweiligen Runtime gehoeren.

Images bauen:

```bash
docker compose build core
docker compose build rapids
docker compose build thunder
```

Nur eine Runtime fuer einen Datensatz starten:

```bash
docker compose run --rm --gpus all core \
  python scripts/run_runtime_suite.py --runtime core --config configs/adult.yaml

docker compose run --rm --gpus all rapids \
  python scripts/run_runtime_suite.py --runtime rapids --config configs/adult.yaml

docker compose run --rm --gpus all thunder \
  python scripts/run_runtime_suite.py --runtime thunder --config configs/adult.yaml
```

Alle Runtime-Gruppen nacheinander fuer einen Datensatz starten:

```bash
bash scripts/run_all_runtimes.sh configs/adult.yaml
```

Hinweise:

- Der `thunder`-Container baut `thundersvm` bewusst getrennt vom Rest, weil diese Laufzeit in der gemeinsamen Umgebung am fragilsten ist.
- Falls dein RAPIDS-Basisimage einen anderen Tag braucht, kannst du `RAPIDS_BASE_IMAGE` vor dem Build setzen.
- Die Preflight-Pruefung bleibt aktiv: Wenn eine spezialisierte Runtime in ihrem Container nicht korrekt verfuegbar ist, bricht genau dieser Lauf hart ab.

## Ergebnisstruktur

Jeder Lauf landet unter `results/<timestamp>_<run_name>/`:

- `raw/runs.csv`: Rohdaten aller Einzelruns
- `summary/aggregated_metrics.csv`: Aggregierte Kennzahlen
- `summary/summary.md`: Kurze Auswertung
- `summary/run_metadata.json`: Metadaten zum Lauf und System
- `plots/*.png`: Automatisch erzeugte Diagramme
- `logs/failures.json`: Fehler pro Run, falls etwas schiefgeht
- `config/*`: Kopie der verwendeten Konfiguration

## Konfiguration

Die YAML-Datei steuert:

- welche Algorithmen laufen
- welche Datensaetze verwendet werden
- welche Szenarien getestet werden
- in welchen Szenarien ein Algorithmus ueber `allowed_scenarios` ueberhaupt laufen darf
- wie oft jeder Run wiederholt wird
- welche Hyperparameter via Cross-Validation durchsucht werden
- wohin Ergebnisse geschrieben werden

Typisches Muster fuer einen wissenschaftlich saubereren Vergleich:

- `cpu_sequential` fuer sequentielle Referenzverfahren
- `cpu_parallel` fuer explizit CPU-parallele Varianten
- `gpu_single` fuer GPU-native Einzelgeraeteverfahren
- `gpu_multi_2x` nur fuer Verfahren mit echter 2-GPU-Unterstuetzung
- bei neuronalen Netzen sollte die Referenz sinnvollerweise ebenfalls auf `gpu_single` laufen, z.B. als PyTorch-Single-GPU-Baseline; verteilte Varianten wie `FSDPMLP` und `GPipeMLP` koennen separat auf `gpu_multi_2x` laufen
- fuer Random Forests stehen nun sowohl `BaselineRF` und `HybridRF` auf CPU als auch `CuMLRF` auf GPU fuer den Direktvergleich bereit

Zusaetzlich koennen Algorithmen ueber `allowed_datasets` auf sinnvolle Datensaetze begrenzt werden, damit z.B. SVM-Varianten nicht auf unpassenden Big-Data-Lasten laufen.

Fuer CSV-Datensaetze kannst du den Loader `csv_classification` verwenden und `path` plus `target_column` setzen.

Vorkonfigurierte CSV-Presets:

- `breast_cancer_csv`
- `adult_csv`
- `airline_csv`
- `covtype_csv`

## GPU-Hinweis

`CuMLKNN` und `CuMLMultiGPUKNN` erwarten eine RAPIDS/cuML-Installation auf dem Server. Die Python-Pipeline faengt fehlende GPU-Abhaengigkeiten sauber ab, installiert RAPIDS aber nicht automatisch.
