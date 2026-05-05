# Parallel Classification Test Environment

Dieses Projekt fuehrt Benchmark-Laeufe fuer mehrere Klassifikationsverfahren aus und speichert alle Ergebnisse in sauber getrennten Run-Ordnern.

Standardmaessig wird fuer jedes Modell eine Cross-Validation-basierte Hyperparametersuche vor dem finalen Test-Run ausgefuehrt.

## Schnellstart

```bash
pip install -e .
python -m pcte_cli --config configs/breast_cancer.yaml
```

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
