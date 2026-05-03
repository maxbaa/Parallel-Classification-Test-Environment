# Parallel Classification Test Environment

Dieses Projekt fuehrt Benchmark-Laeufe fuer mehrere Klassifikationsverfahren aus und speichert alle Ergebnisse in sauber getrennten Run-Ordnern.

Standardmaessig wird fuer jedes Modell eine Cross-Validation-basierte Hyperparametersuche vor dem finalen Test-Run ausgefuehrt.

## Schnellstart

```bash
pip install -e .
python -m pcte_cli --config configs/default_experiment.yaml
```

Alternativ kannst du nach der Installation den Konsolenbefehl verwenden:

```bash
pcte-run --config configs/default_experiment.yaml
```

Fuer einen schnellen Funktionstest ohne langen Vollbenchmark:

```bash
python -m pcte_cli --config configs/smoke_test.yaml
```

Fuer den 2-GPU-Serverlauf mit den CSV-Datasets:

```bash
python -m pcte_cli --config configs/two_gpu_server.yaml
```

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
- wie oft jeder Run wiederholt wird
- welche Hyperparameter via Cross-Validation durchsucht werden
- wohin Ergebnisse geschrieben werden

Fuer CSV-Datensaetze kannst du den Loader `csv_classification` verwenden und `path` plus `target_column` setzen.

Vorkonfigurierte CSV-Presets:

- `breast_cancer_csv`
- `covtype_csv`
- `airline_csv`
- `higgs_csv`

## GPU-Hinweis

`CuMLKNN` und `CuMLMultiGPUKNN` erwarten eine RAPIDS/cuML-Installation auf dem Server. Die Python-Pipeline faengt fehlende GPU-Abhaengigkeiten sauber ab, installiert RAPIDS aber nicht automatisch.
