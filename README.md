# Parallel Classification Test Environment

Dieses Projekt fuehrt Benchmark-Laeufe fuer mehrere Klassifikationsverfahren aus und speichert alle Ergebnisse in sauber getrennten Run-Ordnern.

Standardmaessig wird fuer jedes Modell eine Cross-Validation-basierte Hyperparametersuche vor dem finalen Test-Run ausgefuehrt.

## Schnellstart

```bash
pip install -e .
python -m pcte_cli --config configs/adult.yaml
```

GPU-/Spezialverfahren erwarten, dass `torch`, `torchgpipe`, `cuml` bzw. `thundersvm` bereits passend in der Zielumgebung installiert sind. Das Projekt faengt fehlende oder inkonsistente Spezial-Runtimes nicht durch Fallbacks ab, sondern bricht den Lauf vor dem ersten Experiment mit einer klaren Preflight-Fehlermeldung ab.

Alternativ kannst du nach der Installation den Konsolenbefehl verwenden:

```bash
pcte-run --config configs/adult.yaml
```

## Verfuegbare Benchmark-Configs

```bash
python -m pcte_cli --config configs/adult.yaml
python -m pcte_cli --config configs/diabetes.yaml
python -m pcte_cli --config configs/skin_non_skin.yaml
python -m pcte_cli --config configs/covtype.yaml
```

## Verfuegbare Skalierungs-Configs

```bash
python -m pcte_cli --config configs/adult_scaling.yaml
python -m pcte_cli --config configs/diabetes_scaling.yaml
python -m pcte_cli --config configs/skin_non_skin_scaling.yaml
python -m pcte_cli --config configs/covtype_scaling.yaml
```

Bei den groessten Datensaetzen `covtype_csv`, `diabetes_csv` und `skin_non_skin_csv` solltest du SVM-Laufzeiten realistisch einplanen. Fuer `covtype_csv` werden die SVM-Varianten deshalb bewusst nur mit linearem Kernel konfiguriert.

## Aktuelle Datensaetze

Im Repo liegen aktuell diese CSV-Datensaetze:

- `adult.csv`
- `diabetes.csv`
- `skin-non-skin.csv`
- `covtype.csv`

Die vorkonfigurierten Loader-Presets dafuer sind:

- `adult_csv`
- `diabetes_csv`
- `skin_non_skin_csv`
- `covtype_csv`

## Mehrere Python-Umgebungen

Wenn Docker im aktuellen Host-/Runpod-Setup nicht verfuegbar ist, kannst du das Projekt direkt im bestehenden Container mit drei getrennten Python-Umgebungen ausfuehren:

- `core`: CPU-Verfahren sowie PyTorch-/GPipe-/FSDP-Verfahren
- `rapids`: cuML- und Multi-GPU-cuML-Verfahren
- `thunder`: ThunderSVC isoliert in einer eigenen Umgebung

Die Datensatz-Configs (`configs/adult.yaml`, `configs/diabetes.yaml`, `configs/skin_non_skin.yaml`, `configs/covtype.yaml`) bleiben dabei unveraendert. Ein kleiner Helfer filtert beim Start automatisch nur die Algorithmen, die zur jeweiligen Runtime gehoeren.

Umgebungen anlegen:

```bash
bash scripts/setup_runtime_venvs.sh
```

Nur eine Runtime fuer einen Datensatz starten:

```bash
bash scripts/run_runtime_env.sh core configs/adult.yaml
bash scripts/run_runtime_env.sh rapids configs/adult.yaml
bash scripts/run_runtime_env.sh thunder configs/adult.yaml
```

Alle Runtime-Gruppen nacheinander fuer einen Datensatz starten:

```bash
bash scripts/run_all_envs.sh configs/adult.yaml
```

Hinweise:

- Die Umgebungen werden unter `.venvs/` angelegt und nutzen `--system-site-packages`, damit vorhandene CUDA-/RAPIDS-Pakete im Host-Container weiterverwendet werden.
- `thundersvm` wird in der `thunder`-Umgebung separat installiert, damit es die anderen Laufzeiten nicht beeinflusst.
- Die Preflight-Pruefung bleibt aktiv: Wenn eine spezialisierte Runtime in ihrer Umgebung nicht korrekt verfuegbar ist, bricht genau dieser Lauf hart ab.

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
- fuer Random Forests stehen sowohl `BaselineRF` und `HybridRF` auf CPU als auch `CuMLRF` auf GPU fuer den Direktvergleich bereit

Zusaetzlich koennen Algorithmen ueber `allowed_datasets` auf sinnvolle Datensaetze begrenzt werden, damit z.B. SVM-Varianten nicht auf unpassenden Big-Data-Lasten laufen.

Fuer CSV-Datensaetze kannst du den Loader `csv_classification` verwenden und `path` plus `target_column` setzen.

Fuer Skalierungsplots kannst du in einem Datensatzblock `train_sizes` oder `train_fractions` definieren. Der Loader behaelt dann einen festen Holdout-Testsplit und erzeugt deterministische, stratified Teilmengen des Trainingssplits, damit Laufzeiten ueber verschiedene Trainingsgroessen vergleichbar bleiben.

Die aktuellen Scaling-Configs nutzen prozentuale Stufen:

`[0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.0]`

Damit entspricht `1.0` immer dem vollen verfuegbaren Trainingssplit fuer den jeweiligen Datensatz.

## GPU-Hinweis

`CuMLKNN` und `CuMLMultiGPUKNN` erwarten eine RAPIDS/cuML-Installation auf dem Server. Die Python-Pipeline faengt fehlende GPU-Abhaengigkeiten sauber ab, installiert RAPIDS aber nicht automatisch.
