# GEDLIB Methoden und Optionen

Diese Datei fasst die in GEDLIB verfügbaren Methoden (siehe `ged::Options::GEDMethod`) sowie die in den GEDLIB-Headern dokumentierten Methodenoptionen und deren Default-Werte zusammen. Die Angaben stammen aus den Headern unter `libGraph/external/gedlib/src/methods/` im vorliegenden Repository.

Hinweis:
- Einige Methoden (F1, F2, COMPACT_MIP, BLP_NO_EDGE_LABELS) sind nur verfügbar, wenn GEDLIB mit GUROBI-Unterstützung compiliert wurde. Das ist unten jeweils vermerkt.
- Viele Optionen werden vererbt (z. B. aus `LSAPEBasedMethod`, `LSBasedMethod`, `MIPBasedMethod`). Solche geerbten Optionen sind am Anfang zusammengefasst, damit sie nicht für jede Methode wiederholt werden.

---

## Vererbbare (gemeinsame) Optionen

### Optionen von `ged::LSAPEBasedMethod`
(Gilt für LSAPE-basierte Methoden: Bipartite, Branch, BranchFast, Star, Subgraph, ...)

- `--threads <int>` — number of threads; default: `1`.
- `--lsape-model ECBP|EBP|FLWC|FLCC|FBP|SFBP|FBP0` — model for optimal LSAPE solver; default: `ECBP`.
- `--greedy-method BASIC|REFINED|LOSS|BASIC_SORT|INT_BASIC_SORT` — greedy LSAPE solver method; default: `BASIC`.
- `--optimal TRUE|FALSE` — solve optimally or greedily; default: `TRUE`.
- `--centrality-method NONE|DEGREE|EIGENVECTOR|PAGERANK` — node centrality method; default: `NONE`.
- `--centrality-weight <double 0..1>` — weight for centrality; default: `0.7`.
- `--max-num-solutions ALL|<int>` — maximum number of solutions; default: `1`.

(Quelle: `lsape_based_method.hpp`)


### Optionen von `ged::LSBasedMethod`
(Gilt für lokale-Suche-basierte Methoden: Refine, BP_BEAM, IPFP, ... — diese erben zusätzlich die LSAPE-Optionen dank Mehrfachvererbung in GEDLIB)

- `--initialization-method BIPARTITE_ML|BIPARTITE|BRANCH_FAST|BRANCH_UNIFORM|BRANCH|NODE|RING_ML|RING|SUBGRAPH|WALKS|RANDOM` — initial solution method; default: `RANDOM`.
- `--initialization-options '[--<option> <arg>] [...]'` — option string passed to initialization method; default: `''` (leer).
- `--lower-bound-method BRANCH|BRANCH_FAST|BRANCH_TIGHT|NONE` — lower bound method used for termination; default: `NONE`.
- `--random-substitution-ratio <double in (0,1]>` — ratio of substitutions in random initial solutions; default: `1`.
- `--initial-solutions <int>` — number of initial solutions; default: `1`.
- `--ratio-runs-from-initial-solutions <double (0,1]>` — fraction of runs from initial solutions; default: `1`.
- `--threads <int>` — number of threads (used for initialization and parallel LS runs); default: `1`.
- `--num-randpost-loops <int >=0>` — RANDPOST loops; default: `0`.
- `--max-randpost-retrials <int >=0>` — RANDPOST retrials; default: `10`.
- `--randpost-penalty <double 0..1>` — RANDPOST penalty; default: `0`.
- `--randpost-decay <double 0..1>` — RANDPOST decay; default: `1`.
- `--log <file>` — logfile for RANDPOST; default: `""` (none).
- `--randomness REAL|PSEUDO` — randomness type; default: `REAL`.

(Quelle: `ls_based_method.hpp`)


### Optionen von `ged::MIPBasedMethod`
(Gilt für MIP-basierte Methoden: F1, F2, COMPACT_MIP, BLP_NO_EDGE_LABELS, ... — GUROBI-only)

- `--threads <int>` — number of threads for the MIP/LP solver; default: `1`.
- `--time-limit <double>` — time limit in seconds; default: `0` (no limit if <= 0).
- `--relax TRUE|FALSE` — relax integrality (solve continuous); default: `FALSE`.
- `--project-to-node-map TRUE|FALSE` — project continuous solutions to node maps; default: `TRUE`.
- `--tune TRUE|FALSE` — tune solver parameters before optimization; default: `FALSE`.
- `--tune-time-limit <double>` — time limit for parameter tuning; default: `0` (auto if <=0).
- `--map-root-to-root TRUE|FALSE` — enforce mapping of root nodes (node id 0) to each other; default: `FALSE`.
- `--lsape-model ECBP|EBP|FLWC|FLCC|FBP|SFBP|FBP0` — LSAPE model (inherited); default: `ECBP`.

(Quelle: `mip_based_method.hpp`)


---

## Methoden (alphabetisch, `ged::Options::GEDMethod`)

> Hinweis: Die Methodennamen entsprechen den in den GEDLIB-Quellen verwendeten Bezeichnern.

### ANCHOR_AWARE_GED
- Beschreibung: exakter Ansatz / Anchor-aware GED (in GEDLIB als "Anchor-aware GED" bezeichnet).
- Spezifische Optionen: Keine zusätzlichen Optionen in den Headern dokumentiert; erbt die Optionen von `MIPBasedMethod`/`GEDMethod` falls relevant (je nach Implementierung könnte es methodenspezifische Flags geben).

### BIPARTITE
- Beschreibung: LSAPE-basierte Methode (basiert auf einer bipartiten Zuordnung).
- Spezifische Optionen: keine extra Optionen in Header; erbt alle Optionen aus `LSAPEBasedMethod`.

### BIPARTITE_ML
- Beschreibung: ML-gestützte Variante von Bipartite (siehe `bipartite_ml.hpp`).
- Spezifische Optionen: keine zusätzliche, erbt LSAPE/LS-Optionen; spezifische ML-Optionen sind in `bipartite_ml.hpp`/`.ipp` definiert (nicht als Tabelle dokumentiert).

### BL... (GUROBI-only)
- `F1`, `F2`, `COMPACT_MIP`, `BLP_NO_EDGE_LABELS` — MIP-basierte Methoden, nur verfügbar bei GUROBI.
- Gemeinsame Optionen: siehe `MIPBasedMethod` oben.
- Methodenspezifische Optionen: nicht in Tabellenform in den zugehörigen Headern dokumentiert (siehe Implementierungen `f1.hpp`, `f2.hpp`, `compact_mip.hpp`, `blp_no_edge_labels.hpp` für Details).

### BP_BEAM
- Beschreibung: BP-Beam (Beam search kombiniert mit bipartitematching) — liefert Upper bound.
- Spezifische Optionen (zusätzlich zu `LSBasedMethod`):
  - `--beam-size <int> > 0` — default: `5`.
  - `--num-orderings <int> > 0` — default: `1` (wenn >1, IBP-Beam wird verwendet).
  - Erbt LS-based Optionen.
(Quelle: `bp_beam.hpp`)

### BRANCH
- Beschreibung: Branch (LSAPE-basiert) — Lower/Upper bounds.
- Spezifische Optionen: Keine eigene Tabelle im Header; erbt `LSAPEBasedMethod`-Optionen.
(Quelle: `branch.hpp`)

### BRANCH_COMPACT
- Beschreibung: Variante von Branch (kompakt).
- Spezifische Optionen: nicht als Tabelle dokumentiert; erbt `LSAPEBasedMethod`-Optionen.

### BRANCH_FAST
- Beschreibung: Optimierte Branch-Variante.
- Spezifische Optionen (zusätzlich zu `LSAPEBasedMethod`):
  - `--sort-method STD|COUNTING` — default: `COUNTING` (verwende counting sort wenn die Anzahl unterschiedlicher Kantenlabels konstant ist).
(Quelle: `branch_fast.hpp`)

### BRANCH_TIGHT
- Beschreibung: BranchTight (verbesserte Branching-Regeln / Tight bounds).
- Spezifische Optionen:
  - `--iterations <int> >= 0` — maximal number of iterations; default: `20` (if 0, no iteration-based stop).
  - `--time-limit <double>` — seconds; default: `0` (no limit if <= 0).
  - `--range <double>` — range; default: `0` (no range criterion if <= 0).
  - `--epsilon <double>` — epsilon for convergence; default: `0`.
  - `--regularize NAIVE|K-FACTOR` — default: `NAIVE`.
  - `--threads <int>` — default: `1`.
  - `--upper-bound NO|FIRST|LAST|BEST` — default: `BEST`.
(Quelle: `branch_tight.hpp`)

### BRANCH_UNIFORM
- Beschreibung: Branch-Uniform (ein Branch-Lieferant)
- Spezifische Optionen: keine konkrete Tabelle im Header; erbt `LSAPEBasedMethod`-Optionen.

### BP variants / HYBRID / PARTITION
- `PARTITION`, `HYBRID` und ähnliche Methoden: keine eigene Optionstabellen im Header-Comments; erben typischerweise LSAPE- oder GED-Optionen. Details in den jeweiligen Header/implementation files.

### HED
- Beschreibung: HED-Methode
- Spezifische Optionen: nicht als Tabelle dokumentiert; erbt ggf. LS/LSAPE-Optionen.

### IPFP
- Beschreibung: LS-basierte Methode (IPFP).
- Spezifische Optionen: keine eigene Tabelle; erbt `LSBasedMethod`-Optionen.

### NODE
- Beschreibung: Node-basierte LSAPE-Methode.
- Spezifische Optionen: keine eigene Tabelle; erbt `LSAPEBasedMethod`-Optionen.

### PARTITION
- Beschreibung: Partition method.
- Spezifische Optionen: nicht als Tabelle dokumentiert in Header; erbt GED/LSAPE-Optionen.

### REFINE
- Beschreibung: Refine / K-Refine (local search upper bound).
- Spezifische Optionen (zusätzlich zu `LSBasedMethod`):
  - `--max-swap-size <int> >= 2` — maximum swap size; default: `2`. (If >2, K-Refine is used.)
  - `--naive TRUE|FALSE` — naive computation of swap cost; default: `FALSE`.
  - `--add-dummy-assignment TRUE|FALSE` — add dummy assignment to initial node map; default: `TRUE`.
(Quelle: `refine.hpp`)

### RING
- Beschreibung: Ring-based LSAPE method.
- Spezifische Optionen: keine eigene Tabelle; erbt `LSAPEBasedMethod`-Optionen.

### RING_ML
- Beschreibung: ML-gestützte Variante von Ring.
- Spezifische Optionen: siehe `ring_ml.hpp` für ML-spezifische Parameter; keine zentrale Tabelle im Header.

### SIMULATED_ANNEALING
- Beschreibung: Simulated Annealing method.
- Spezifische Optionen: nicht als Tabelle im Header; wahrscheinlich erbt LS/LSAPE-Optionen.

### STAR
- Beschreibung: Star-based LSAPE method (lower/upper bound for uniform costs)
- Spezifische Optionen (zusätzlich zu `LSAPEBasedMethod`):
  - `--sort-method STD|COUNTING` — default: `COUNTING`.
(Quelle: `star.hpp`)

### SUBGRAPH
- Beschreibung: Subgraph-based approximation (bipartite + subgraph extraction)
- Spezifische Optionen:
  - `--load <filename>` — path to existing configuration file; default: not specified.
  - `--save <filename>` — path where to save configuration file; default: not specified.
  - `--subproblem-solver ANCHOR_AWARE_GED|F1|F2|COMPACT_MIP` — solver for subproblems; default: `ANCHOR_AWARE_GED` (F1/F2/COMPACT_MIP only available with GUROBI).
  - `--subproblem-solver-options '[--<option> <arg>] [...]'` — option string passed to subproblem solver; default: `''`.
  - `--depth-range <min>,<max>` — default: `1,5`.
(Quelle: `subgraph.hpp`)

### WALKS
- Beschreibung: Walks-based LSAPE method.
- Spezifische Optionen: keine Tabelle im Header; erbt LSAPE options.

### Weitere Hinweise zu Methoden ohne explizite Tabellen
- Viele Methoden in `all_methods.hpp` (z. B. `branch_compact`, `branch_uniform`, `node`, `bipartite_ml`, `ring_ml`, `ipfp`, `hybrid`, `partition`, ...) haben entweder:
  - keine eigenen per-method Optionstabellen in den Headerkommentaren (dann vererben sie typischerweise die LSAPEBasedMethod/LSBasedMethod/MIPBasedMethod-Optionen), oder
  - die Optionen sind in den Implementationsdateien (`.ipp`) untergebracht oder als Parser-Funktionen (`ls_parse_option_`, `lsape_parse_option_`, `mip_parse_option_`) implementiert — diese enthalten die tatsächliche Optionensyntax, sind aber nicht immer in Tabellen kommentiert.

---

## Wo schauen, wenn du mehr wissen willst
- Die primäre Quelle sind die Header unter `libGraph/external/gedlib/src/methods/`.
- Für detailierte Implementierungs-/Parser-Infos sieh dir die betreffenden `*.hpp`- und `*.ipp`-Dateien an; die Methoden implementieren meist `*_parse_option_` und `*_set_default_options_`, die die genaue Syntax und Defaultwerte festlegen.

---

## Hinweise / Annahmen
- Diese Zusammenstellung gibt die in den Header-Kommentaren dokumentierten Optionen und Default-Werte wieder (Stand: Dateien im Repository). Falls du exakte, maschinenlesbare Default-Werte benötigst, kann ich die jeweiligen `*_set_default_options_`-Implementationen aus den `.ipp`-Dateien auslesen und die Dokumentation präzisieren.
- Wenn du möchtest, kann ich zusätzlich für jede Methode die mögliche `--<option>`-Syntax automatisiert aus den `*_parse_option_`-Implementierungen extrahieren und in tabellarischer Form ergänzen.

---

## Vollständige Methodenliste (Quelle: `all_methods.hpp`)

Die folgende Liste enthält die Methodeklassen, die in `libGraph/external/gedlib/src/methods/all_methods.hpp` referenziert werden. Nutze diese Liste als definitive Übersicht der in GEDLIB verfügbaren Methoden.

- branch_tight (BranchTight)
- anchor_aware_ged (Anchor aware / Exact)
- partition (Partition)
- hybrid (Hybrid)
- branch_compact (BranchCompact)
- hed (HED)
- simulated_annealing (SimulatedAnnealing)

(GUROBI-only MIP methods)
- f1 (F1)  [requires GUROBI]
- f2 (F2)  [requires GUROBI]
- compact_mip (CompactMIP)  [requires GUROBI]
- blp_no_edge_labels (BLPNoEdgeLabels)  [requires GUROBI]

(LSAPE-based / LSAPE-derived)
- bipartite (Bipartite)
- branch (Branch)
- branch_fast (BranchFast)
- branch_uniform (BranchUniform)
- node (Node)
- ring (Ring)
- star (Star)
- subgraph (Subgraph)
- walks (Walks)

(LS-based / local search)
- ipfp (IPFP)
- refine (Refine)
- bp_beam (BPBeam)

(ML-based)
- bipartite_ml (BipartiteML)
- ring_ml (RingML)


---

## Extraktion präziser Default-Werte (optional)

Die hier dokumentierten Default-Werte stammen aus den Header-Kommentaren (`*.hpp`). Für 100% präzise Defaults sollten die jeweiligen `*_set_default_options_()` Implementierungen in den `*.ipp`-Dateien ausgewertet werden. Wenn du möchtest, kann ich das automatisiert für alle Methoden machen und die exakten Default-Werte in JSON/CSV ausgeben. Vorschlag:

- Ich parsen jeweils die `*_set_default_options_()` Implementierung in den `.ipp`-Dateien und extrahiere die gesetzten Variablen/Defaults.
- Ausgabeformat: JSON (Method -> option -> default) und optional CSV.

Sag mir, ob ich die exakte Default-Extraktion jetzt automatisch ausführen soll — falls ja, starte ich die automatisierte Extraktion und füge die Ergebnisse in `Methods.md` oder in separaten `Methods.defaults.json` / `Methods.defaults.csv` Dateien ein.
