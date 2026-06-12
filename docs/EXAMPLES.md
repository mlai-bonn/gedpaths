
## Examples of creating edit mappings for different methods
```bash
./CreateMappings \
-db MUTAG -method F2 -method_options threads 30
```

```bash
./CreateMappings \
-db MUTAG -method REFINE -method_options threads 30 max-swap-size 5 lower-bound-method BRANCH
```

## Examples of creating paths with different strategies
```bash
./CreatePaths \
-db MUTAG -method F2 -path_strategy Random
```

```bash
./CreatePaths \
-db MUTAG -method REFINE -path_strategy Random DeleteIsolatedNodes
```

```bash
./CreatePaths \
-db MUTAG -method REFINE -path_strategy InsertEdges DeleteIsolatedNodes
```

```bash
./CreatePaths \
-db MUTAG -method REFINE -path_strategy DeleteEdges DeleteIsolatedNodes
```
