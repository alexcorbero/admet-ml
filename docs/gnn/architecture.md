# Arquitectura G7 — GNN molecular 3D

## 1. Estructura

```text
admet-ml/
├── data/
│   ├── conformers/
│   ├── graphs/
│   └── results/
│       ├── aqsoldb/
│       ├── gnn/
│       └── comparison/
│
├── src/
│   ├── features/                  # G0–G6
│   ├── gnn/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── atom_features.py
│   │   ├── edge_features.py
│   │   ├── density_features.py
│   │   ├── graph_builder.py
│   │   ├── normalizer.py
│   │   └── dataset.py
│   │
│   └── models/
│       └── gnn.py
│
├── notebooks/
│   ├── 07_gnn_graph_generation.ipynb
│   ├── 08_gnn_training.ipynb
│   └── 09_gnn_evaluation.ipynb
│
├── tests/
│   └── gnn/
│
└── docs/
    └── gnn/
        ├── spec.md
        ├── architecture.md
        ├── tasks.md
        └── reglesIA.md
```

## 2. Flujo

```text
RDKit mol + conformer
        │
        ├─ atom_features.py ─────► z, x_chem
        ├─ density_features.py ──► x_dg
        ├─ edge_features.py ─────► edge_index, edge_attr
        └─ conformer ────────────► pos
                                   │
                                   ▼
                           graph_builder.py
                                   │
                                   ▼
                              PyG Data
                                   │
                                   ▼
                    07_gnn_graph_generation
                         validate + cache
                                   │
                                   ▼
                           raw graph cache
                                   │
                                   ▼
                        08_gnn_training
                  split + normalize + train
                                   │
                                   ▼
                       checkpoint + history
                                   │
                                   ▼
                       09_gnn_evaluation
                    train/validation metrics
                           │            │
                           │            └── classical results
                           ▼
                    common comparison
```

## 3. Dependencias

```text
src/features ───────────────► gnn/density_features

config ─────────────────────► atom_features
                           ├► edge_features
                           ├► density_features
                           ├► graph_builder
                           ├► normalizer
                           └► model

atom_features ──┐
edge_features ──┼───────────► graph_builder ─► dataset
density_features┘

graph_builder + normalizer ─► models/gnn.py
```

Reglas:

- `src/features/` no depende de `src/gnn/`;
- los módulos de features no dependen del modelo;
- `graph_builder.py` no entrena ni normaliza;
- `models/gnn.py` no usa RDKit;
- 07 no entrena;
- 08 no realiza la comparación final;
- 09 no modifica pesos ni normalizers.

## 4. Contratos de módulos

### `config.py`

Define:

```python
GraphConfig
AtomicFeaturesConfig
ModelConfig
TrainingConfig
GNNConfig
```

Debe ser serializable y no depender de RDKit/PyTorch.

### `atom_features.py`

API:

```python
get_atom_feature_names()
build_atom_features(mol)
```

Salida:

```text
z       [N]
x_chem  [N, F_chem]
```

Responsabilidad: química local por átomo.

### `edge_features.py`

API:

```python
get_edge_feature_names(config)
build_edge_features(mol, conf_id, config)
```

Salida:

```text
edge_index  [2, E]
edge_attr   [E, F_edge]
```

Responsabilidad: conectividad covalente/radius/hybrid y atributos químicos/geométricos.

### `density_features.py`

API:

```python
get_density_feature_names(config)
build_density_features(mol, conf_id, config)
```

Salida:

```text
x_dg  [N, F_dg]
```

Responsabilidad: densidad y curvatura local por átomo, sin agregación molecular.

### `graph_builder.py`

API:

```python
get_node_feature_names(config)
get_graph_feature_names(config)
build_graph(...)
```

Responsabilidad: componer un `torch_geometric.data.Data`.

No ajusta scalers.

### `normalizer.py`

API esperada:

```python
GraphFeatureNormalizer.from_config(config)
fit(train_graphs)
transform_graph(graph)
transform_graphs(graphs)
```

Responsabilidad: normalización ajustada exclusivamente con train.

### `dataset.py`

Responsabilidad:

- construir datasets de grafos;
- cachear/cargar grafos crudos;
- mantener metadata;
- seleccionar subsets por IDs de split;
- validar compatibilidad del cache.

No decide los scaffold splits.

### `src/models/gnn.py`

API:

```python
class MolecularGNN(nn.Module):
    def encode(self, data):
        ...

    def forward(self, data):
        ...
```

Arquitectura inicial:

```text
z ─► embedding ─┐
                ├► node encoder
x ──────────────┘
       │
       ▼
message passing × L
       │
       ▼
global pooling
       │
       ├► encode() = molecular embedding
       │
       ▼
readout
       │
       ▼
prediction
```

GINE será el operador inicial de message passing.

## 5. Notebooks

### `07_gnn_graph_generation.ipynb`

Debe:

1. cargar datos/conformeros;
2. definir `GNNConfig`;
3. construir un grafo de ejemplo;
4. mostrar shapes y esquema;
5. generar grafos;
6. registrar fallos;
7. guardar cache y manifest.

Salida principal: **grafos crudos**.

### `08_gnn_training.ipynb`

Debe:

1. cargar grafos;
2. recuperar split/seed;
3. ajustar normalizer con train;
4. crear DataLoaders;
5. instanciar modelo;
6. ejecutar training loop;
7. evaluar validation por época;
8. aplicar early stopping por MAE;
9. guardar best checkpoint y normalizer.

Salida principal: **checkpoint por seed + historial**.

### `09_gnn_evaluation.ipynb`

Debe:

1. cargar checkpoint y normalizer;
2. generar predicciones de train/validation;
3. calcular MAE, RMSE y R²;
4. agregar resultados entre seeds;
5. cargar resultados XGBoost/RandomForest;
6. armonizar esquemas;
7. comparar seed a seed;
8. controlar cobertura;
9. guardar resultados comunes.

Salida principal: **tabla comparativa de modelos**.

## 6. Persistencia

```text
data/graphs/aqsoldb/<graph_config_id>/
├── graphs.pt
└── manifest.*

data/models/gnn/<experiment_id>/<seed>/
├── config.json
├── model.pt
└── normalizer.*

data/results/gnn/<experiment_id>/
├── history_seed_<seed>.csv
├── metrics.csv
├── train_predictions.csv
└── validation_predictions.csv

data/results/comparison/
├── model_metrics.csv
├── model_predictions.csv
└── paired_seed_differences.csv
```

## 7. Esquema común de comparación

Métricas:

```text
model_family
model_name
representation
variant
seed
split
n_samples
coverage
MAE
RMSE
R2
```

Predicciones:

```text
model_family
model_name
representation
variant
seed
split
row_id
smiles
y_true
y_pred
residual
```

`09_gnn_evaluation.ipynb` es responsable de adaptar los resultados clásicos a este esquema.

## 8. Tests mínimos

Se cubrirán:

- configuración;
- atom features;
- edge modes;
- DG features;
- graph builder;
- cache;
- normalizer;
- forward/backward;
- batching;
- checkpoint roundtrip;
- traslación;
- rotación;
- permutación;
- smoke training;
- comparación de esquemas de resultados.

Los notebooks deben ejecutar correctamente desde kernel limpio en modo smoke.
