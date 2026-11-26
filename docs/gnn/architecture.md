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
│   ├── prepare_aqsoldb_conformers.ipynb
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
                  AqSolDB
                     │
                     ▼
        prepare_aqsoldb_conformers
                     │
                     ▼
             RDKit MolFromSmiles
                     │
                     ▼
             Chem.AddHs()
                     │
                     ▼
             Generate conformers
                     │
                     ▼
             Optimize conformers
                     │
                     ▼
          Select lowest valid energy
                     │
                     ▼
            Validate 3D geometry
                     │
              ┌──────┴──────┐
              ▼             ▼
         Valid mols      Excluded mols
              │             │
              ▼             ▼
       conformers.pkl   excluded.csv
              │
              ▼
       07_gnn_graph_generation
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
                    raw graph cache
                           │
                           ▼
                    08_gnn_training
                           │
                           ▼
                  checkpoint + history
                           │
                           ▼
                   09_gnn_evaluation
                           │
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

graph_builder ─► Data
normalizer ───► Data normalizado
                    │
                    ▼
               DataLoader
                    │
                    ▼
               models/gnn.py
```

Reglas:

- `src/features/` no depende de `src/gnn/`;
- los módulos de features no dependen del modelo;
- `graph_builder.py` no entrena ni normaliza;
- `models/gnn.py` no usa RDKit;
- `models/gnn.py` no importa `graph_builder.py` ni `normalizer.py`; recibe objetos Data/Batch ya construidos y, cuando corresponda, normalizados.
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

`AtomicFeaturesConfig.density_transform` controla la transformación numérica
de las features DG continuas (`identity` o `signed_log1p`). Las máscaras de
validez no se transforman.

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

Responsabilidad: conectividad covalente/radius/hybrid y atributos
químicos/geométricos de arista.

La geometría de arista base utiliza distancia 3D y Gaussian RBF.
Los kernels exponential/Lorentz pertenecen a las features DG de nodos y no
forman parte de `edge_attr` en la representación base.

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

El modelo consume:
- z
- x
- edge_index
- edge_attr
- batch

`pos` no forma parte del input directo de GINE.

## 5. Notebooks

### `prepare_aqsoldb_conformers.ipynb`

Responsable de preparar y persistir los conformers 3D de AqSolDB.

Debe:

1. Cargar las particiones oficiales de AqSolDB (`train_val` y `test`).
2. Conservar los identificadores, SMILES, targets y splits originales.
3. Generar moléculas RDKit a partir de los SMILES.
4. Añadir hidrógenos explícitos.
5. Generar varios conformers por molécula mediante ETKDG.
6. Optimizar los conformers utilizando un campo de fuerza definido.
7. Seleccionar el conformer optimizado de menor energía válida y finita.
8. Validar las coordenadas y la geometría 3D.
9. Registrar explícitamente las moléculas excluidas y sus causas.
10. Persistir los conformers seleccionados y sus metadatos.

Debe reutilizar las funciones existentes en `src/features/conformers.py`.

Salida principal:

- `data/conformers/aqsoldb/conformers.pkl`
- `data/conformers/aqsoldb/manifest.json`
- `data/conformers/aqsoldb/excluded.csv`

Cada registro válido debe contener:

- `mol_id`
- `smiles`
- `y`
- `split`
- `seed`
- `mol`
- `conf_id`

El campo `seed` representa la semilla utilizada para generar los conformers, no la semilla de entrenamiento.

El campo `conf_id` debe contener el identificador real del conformer seleccionado en RDKit.

El notebook no construye grafos, no normaliza features y no entrena modelos.

La generación de conformers es una etapa independiente y previa a la generación de grafos.

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
5. cargar resultados XGBoost;
6. armonizar esquemas;
7. comparar seed a seed;
8. controlar cobertura;
9. guardar resultados comunes.

Salida principal: **tabla comparativa de modelos**.

## 6. Persistencia

`experiment_id` será el identificador canónico de una configuración experimental G7. Debe utilizarse de forma consistente en todas las rutas de modelos y resultados.

La estructura de persistencia será:

```text
data/conformers/aqsoldb/
├── conformers.pkl
├── manifest.json
└── excluded.csv

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

Responsabilidades:

- `conformers.pkl` contiene los registros válidos de AqSolDB con sus moléculas RDKit y conformers seleccionados.
- `manifest.json` registra los parámetros utilizados para generar los conformers, la semilla, el campo de fuerza, el número de moléculas procesadas, válidas y excluidas, y la información necesaria para reproducir la generación.
- `excluded.csv` registra las moléculas excluidas y el motivo correspondiente.
- `graph_config_id` identifica de forma reproducible la configuración utilizada para construir los grafos crudos;
- `experiment_id` identifica una configuración experimental GNN;
- `<seed>` identifica una ejecución concreta de entrenamiento dentro del experimento;
- `model.pt` contiene los pesos del mejor modelo y la metadata mínima necesaria del checkpoint;
- `config.json` contiene la configuración completa del experimento;
- `normalizer.*` contiene exclusivamente las estadísticas ajustadas con train para ese seed;
- `history_seed_<seed>.csv` contiene el historial de entrenamiento por época;
- `metrics.csv` contiene las métricas resumidas de las ejecuciones del experimento;
- `train_predictions.csv` y `validation_predictions.csv` contienen las predicciones utilizadas posteriormente por `09_gnn_evaluation.ipynb`.

No se utilizarán alternativamente `<experiment>` y `<experiment_id>`: **`<experiment_id>` es el nombre canónico en toda la documentación y el código**.

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
