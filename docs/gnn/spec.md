# Especificación G7 — GNN molecular 3D

## 1. Objetivo

Implementar desde cero una **Graph Neural Network molecular 3D** para predecir propiedades ADMET. La primera aplicación será la predicción de solubilidad acuosa (`logS`) sobre `Solubility_AqSolDB`.

Cada molécula se representa mediante un grafo construido a partir de un conformero 3D:

- nodos = átomos;
- aristas = enlaces químicos y/o vecinos espaciales;
- nodos = química local + densidad/curvatura opcional;
- aristas = química de enlace + geometría 3D;
- message passing = aprendizaje de la representación;
- pooling = embedding molecular;
- readout = predicción escalar.

El encoder debe exponer `encode()` para poder reutilizar el embedding molecular en futuras arquitecturas de fusión con descriptores globales.

La fusión mediante Transformer/attention queda fuera de G7.

## 2. Datos y conformeros

La entrada mínima por molécula será:

- identificador estable;
- SMILES;
- target experimental;
- split/seed;
- `Chem.Mol`;
- conformero 3D válido.

Requisitos del conformero:

- generación reproducible mediante semilla;
- misma geometría para nodos, aristas y features geométricas;
- política única de hidrógenos;
- no añadir, eliminar ni reordenar átomos después de fijar el conformero;
- rechazar coordenadas no finitas o geometrías degeneradas de forma explícita;
- contabilizar cualquier molécula excluida.

### 2.1. Selección del conformero

G7 utilizará inicialmente **un único conformero por molécula**.

El conformero utilizado será el **conformero optimizado de menor energía** entre los conformeros generados con energía válida y finita.

La selección debe conservar explícitamente la correspondencia entre:

```text
conf_ids[i] ↔ optimization_results[i]
```

donde `conf_ids[i]` es el identificador real asignado por RDKit al conformero correspondiente.

La función encargada de seleccionar el conformero de menor energía debe devolver el **RDKit `conf_id` real**, no la posición del conformero dentro de la lista de resultados.

Conceptualmente:

```python
best_position = argmin(energies)
best_conf_id = conf_ids[best_position]
```

Todas las operaciones posteriores de G7 deben utilizar ese mismo `best_conf_id`:

- extracción de `pos`;
- construcción de aristas geométricas;
- cálculo de distancias;
- densidad;
- curvatura;
- construcción final del grafo.

El `conf_id` seleccionado debe almacenarse en los metadatos del grafo y/o del manifest de generación para permitir reproducir exactamente la representación.

Si no existe ningún conformero con geometría y energía válidas, la molécula se excluye de forma explícita y se registra la causa.

### Política de hidrógenos

Todos los módulos G7 reciben la misma molécula RDKit con hidrógenos explícitos:

```text
Chem.AddHs()
    ↓
conformer
    ↓
atom_features
edge_features
density_features
graph_builder
```

## 3. Representación del grafo

### 3.1. Nodos

Cada átomo RDKit corresponde a un nodo.

Se almacenará:

```text
z       número atómico
x_chem  features químicas locales
x_dg    densidad/curvatura opcional
x       concatenación usada por el modelo
pos     coordenadas 3D
```

Features químicas iniciales:

- carga formal;
- grado respecto a átomos pesados;
- H asociados;
- masa atómica;
- aromaticidad;
- pertenencia a anillo;
- hibridación one-hot;
- quiralidad/CIP one-hot.

El número atómico se conserva en `z` y se procesa mediante embedding.

No se repetirán descriptores moleculares globales como MolWt, LogP o TPSA en cada nodo.

### 3.2. Densidad y geometría diferencial

G7 podrá utilizar las formulaciones locales de G6 **antes de la agregación molecular**.

Por átomo y canal fuente se podrán incluir:

- `rho`;
- `H`;
- `K`;
- `kappa_min`;
- `kappa_max`;
- máscara de validez.

Las curvaturas inválidas usarán valor neutro + máscara explícita.

No se usarán estadísticas moleculares G6 (`mean`, `std`, cuantiles, etc.) como node features.

Kernels admitidos:

- `exponential`;
- `lorentz`.

La transformación numérica de las features continuas DG es configurable:

- `identity`;
- `signed_log1p`, aplicada como `sign(x) * log1p(abs(x))`.

La transformación se aplica a `rho` y a las curvaturas, pero no a la máscara
binaria `curvature_valid`.

### 3.3. Aristas

Modos soportados:

- `covalent`;
- `radius`;
- `hybrid`.

El modo principal será `hybrid`.

Requisitos:

- aristas bidireccionales;
- sin duplicados;
- sin self-loops;
- los enlaces RDKit se conservan en `hybrid`;
- los vecinos espaciales se determinan por cutoff configurable.

Features de arista:

- `is_bonded`;
- tipo de enlace one-hot;
- aromaticidad;
- conjugación;
- pertenencia a anillo;
- distancia 3D;
- Gaussian RBF de la distancia.

La representación base de aristas no utiliza kernels exponential/Lorentz.

Los kernels exponential/Lorentz se reservan para las features locales de
densidad y geometría diferencial de los nodos.

## 4. Objeto PyTorch Geometric

La construcción debe producir:

```python
Data(
    z=...,          # [N]
    x=...,          # [N, F_node]
    x_chem=...,     # [N, F_chem]
    x_dg=...,       # [N, F_dg]
    pos=...,        # [N, 3]
    edge_index=..., # [2, E]
    edge_attr=...,  # [E, F_edge]
    y=...,          # [1]
)
```

Metadatos opcionales:

- `smiles`;
- `mol_id`;
- `conf_id`.

Los nombres y el orden de las features deben poder recuperarse programáticamente.

## 5. Normalización

La normalización se ajustará **solo con train**.

No se normalizarán:

- `z`;
- one-hot;
- flags binarios;
- máscaras de validez.

Las features continuas podrán estandarizarse según configuración.

Las curvaturas inválidas deben seguir siendo distinguibles después de normalizar.

El normalizer debe guardarse junto al checkpoint correspondiente.

## 6. Modelo

La primera arquitectura será una **GNN de message passing sensible a `edge_attr`**, usando GINE como baseline inicial.

Debe incluir:

1. embedding de `z`;
2. encoder de nodo;
3. varias capas de message passing;
4. uso explícito de `edge_index` y `edge_attr`;
5. activación y normalización;
6. residuals configurables;
7. dropout;
8. pooling molecular;
9. readout escalar.

API obligatoria:

```python
encode(data) -> graph_embedding
forward(data) -> prediction
```

`pos` se conserva en `Data` como geometría molecular y para la
construcción/verificación de features geométricas, pero no se concatena
directamente con `x` ni se introduce como coordenadas cartesianas en GINE.

La información 3D llega al modelo mediante features invariantes derivadas
de la geometría, como distancias, Gaussian RBF, densidad y curvatura.

## 7. Flujo experimental

### `07_gnn_graph_generation.ipynb`

Responsable de:

- construir grafos;
- validarlos;
- contabilizar fallos;
- guardar grafos crudos y metadata.

No normaliza ni entrena.

### `08_gnn_training.ipynb`

Responsable de:

- cargar grafos;
- recuperar scaffold splits;
- ajustar el normalizer solo con train;
- crear DataLoaders;
- entrenar;
- aplicar early stopping;
- seleccionar por validation MAE;
- guardar checkpoint, normalizer e historial.

Losses permitidas:

- Huber;
- MAE;
- MSE como ablation.

### `09_gnn_evaluation.ipynb`

Responsable de:

- cargar el mejor checkpoint;
- generar predicciones de train y validation;
- calcular MAE, RMSE y R²;
- analizar cobertura y residuals;
- comparar GNN con ML clásico.

## 8. Comparación con ML clásico

La comparación incluirá cuando exista resultado compatible:

- XGBoost;
- G7-A;
- G7-B;
- G7-C.

Esquema mínimo de métricas:

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

Esquema mínimo de predicciones:

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

Una comparación directa requiere:

- mismo target;
- mismo seed;
- mismo scaffold split;
- mismas definiciones de métricas.

Si la cobertura difiere, se mostrarán:

1. métricas nativas de cada modelo;
2. métricas sobre la intersección común de moléculas.

Las comparaciones principales entre seeds usarán además diferencias emparejadas.

## 9. Ablations

| Experimento | Química | 3D | Densidad | Curvatura |
| --- | :---: | :---: | :---: | :---: |
| G7-A | ✓ | ✓ | — | — |
| G7-B | ✓ | ✓ | ✓ | — |
| G7-C | ✓ | ✓ | ✓ | ✓ |

La comparación clave será G7-B vs G7-C.

## 10. Corrección y reproducibilidad

Se comprobará:

- correspondencia nodo ↔ átomo RDKit;
- coherencia de shapes y dtypes;
- ausencia de NaN/Inf;
- invariancia a traslación;
- invariancia a rotación;
- invariancia a permutación coherente del grafo;
- bidireccionalidad y ausencia de duplicados;
- batching PyG;
- reproducibilidad por seed;
- ausencia de leakage.

Train se utiliza para ajustar pesos y normalización.

Validation se utiliza para early stopping y comparación de configuraciones.

Test, cuando se incorpore a la evaluación final, no se usará para seleccionar hiperparámetros.

## 11. Cache y persistencia

Los grafos crudos podrán cachearse por configuración.

El cache debe identificar al menos:

- esquema de features;
- política de H;
- conformero;
- cutoff;
- configuración RBF de aristas;
- configuración de densidad/DG, incluidos kernel, orden y escalas cuando estén activados.

Los grafos crudos no incluirán normalización dependiente del split.

Artefactos mínimos de una ejecución:

- model.pt
- config.json
- normalizer.*
- history.csv
- metadata del split/seed

model.pt contiene:
- model_state_dict
- best_epoch
- best_validation_mae

history.csv contiene:
- epoch
- train_loss
- train_mae
- validation_loss
- validation_mae

## 12. Criterios de aceptación

G7 se considera funcionalmente implementado cuando:

1. una molécula 3D válida se convierte de forma determinista en `Data`;
2. las features tienen esquema fijo y coherente;
3. los fallos de molécula/conformero se gestionan explícitamente;
4. los tres modos de arista funcionan;
5. densidad y curvatura se pueden activar/desactivar;
6. la normalización usa solo train;
7. el modelo realiza message passing usando `edge_attr`;
8. `encode()` devuelve un embedding molecular;
9. `08_gnn_training` guarda el mejor checkpoint por validation MAE;
10. `09_gnn_evaluation` calcula métricas de train/validation;
11. G7-A/B/C se comparan sobre los mismos splits;
12. la comparación con ML clásico controla seed, split y cobertura;
13. pasan las pruebas geométricas y de integración.

No se exige un umbral concreto de MAE para considerar correcto el software.

## 13. Fuera de alcance

- Transformer de fusión con descriptores globales;
- multi-conformer;
- GNNs equivariantes E(3)/SE(3);
- pretraining;
- multi-task ADMET;
- modelos generativos;
- optimización extensa de hiperparámetros.
