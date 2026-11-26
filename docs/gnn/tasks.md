# Tareas G7 — Implementación agéntica

> `[ ]` pendiente · `[x]` implementada y verificada.

## Fase 0. Preparación

- [x] **T0.1** Aprobar `spec.md`, `architecture.md`, `tasks.md` y `reglesIA.md`.
- [x] **T0.2** Crear la rama de implementación G7.
- [x] **T0.3** Fijar y documentar la política de hidrógenos.
- [x] **T0.4** Verificar PyTorch, PyTorch Geometric, RDKit y dispositivo disponible.

## Fase 1. Configuración (`src/gnn/config.py`)

- [x] **T1.1** Implementar `GraphConfig`.
- [x] **T1.2** Implementar `AtomicFeaturesConfig`.
- [x] **T1.3** Implementar `ModelConfig`.
- [x] **T1.4** Implementar `TrainingConfig`.
- [x] **T1.5** Implementar `GNNConfig` y serialización.
- [x] **T1.6** Añadir validaciones y tests.

## Fase 2. Features de nodo (`src/gnn/atom_features.py`)

- [x] **T2.1** Definir esquema y nombres de features.
- [x] **T2.2** Implementar hibridación one-hot.
- [x] **T2.3** Implementar quiralidad/CIP.
- [x] **T2.4** Implementar carga, grado, H, masa, aromaticidad y anillo.
- [x] **T2.5** Implementar `build_atom_features()`.
- [x] **T2.6** Probar H explícitos, aromaticidad, quiralidad y elementos raros.

## Fase 3. Aristas (`src/gnn/edge_features.py`)

- [x] **T3.1** Implementar enlaces RDKit.
- [x] **T3.2** Implementar vecinos por cutoff.
- [x] **T3.3** Implementar modo `hybrid`.
- [x] **T3.4** Implementar aristas bidireccionales sin duplicados/self-loops.
- [x] **T3.5** Implementar atributos químicos de enlace.
- [x] **T3.6** Implementar distancia y Gaussian RBF.
- [x] **T3.7** Verificar que los kernels exponential/Lorentz no forman parte de `edge_attr`; pertenecen a densidad/DG (Fase 4).
- [x] **T3.8** Implementar naming de `edge_attr`.
- [x] **T3.9** Añadir tests de los tres modos.

## Fase 4. Densidad y curvatura (`src/gnn/density_features.py`)

- [x] **T4.1** Definir canales y nombres.
- [x] **T4.2** Reutilizar derivadas analíticas de G6.
- [x] **T4.3** Implementar densidad por átomo.
- [x] **T4.4** Implementar densidad por carga opcional.
- [x] **T4.5** Implementar `H`, `K`, `kappa_min`, `kappa_max`.
- [x] **T4.6** Implementar máscaras de validez.
- [x] **T4.7** Implementar valor neutro para curvatura inválida.
- [x] **T4.8** Implementar transformación numérica configurable.
- [x] **T4.9** Verificar shapes y ausencia de NaN/Inf.
- [x] **T4.10** Añadir tests de `Other`, canales ausentes y singularidades.

## Fase 5. Grafo (`src/gnn/graph_builder.py`)

- [x] **T5.1** Implementar nombres de features de grafo.
- [x] **T5.2** Implementar `build_graph()`.
- [x] **T5.3** Añadir target y metadata.
- [x] **T5.4** Validar shapes, dtypes y finitud.
- [x] **T5.5** Verificar batching PyG.
- [x] **T5.6** Añadir smoke test RDKit → `Data` → batch.

## Fase 6. Dataset y cache (`src/gnn/dataset.py`)

- [x] **T6.1** Definir ID reproducible de configuración.
- [x] **T6.2** Implementar generación de grafos del dataset.
- [x] **T6.3** Registrar éxitos/fallos.
- [x] **T6.4** Implementar persistencia/carga.
- [x] **T6.5** Detectar caches incompatibles.
- [x] **T6.6** Implementar subsets por IDs de split.
- [x] **T6.7** Verificar que el cache contiene grafos crudos.

## Fase 7. Normalización (`src/gnn/normalizer.py`)

- [x] **T7.1** Definir columnas escalables.
- [x] **T7.2** Implementar `fit(train_graphs)`.
- [x] **T7.3** Respetar máscaras de curvatura.
- [x] **T7.4** Implementar transformaciones.
- [x] **T7.5** Implementar guardado/carga de estadísticas.
- [x] **T7.6** Probar ausencia de leakage.
- [x] **T7.7** Verificar valores inválidos neutros.

## Fase 8. Modelo (`src/models/gnn.py`)

- [x] **T8.1** Implementar embedding atómico.
- [x] **T8.2** Implementar encoder de nodos.
- [x] **T8.3** Implementar message passing GINE.
- [x] **T8.4** Apilar capas con norm/activación/dropout.
- [x] **T8.5** Implementar residuals.
- [x] **T8.6** Implementar pooling.
- [x] **T8.7** Implementar readout.
- [x] **T8.8** Implementar `encode()`.
- [x] **T8.9** Probar forward/backward.
- [x] **T8.10** Verificar shapes y parámetros.

## Fase 9. `07_gnn_graph_generation.ipynb`

- [ ] **T9.0** Generar y persistir conformers 3D de AqSolDB.
- [ ] **T9.1** Cargar datos/conformeros y configuración.
- [ ] **T9.2** Mostrar un grafo de ejemplo.
- [ ] **T9.3** Mostrar esquema de features.
- [ ] **T9.4** Generar grafos crudos.
- [ ] **T9.5** Registrar fallos.
- [ ] **T9.6** Guardar grafos, metadata y config ID.
- [ ] **T9.7** Verificar cache incompatible.
- [ ] **T9.8** Ejecutar `Restart & Run All` en smoke.

## Fase 10. `08_gnn_training.ipynb`

- [ ] **T10.1** Cargar grafos y scaffold splits.
- [ ] **T10.2** Ajustar normalizer con train.
- [ ] **T10.3** Crear DataLoaders.
- [ ] **T10.4** Fijar seeds/dispositivo.
- [ ] **T10.5** Configurar Huber/MAE/MSE.
- [ ] **T10.6** Implementar training loop con AdamW.
- [ ] **T10.7** Implementar validation sin gradientes.
- [ ] **T10.8** Aplicar gradient clipping.
- [ ] **T10.9** Implementar early stopping por validation MAE.
- [ ] **T10.10** Guardar config, normalizer, best checkpoint e historial.
- [ ] **T10.11** Restaurar mejores pesos.
- [ ] **T10.12** Ejecutar smoke training.
- [ ] **T10.13** Ejecutar `Restart & Run All` en smoke.

## Fase 11. `09_gnn_evaluation.ipynb`

- [ ] **T11.1** Cargar best checkpoint y normalizer por seed.
- [ ] **T11.2** Predecir train y validation.
- [ ] **T11.3** Calcular MAE, RMSE y R².
- [ ] **T11.4** Registrar `n_samples` y cobertura.
- [ ] **T11.5** Guardar predicciones y métricas GNN.
- [ ] **T11.6** Agregar resultados entre seeds.
- [ ] **T11.7** Cargar resultados XGBoost.
- [ ] **T11.8** Adaptarlos al esquema común.
- [ ] **T11.9** Comparar solo seeds/splits compatibles.
- [ ] **T11.10** Comparar también sobre `row_id` comunes si cambia la cobertura.
- [ ] **T11.11** Calcular diferencias emparejadas por seed.
- [ ] **T11.12** Analizar generalization gap y residuals.
- [ ] **T11.13** Guardar `data/results/comparison/`.
- [ ] **T11.14** Ejecutar `Restart & Run All`.

## Fase 12. Invariancias e integración

- [ ] **T12.1** Invariancia a traslación.
- [ ] **T12.2** Invariancia a rotación.
- [ ] **T12.3** Invariancia a permutación coherente.
- [ ] **T12.4** Verificar aristas bidireccionales/sin duplicados.
- [ ] **T12.5** Verificar roundtrip checkpoint + normalizer.
- [ ] **T12.6** Ejecutar toda la suite GNN.

## Cierre de tarea

Una tarea pasa a `[x]` solo si cumple el contrato, pasan sus verificaciones, no rompe tareas previas y el cambio queda resumido.
