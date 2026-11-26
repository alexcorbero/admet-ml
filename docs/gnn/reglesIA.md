---
nombre: reglesIA.md
version: "1.1"
descripcion: Reglas de programación agéntica para G7
---

# Reglas de programación agéntica — G7

## 1. Principio

G7 se implementa desde cero mediante desarrollo guiado por especificaciones.

Antes de cada tarea el agente debe consultar:

1. `spec.md`;
2. `architecture.md`;
3. `tasks.md`;
4. `reglesIA.md`.

Jerarquía:

- `spec.md`: comportamiento;
- `architecture.md`: interfaces y dependencias;
- `tasks.md`: alcance de la tarea;
- `reglesIA.md`: proceso de trabajo.

Si los documentos se contradicen, el agente debe señalarlo antes de implementar.

## 2. Ciclo por tarea

Para cada `Tn.m`:

1. leer el contrato;
2. inspeccionar solo dependencias relevantes;
3. declarar alcance;
4. identificar entradas, salidas y casos límite;
5. implementar únicamente esa tarea;
6. ejecutar tests/smoke checks;
7. comparar con el contrato;
8. resumir archivos modificados y decisiones;
9. cerrar la tarea antes de iniciar otra.

No ampliar el alcance para refactorizar código no relacionado.

## 3. Límites

Los agentes no deben:

- modificar G0–G6 sin necesidad documentada;
- cambiar scaffold splits;
- cambiar política de H sin actualizar documentación y cache;
- introducir descriptores globales como node features;
- implementar el Transformer futuro;
- introducir multi-conformer, equivariance o pretraining sin nueva especificación;
- añadir dependencias sin justificarlo;
- eliminar moléculas problemáticas silenciosamente;
- ocultar NaN/Inf;
- sustituir curvaturas inválidas por cero sin su máscara;
- usar test para seleccionar hiperparámetros.

## 4. Datos y leakage

Solo train puede utilizarse para:

- ajustar normalización;
- optimizar pesos;
- estimar estadísticas aprendidas del preprocessing.

Validation puede utilizarse para:

- early stopping;
- selección de checkpoint;
- comparación de configuraciones.

Test queda reservado para evaluación final futura.

El target no participa en la construcción de `x`, `edge_attr`, `pos` ni del normalizer.

Los grafos cacheados deben ser crudos y reutilizables independientemente del split.

## 5. Corrección química y geométrica

Verificar:

- índice RDKit = índice de nodo;
- alineación entre `z`, `x_chem`, `x_dg` y `pos`;
- política de H uniforme;
- identidad atómica real conservada en `z`;
- `Other` solo para canales de densidad;
- `is_bonded` diferencia enlace de proximidad espacial;
- distancias finitas;
- aristas coherentes;
- curvaturas inválidas con máscara;
- ningún NaN/Inf llega al modelo.

### Preparación y persistencia de conformers

El notebook `prepare_aqsoldb_conformers.ipynb` es el único responsable de generar y persistir los conformers iniciales de AqSolDB.

El agente debe:

- utilizar las particiones oficiales de AqSolDB sin modificar sus filas;
- conservar `mol_id`, `smiles`, `y` y `split`;
- generar conformers de forma reproducible;
- utilizar una semilla explícita y registrar la semilla efectiva utilizada;
- reutilizar `src/features/conformers.py`;
- mantener una única política de hidrógenos;
- conservar los hidrógenos explícitos;
- generar varios conformers y seleccionar uno por molécula;
- seleccionar el conformer optimizado de menor energía válida y finita;
- conservar el `conf_id` real devuelto por RDKit;
- validar coordenadas finitas y geometrías no degeneradas;
- registrar todas las moléculas excluidas y sus motivos.

Restricciones:

- no eliminar hidrógenos después de generar los conformers;
- no añadir, eliminar ni reordenar átomos después de fijar la geometría;
- no seleccionar conformers utilizando el target experimental;
- no cambiar los splits oficiales;
- no seleccionar un conformer utilizando únicamente su posición en una lista;
- no mezclar energías calculadas con diferentes campos de fuerza;
- no excluir moléculas silenciosamente;
- no sustituir geometrías inválidas por geometrías artificiales.

Si una molécula no puede generar un conformer válido, debe quedar registrada como excluida.

Si no se dispone de parámetros para el campo de fuerza seleccionado, debe registrarse el fallo. No se debe cambiar silenciosamente a otro campo de fuerza.

El artefacto persistido debe contener una única representación seleccionada por molécula.

Todas las operaciones posteriores de G7 deben utilizar el mismo `conf_id` almacenado.

## 6. Modelo

La GNN debe realizar message passing real.

No es válido:

- MLP sobre vector molecular agregado;
- ignorar `edge_index`;
- recibir `edge_attr` pero no usarlo;
- hacer pooling antes del message passing.
- concatenar `data.pos` con `data.x` en la arquitectura GINE;

El encoder debe:

- usar embedding de `z`;
- incorporar `x`;
- propagar por `edge_index`;
- usar `edge_attr`;
- obtener embeddings atómicos;
- aplicar pooling;
- exponer `encode()`.
- la geometría 3D debe entrar mediante atributos invariantes derivados.

Cualquier cambio de GINE requiere actualizar antes la arquitectura.

## 7. Notebook de preparación de conformers

`prepare_aqsoldb_conformers.ipynb`:

- carga las particiones oficiales de AqSolDB;
- genera conformers 3D;
- optimiza las geometrías;
- selecciona el conformer válido de menor energía;
- valida la geometría;
- registra moléculas excluidas;
- guarda el artefacto de conformers y su manifest.

No:

- construye grafos;
- calcula features GNN;
- normaliza datos;
- entrena modelos;
- modifica los targets;
- modifica los splits.

La generación debe ser reproducible a partir de los parámetros y semillas registrados.

El artefacto se guarda en:

`data/conformers/aqsoldb/conformers.pkl`

Los artefactos generados dentro de `data/` no deben incorporarse al repositorio Git.

## 8. Notebook 07 — grafos

`07_gnn_graph_generation.ipynb`:

- carga los datos oficiales de AqSolDB;
- carga el artefacto persistido de conformers;
- valida la correspondencia entre moléculas y conformers;
- construye grafos;
- valida shapes y esquemas;
- registra fallos;
- guarda grafos crudos.

No:

- genera conformers;
- modifica geometrías;
- cambia la política de hidrógenos;
- normaliza;
- entrena;
- calcula comparaciones finales.

Si el artefacto de conformers no existe o es incompatible, debe detenerse explícitamente.

## 9. Notebook 08 — entrenamiento

`08_gnn_training.ipynb` debe:

- fijar seeds;
- ajustar normalizer solo con train;
- crear DataLoaders;
- usar correctamente `train()`/`eval()`;
- desactivar gradientes en validation;
- usar AdamW;
- aplicar clipping si está configurado;
- hacer early stopping por validation MAE;
- restaurar los mejores pesos;
- guardar config, normalizer, checkpoint e historial;
- detenerse de forma explícita ante NaN/Inf en loss o gradientes.

Huber puede ser la loss inicial; MAE sigue siendo la métrica primaria.

## 10. Notebook 09 — evaluación

`09_gnn_evaluation.ipynb`:

- no entrena;
- no reajusta normalizers;
- recalcula predicciones con best checkpoints;
- evalúa train y validation;
- calcula MAE, RMSE y R²;
- conserva IDs/SMILES;
- compara GNN con ML clásico.

Toda tabla comparativa debe incluir como mínimo:

- modelo;
- representación;
- seed;
- split;
- `n_samples`;
- cobertura;
- MAE;
- RMSE;
- R².

## 11. Comparación GNN vs ML clásico

Para comparar con XGBoost:

- usar mismos seeds y scaffold splits;
- verificar `row_id`;
- usar las mismas métricas;
- informar cobertura;
- si cambia la cobertura, comparar además la intersección común;
- usar diferencias emparejadas por seed en comparaciones principales;
- usar train para diagnosticar ajuste y validation para generalización.

No se debe concluir superioridad a partir de splits distintos o poblaciones distintas.

## 12. Pruebas mínimas

### Configuración
- defaults válidos;
- errores en valores inválidos;
- serialización.

### Node features
- shape;
- orden;
- finitud;
- aromaticidad;
- quiralidad;
- H;
- elementos raros.

### Edges
- covalent/radius/hybrid;
- bidireccionalidad;
- no duplicados;
- cutoff;
- atributos;
- RBF/kernel.

### DG
- canales ausentes;
- `Other`;
- máscaras;
- singularidades;
- no NaN/Inf.

### Grafo
- shapes;
- dtypes;
- metadata;
- batching.

### Normalizer
- fit solo train;
- máscaras;
- columnas no escaladas;
- guardado/carga.

### Modelo
- forward;
- backward;
- batch variable;
- uso de `edge_attr`.

### Integración
- traslación;
- rotación;
- permutación;
- checkpoint roundtrip;
- notebooks en kernel limpio.

## 13. Git

- una tarea por cambio pequeño y revisable;
- commits con ID de tarea;
- no hacer merge a `main` sin aprobación;
- no mezclar tareas sin necesidad;
- mantener 07 = grafos, 08 = training, 09 = evaluación.

## 14. Cambios de diseño

Si aparece una decisión no contemplada:

1. actualizar `spec.md` o `architecture.md`;
2. actualizar `tasks.md`;
3. después modificar código.

La implementación no debe convertirse en la fuente de requisitos.

## 15. Tarea terminada

Una tarea solo se considera terminada si:

- cumple el contrato;
- pasan las verificaciones;
- no rompe tareas previas;
- no introduce leakage;
- el agente puede explicar el cambio;
- documentación y código siguen alineados.

Una mejora de métrica no compensa una implementación metodológicamente incorrecta.
