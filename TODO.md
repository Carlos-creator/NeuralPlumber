# TODO — NeuralPlumber

> Última actualización: 2026-04-28
> Estado general: Pipeline funcionando, baseline pendiente de medición estadística.

---

## Progreso global

```
[x] Bloque 0 — Setup y compatibilidad
[ ] Bloque 1 — Experimento A (Baseline)
[ ] Bloque 2 — Experimento B (Dataset expandido)
[ ] Bloque 3 — Experimento C (Fitness estructural)
[ ] Bloque 4 — Evaluación comparativa y entrega
```

---

## Bloque 0 — Setup ✅ Completo

- [x] Crear estructura de carpetas `NeuralPlumber/`
- [x] Escribir `src/metrics/structural.py` (4 métricas)
- [x] Escribir `src/metrics/diversity.py`
- [x] Escribir `src/fitness/f3_combined.py`
- [x] Escribir `src/data/vglc_parser.py`
- [x] Escribir `src/baseline/reproduce_volz.py`
- [x] Escribir `src/visualization/level_renderer.py`
- [x] Escribir `src/visualization/metrics_plotter.py`
- [x] Fix compatibilidad PyTorch 2.5 en `clone/DagstuhlGAN/pytorch/models/dcgan.py`
  - [x] Puntos → guiones bajos en nombres de módulos
  - [x] `load_compatible()` para remapear claves del `.pth` antiguo
  - [x] `input.is_cuda` en lugar de `isinstance(...)`
  - [x] `nn.Softmax(dim=1)` y `weights_only=False`
- [x] Verificar pipeline completo (modelo carga, genera nivel, métricas calculan, visualización OK)
- [x] Primer sample medido: `structural_avg = 0.506`
- [x] Inicializar repositorio git
- [x] `requirements.txt`, `.gitignore`, `.gitattributes`

---

## Bloque 1 — Experimento A: Baseline

> **Responsable:** ---
> **Objetivo:** Medir métricas estructurales del modelo original de Volz et al. sobre 100 niveles.

- [ ] Ejecutar baseline estadístico:
  ```bash
  cd ..   # pararse en ML/Proyecto/
  python NeuralPlumber/src/baseline/reproduce_volz.py --n_samples 100
  ```
- [ ] Verificar que se genera `experiments/baseline/metrics_baseline.json`
- [ ] Anotar los valores en la tabla de resultados (ver abajo)
- [ ] Guardar 10 niveles de ejemplo en `experiments/baseline/sample_levels.json`
- [ ] Crear notebook `notebooks/01_baseline_analysis.ipynb` con visualización de los 10 niveles

---

## Bloque 2 — Experimento B: Dataset expandido

> **Responsable:** ---
> **Objetivo:** Re-entrenar el GAN con los 15 niveles del VGLC y medir si mejora la coherencia.

- [ ] Construir dataset expandido:
  ```bash
  python NeuralPlumber/src/data/vglc_parser.py \
    --input_dir "Datasets/TheVGLC/Super Mario Bros/Processed/" \
    --output NeuralPlumber/data/dataset_full.json
  ```
- [ ] Verificar cantidad de ventanas generadas (esperado: ~2400-2600)
- [ ] Comparar tamaño con dataset original:
  ```bash
  python -c "
  import json
  orig = json.load(open('clone/DagstuhlGAN/pytorch/example.json'))
  full = json.load(open('NeuralPlumber/data/dataset_full.json'))
  print(f'Original:  {len(orig)} ventanas')
  print(f'Expandido: {len(full)} ventanas')
  "
  ```
- [ ] Re-entrenar el GAN:
  ```bash
  cd clone/DagstuhlGAN/pytorch/
  cp example.json example_original.json
  cp ../../../NeuralPlumber/data/dataset_full.json example.json
  python main.py --nz 32 --ngf 64 --ndf 64 --batchSize 32 --niter 5000
  ```
- [ ] Guardar modelo como `NeuralPlumber/models/netG_15levels.pth`
- [ ] Ejecutar métricas sobre 100 niveles del modelo re-entrenado:
  ```bash
  python NeuralPlumber/src/baseline/reproduce_volz.py \
    --model_path NeuralPlumber/models/netG_15levels.pth \
    --n_samples 100 \
    --output_dir NeuralPlumber/experiments/expanded_data/
  ```
- [ ] Anotar resultados en tabla comparativa
- [ ] Crear notebook `notebooks/02_dataset_expansion.ipynb`

> **Nota:** El re-entrenamiento puede tardar 30-60 min. Dejarlo correr y avanzar con Bloque 3 en paralelo.

---

## Bloque 3 — Experimento C: Fitness estructural (F3)

> **Responsable:** ---
> **Objetivo:** Integrar F3 en el CMA-ES y verificar que genera niveles más coherentes.

- [ ] Implementar `src/evolution/cmaes_runner.py`:
  - [ ] Wrapper Python del CMA-ES que use `load_compatible()` para el generador
  - [ ] Soporte para `--fitness f1`, `--fitness f2`, `--fitness f3`
  - [ ] Guardar mejores vectores latentes y niveles generados
  - [ ] Log de fitness por iteración (formato compatible con `cmaes_java_experiment/timeline*.txt`)
- [ ] Verificar que `f3_static` funciona sin necesidad del agente Java:
  ```bash
  python -c "
  import sys; sys.path.insert(0, 'NeuralPlumber/src')
  import numpy as np
  from fitness.f3_combined import f3_static
  level = np.zeros((14, 28), dtype=int)
  level[-1, :] = 0   # suelo completo
  print('F3 static (nivel perfecto):', f3_static(level))
  "
  ```
- [ ] Ejecutar CMA-ES con F3 estático (sin Java) — primero para validar:
  ```bash
  python NeuralPlumber/src/evolution/cmaes_runner.py \
    --fitness f3_static --runs 20 --evals 500
  ```
- [ ] (Opcional/avanzado) Conectar con agente Java para F3 completo
- [ ] Guardar resultados en `experiments/structural_fitness/`
- [ ] Crear notebook `notebooks/03_structural_metrics.ipynb`

---

## Bloque 4 — Evaluación comparativa y entrega

> **Responsable:** Todos
> **Objetivo:** Comparar A vs B vs C, generar figuras y escribir conclusiones.

- [ ] Ejecutar `src/visualization/metrics_plotter.py` con los 3 experimentos:
  ```bash
  python NeuralPlumber/src/visualization/metrics_plotter.py \
    --experiments_dir NeuralPlumber/experiments/ \
    --save_dir NeuralPlumber/experiments/
  ```
- [ ] Completar notebook `notebooks/04_results_comparison.ipynb` con:
  - [ ] Tabla comparativa A vs B vs C
  - [ ] Gráfica de barras de métricas estructurales
  - [ ] Grid visual de niveles representativos de cada experimento
  - [ ] Discusión de resultados
- [ ] Hacer commit y push final a GitHub
- [ ] Revisar que el README refleja los resultados finales

---

## Tabla de resultados (completar a medida que se obtienen)

| Métrica | Exp A (baseline) | Exp B (15 niveles) | Exp C (F3) |
|---|---|---|---|
| `pipe_completeness` | — | — | — |
| `ground_continuity` | — | — | — |
| `gap_traversability` | — | — | — |
| `enemy_placement` | — | — | — |
| `structural_avg` | — | — | — |
| Tasa jugabilidad | — | — | — |

> Sample único medido: `structural_avg = 0.506` (no estadístico, n=1)

---

## Fixes pendientes menores

- [ ] Agregar argumento `--model_path` a `reproduce_volz.py` para poder apuntar a distintos modelos (actualmente hardcodeado)
- [ ] Verificar que `vglc_parser.py` maneja niveles con longitudes distintas (algunos niveles del VGLC pueden tener más de 14 filas)
- [ ] Revisar `main.py` del DagstuhlGAN para compatibilidad con PyTorch 2.5 antes del re-entrenamiento (puede tener más usos de `Variable()`)

---

## Notas

- El modelo preentrenado `netG_epoch_5000.pth` está en `clone/DagstuhlGAN/pytorch/` — no se sube al repo
- Los `.pth` nuevos van en `NeuralPlumber/models/` — tampoco se suben (ver `.gitignore`)
- `data/dataset_full.json` tampoco se sube — se regenera con `vglc_parser.py`
- Para compartir modelos entre integrantes: Google Drive o similar
