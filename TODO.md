# TODO — NeuralPlumber

> Última actualización: 2026-05-17
> Estado general: Experimentos A, B y C completados. Pendiente notebooks y entrega final.

---

## Progreso global

```
[x] Bloque 0 — Setup y compatibilidad
[x] Bloque 1 — Experimento A (Baseline)
[x] Bloque 2 — Experimento B (Dataset expandido)
[x] Bloque 3 — Experimento C (Fitness estructural)
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
- [x] Verificar pipeline completo (`structural_avg = 0.506` en 1 sample)
- [x] Inicializar repositorio git
- [x] `requirements.txt`, `.gitignore`, `.gitattributes`
- [x] `run_experiments.bat` — script para reproducir todo el pipeline en Windows

---

## Bloque 1 — Experimento A: Baseline ✅ Completo

- [x] Agregar `--model_path` y `--output_dir` a `reproduce_volz.py`
- [x] Ejecutar baseline estadístico (n=100)
- [x] Verificar generación de `experiments/baseline/metrics_baseline.json`
- [x] Guardar 10 niveles de ejemplo en `experiments/baseline/sample_levels.json`
- [x] Visualizar 6 niveles → `experiments/baseline/sample_levels.png`

**Resultados:**

| Métrica | Valor |
|---|---|
| `pipe_completeness` | 0.2075 |
| `ground_continuity` | 0.5093 |
| `gap_traversability` | 0.9100 |
| `enemy_placement` | 0.9667 |
| `structural_avg` | **0.6484** |

---

## Bloque 2 — Experimento B: Dataset expandido ✅ Completo

- [x] Construir dataset expandido con `vglc_parser.py`
  - 15 niveles → 2518 ventanas (vs 173 originales, 14.6x más datos)
- [x] Comparar tamaño con dataset original
- [x] Re-entrenar GAN (5000 épocas, ~50 min CPU)
- [x] Guardar modelo como `NeuralPlumber/models/netG_15levels.pth`
- [x] Ejecutar métricas sobre 100 niveles del modelo re-entrenado
- [x] Guardar curvas de entrenamiento → `experiments/expanded_data/training_progress.png`

**Resultados:**

| Métrica | Valor |
|---|---|
| `pipe_completeness` | 0.1707 |
| `ground_continuity` | 0.5211 |
| `gap_traversability` | 0.9175 |
| `enemy_placement` | 0.9600 |
| `structural_avg` | **0.6423** |

> Observación: más datos solos no mejoran el baseline. El cuello de botella es el muestreo aleatorio, no el dataset.

---

## Bloque 3 — Experimento C: Fitness estructural (F3) ✅ Completo

- [x] Implementar `src/evolution/cmaes_runner.py`
  - [x] Soporte para `--fitness f3_static` y `--fitness structural_avg`
  - [x] Log de fitness por generación
  - [x] Guardar mejores niveles y resultados JSON
- [x] Corregir bug de doble negación en `make_fitness_fn`
- [x] Ejecutar CMA-ES con F3 estático (40 runs, 1000 evals)
- [x] Guardar resultados en `experiments/structural_fitness/`
- [x] Graficar evolución del fitness → `experiments/structural_fitness/cmaes_evolution.png`
- [x] Visualizar mejores niveles → `experiments/structural_fitness/sample_levels_cmaes.png`
- [x] Gráfica comparativa A vs B vs C → `experiments/comparison_metrics.png`

**Resultados:**

| Métrica | Valor |
|---|---|
| `pipe_completeness` | 0.4567 (+120% vs A) |
| `ground_continuity` | 0.8326 (+63% vs A) |
| `gap_traversability` | 1.0000 |
| `enemy_placement` | 1.0000 |
| `structural_avg` | **0.8223** (+27% vs A) |

---

## Bloque 4 — Evaluación comparativa y entrega

> **Responsable:** Todos
> **Objetivo:** Completar notebooks y escribir conclusiones para la entrega final.

- [ ] Completar notebook `notebooks/01_baseline_analysis.ipynb`
- [ ] Completar notebook `notebooks/02_dataset_expansion.ipynb`
- [ ] Completar notebook `notebooks/03_structural_metrics.ipynb`
- [ ] Completar notebook `notebooks/04_results_comparison.ipynb`
- [ ] Hacer commit y push final a GitHub

---

## Tabla de resultados completa

| Métrica | Exp A (baseline) | Exp B (15 niveles) | Exp C (CMA-ES+F3) |
|---|---|---|---|
| `pipe_completeness` | 0.2075 | 0.1707 | **0.4567** |
| `ground_continuity` | 0.5093 | 0.5211 | **0.8326** |
| `gap_traversability` | 0.9100 | 0.9175 | **1.0000** |
| `enemy_placement` | 0.9667 | 0.9600 | **1.0000** |
| `structural_avg` | 0.6484 | 0.6423 | **0.8223** |

---

## Notas

- El modelo preentrenado `netG_epoch_5000.pth` está en `clone/DagstuhlGAN/pytorch/` — no se sube al repo
- El modelo re-entrenado `netG_15levels.pth` está en `NeuralPlumber/models/` — tampoco se sube
- `data/dataset_full.json` tampoco se sube — se regenera con `vglc_parser.py`
- Los resultados de experimentos (`.json`, `.png`) tampoco se suben — se regeneran corriendo los scripts
- Para compartir modelos entre integrantes: Google Drive o similar
