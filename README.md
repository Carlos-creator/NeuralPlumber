# NeuralPlumber

> Generación Procedural de Niveles de Super Mario Bros mediante GANs y Optimización Evolutiva
> INF398 — Introducción al Aprendizaje Automático

**Equipo:** Carlos Ramírez · Nicolás Rogel · Carlos Saavedra

---

## Descripción

NeuralPlumber extiende el trabajo de Volz et al. (2018) para mejorar la **coherencia estructural** de los niveles de Mario generados por una DCGAN + CMA-ES. Las dos contribuciones principales son:

1. **Dataset expandido**: de 1 a 15 niveles de entrenamiento (VGLC)
2. **Fitness estructural**: nuevas métricas de coherencia integradas al CMA-ES

---

## Requisitos

### Python

```bash
conda create -n neuralplumber python=3.8
conda activate neuralplumber
pip install -r requirements.txt
```

### Java

```
Java 8+
Apache Ant (para compilar marioaiDagstuhl)
```

### Estructura de dependencias externas

Este proyecto depende de dos recursos que deben estar en la carpeta padre (`../`):

```
ML/Proyecto/
├── NeuralPlumber/          ← este repositorio
├── clone/DagstuhlGAN/      ← repositorio base (Volz et al.)
└── Datasets/TheVGLC/       ← Video Game Level Corpus
```

---

## Instalación

```bash
# 1. Clonar / ubicarse en el proyecto
cd NeuralPlumber/

# 2. Instalar dependencias Python
pip install -r requirements.txt

# 3. Compilar el framework Java de Mario (para evaluación con agente)
cd ../clone/DagstuhlGAN/marioaiDagstuhl/
ant build
cd ../../../NeuralPlumber/

# 4. Configurar ruta de Python para el bridge Java↔Python
echo "$(which python)" > ../clone/DagstuhlGAN/marioaiDagstuhl/src/basicMap/my_python_path.txt
```

---

## Uso rápido

### 1. Reproducir baseline (Volz et al.)

```bash
python src/baseline/reproduce_volz.py
# Output: métricas baseline en experiments/baseline/
```

### 2. Construir dataset expandido

```bash
python src/data/vglc_parser.py \
  --input_dir ../Datasets/TheVGLC/Super\ Mario\ Bros/Processed/ \
  --output data/dataset_full.json
# Output: data/dataset_full.json (~2500 ventanas de 15 niveles)
```

### 3. Re-entrenar el GAN con más datos

```bash
cd ../clone/DagstuhlGAN/pytorch/
python main.py \
  --nz 32 --ngf 64 --ndf 64 \
  --batchSize 32 --niter 5000 \
  --lrD 0.00005 --lrG 0.00005 \
  --json_path ../../../NeuralPlumber/data/dataset_full.json
# Output: netG_epoch_5000.pth (modelo re-entrenado)
```

### 4. Calcular métricas estructurales

```python
import numpy as np
from src.metrics.structural import structural_score

# level: array (14, 28) con IDs de tiles [0-9]
level = np.load("experiments/baseline/level_sample.npy")
metrics = structural_score(level)
print(metrics)
# {
#   'pipe_completeness':  0.83,
#   'ground_continuity':  0.91,
#   'gap_traversability': 0.75,
#   'enemy_placement':    1.00,
#   'structural_avg':     0.87
# }
```

### 5. Ejecutar CMA-ES con F3 (fitness combinado)

```bash
python src/evolution/cmaes_runner.py \
  --model models/netG_15levels.pth \
  --fitness f3 \
  --runs 40 \
  --evals 1000 \
  --output experiments/structural_fitness/
```

---

## Estructura del proyecto

```
NeuralPlumber/
├── README.md                       # Este archivo
├── proyecto_inf398.md              # Documentación completa del proyecto
├── requirements.txt                # Dependencias Python
│
├── src/
│   ├── data/
│   │   ├── vglc_parser.py          # Lee .txt del VGLC → JSON
│   │   └── dataset_builder.py      # Construye dataset_full.json
│   ├── metrics/
│   │   ├── structural.py           # pipe_completeness, ground_continuity, etc.
│   │   ├── playability.py          # Wrapper del agente A*
│   │   └── diversity.py            # Diversidad entre niveles generados
│   ├── fitness/
│   │   ├── f1_f2.py                # Fitness del paper base
│   │   └── f3_combined.py          # F3: jugabilidad + estructura + dificultad
│   ├── evolution/
│   │   └── cmaes_runner.py         # CMA-ES con fitness configurable
│   ├── visualization/
│   │   ├── level_renderer.py       # Renderiza nivel como imagen
│   │   └── metrics_plotter.py      # Gráficas comparativas
│   └── baseline/
│       └── reproduce_volz.py       # Reproduce experimentos Volz et al.
│
├── models/
│   └── netG_15levels.pth           # GAN re-entrenado (generado)
│
├── data/
│   ├── example.json                # Dataset original (173 ventanas, 1 nivel)
│   └── dataset_full.json           # Dataset expandido (~2500 ventanas, 15 niveles)
│
├── experiments/
│   ├── baseline/                   # Resultados Experimento A
│   ├── expanded_data/              # Resultados Experimento B
│   └── structural_fitness/         # Resultados Experimento C
│
└── notebooks/
    ├── 01_baseline_analysis.ipynb
    ├── 02_dataset_expansion.ipynb
    ├── 03_structural_metrics.ipynb
    └── 04_results_comparison.ipynb
```

---

## Experimentos

Se comparan 3 configuraciones:

| Experimento | Modelo | Fitness | Objetivo |
|---|---|---|---|
| **A — Baseline** | `netG_epoch_5000.pth` (1 nivel) | F1/F2 original | Reproducir Volz et al. |
| **B — Datos** | `netG_15levels.pth` (15 niveles) | F1/F2 original | Aislar efecto del dataset |
| **C — Full** | `netG_15levels.pth` (15 niveles) | F3 combinado | Propuesta completa |

---

## Métricas

| Métrica | Módulo | Descripción |
|---|---|---|
| Tasa de jugabilidad | `playability.py` | % de niveles completados por A* |
| Completitud de tuberías | `structural.py` | % de tuberías estructuralmente válidas |
| Continuidad de suelo | `structural.py` | % de tiles sólidos en fila inferior |
| Traversabilidad | `structural.py` | % de huecos ≤ 4 tiles de ancho |
| Placement de enemigos | `structural.py` | % de enemigos sobre superficie sólida |
| Score estructural | `structural.py` | Promedio de las 4 anteriores |
| Diversidad | `diversity.py` | Distancia L1 promedio entre niveles |

---

## Mapeo de tiles

| ID | Símbolo | Tipo |
|---|---|---|
| 0 | X | Suelo/Sólido |
| 1 | S | Bloque rompible |
| 2 | - | Vacío (pasable) |
| 3 | ? | Bloque pregunta lleno |
| 4 | Q | Bloque pregunta vacío |
| 5 | E | Enemigo (Goomba) |
| 6 | < | Tubería arriba-izquierda |
| 7 | > | Tubería arriba-derecha |
| 8 | [ | Tubería izquierda |
| 9 | ] | Tubería derecha |

---

## Referencias

- Volz et al. "Evolving Mario Levels in the Latent Space of a DCGAN." GECCO 2018.
- Shu et al. "Experience-Driven PCG via Reinforcement Learning." CoG 2021.
- Summerville et al. "The VGLC: The Video Game Level Corpus." PCG Workshop 2016.
- Vapnik. "An Overview of Statistical Learning Theory." IEEE TNN 1999.
