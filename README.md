# NeuralPlumber

> Generación Procedural de Niveles de Super Mario Bros mediante GANs y Optimización Evolutiva
> INF398 — Introducción al Aprendizaje Automático

**Equipo:** Carlos Ramírez · Nicolás Rogel · Carlos Saavedra

---

## Estado actual

| Componente | Estado |
|---|---|
| Estructura del proyecto | Listo |
| Compatibilidad PyTorch 2.5 (`dcgan.py`) | Resuelto |
| Carga de modelo preentrenado | Funcionando |
| Generación de niveles | Funcionando |
| Métricas estructurales | Funcionando |
| Visualización de niveles | Funcionando |
| Baseline estadístico (100 niveles) — Exp A | Completo |
| Dataset expandido (2518 ventanas, 15 niveles) | Completo |
| Re-entrenamiento GAN con 15 niveles — Exp B | Completo |
| CMA-ES con F3 (40 runs, 1000 evals) — Exp C | Completo |
| Gráficas comparativas A vs B vs C | Completo |

---

## Resultados

### Tabla comparativa A vs B vs C

| Métrica | Exp A (baseline) | Exp B (15 niveles) | Exp C (CMA-ES + F3) |
|---|---|---|---|
| `pipe_completeness` | 0.2075 | 0.1707 | **0.4567** |
| `ground_continuity` | 0.5093 | 0.5211 | **0.8326** |
| `gap_traversability` | 0.9100 | 0.9175 | **1.0000** |
| `enemy_placement` | 0.9667 | 0.9600 | **1.0000** |
| **`structural_avg`** | 0.6484 | 0.6423 | **0.8223** |

**Observaciones clave:**
- Exp B (más datos solos) no mejora significativamente sobre el baseline — el dataset no es el cuello de botella
- Exp C (CMA-ES + F3) mejora todas las métricas: `pipe_completeness` +120%, `ground_continuity` +63%, `structural_avg` +27%
- El CMA-ES supera al baseline desde la primera generación — el espacio latente del GAN es suave y apto para optimización
- `gap_traversability` y `enemy_placement` alcanzan 1.0 — todos los niveles son completamente traversables

### Archivos de resultados generados

| Archivo | Descripción |
|---|---|
| `experiments/baseline/metrics_baseline.json` | Métricas Exp A (n=100) |
| `experiments/baseline/sample_levels.json` | 10 niveles de ejemplo Exp A |
| `experiments/baseline/sample_levels.png` | Visualización Exp A |
| `experiments/expanded_data/metrics_baseline.json` | Métricas Exp B (n=100) |
| `experiments/expanded_data/training_log.txt` | Log de entrenamiento GAN |
| `experiments/expanded_data/training_progress.png` | Curvas de pérdida del entrenamiento |
| `experiments/structural_fitness/cmaes_results_netG_15levels_f3_static.json` | Resultados Exp C (40 runs) |
| `experiments/structural_fitness/best_levels_netG_15levels_f3_static.json` | Mejores niveles Exp C |
| `experiments/structural_fitness/sample_levels_cmaes.png` | Visualización Exp C |
| `experiments/structural_fitness/cmaes_evolution.png` | Evolución del fitness CMA-ES |
| `experiments/comparison_metrics.png` | Gráfica comparativa A vs B vs C |

---

## Requisitos

### Python

```bash
conda create -n neuralplumber python=3.8
conda activate neuralplumber
pip install -r requirements.txt
```

**Versión probada:** Python 3.12 · PyTorch 2.5.1+cu121 · cma 4.4.4

### Estructura de dependencias externas

```
ML/Proyecto/
├── NeuralPlumber/               <- este repositorio
├── clone/DagstuhlGAN/           <- clonar desde https://github.com/CIGbalance/DagstuhlGAN
└── Datasets/TheVGLC/            <- clonar desde https://github.com/TheVGLC/TheVGLC
```

---

## Instalación

```bash
# 1. Instalar dependencias Python
pip install -r requirements.txt

# 2. Fix de compatibilidad PyTorch 2.x ya incluido en el repo
#    clone/DagstuhlGAN/pytorch/models/dcgan.py ya fue parcheado.
```

---

## Uso

### Verificar que todo funciona

```bash
cd ..   # pararse en ML/Proyecto/
python -c "
import sys
sys.path.insert(0, 'clone/DagstuhlGAN/pytorch')
sys.path.insert(0, 'NeuralPlumber/src')
import torch
from models.dcgan import DCGAN_G, load_compatible
from metrics.structural import structural_score
gen = DCGAN_G(32, 32, 10, 64, 0, 0)
load_compatible(gen, 'clone/DagstuhlGAN/pytorch/netG_epoch_5000.pth')
gen.eval()
with torch.no_grad():
    out = gen(torch.randn(1, 32, 1, 1))
level = out[0, :, :14, :28].argmax(0).numpy()
print(structural_score(level))
"
```

### Experimento A — Baseline (100 niveles, modelo original)

```bash
cd ..   # pararse en ML/Proyecto/
python NeuralPlumber/src/baseline/reproduce_volz.py --n_samples 100
# Output: NeuralPlumber/experiments/baseline/metrics_baseline.json
```

### Construir dataset expandido

```bash
python NeuralPlumber/src/data/vglc_parser.py \
  --input_dir "Datasets/TheVGLC/Super Mario Bros/Processed/" \
  --output NeuralPlumber/data/dataset_full.json
# Output: 2518 ventanas de 15 niveles de Mario
```

### Re-entrenar el GAN (Experimento B)

```bash
cd clone/DagstuhlGAN/pytorch/
cp example.json example_original.json
cp ../../../NeuralPlumber/data/dataset_full.json example.json
python main.py --nz 32 --ngf 64 --ndf 64 --batchSize 32 --niter 5000
cp netG_epoch_5000.pth ../../../NeuralPlumber/models/netG_15levels.pth
# Tiempo estimado: 45-60 min en CPU
```

### Medir métricas del modelo re-entrenado (Experimento B)

```bash
cd ..   # pararse en ML/Proyecto/
python NeuralPlumber/src/baseline/reproduce_volz.py \
  --n_samples 100 \
  --model_path NeuralPlumber/models/netG_15levels.pth \
  --output_dir NeuralPlumber/experiments/expanded_data/
```

### CMA-ES con F3 (Experimento C)

```bash
cd ..   # pararse en ML/Proyecto/
python NeuralPlumber/src/evolution/cmaes_runner.py \
  --model NeuralPlumber/models/netG_15levels.pth \
  --fitness f3_static --runs 40 --evals 1000 \
  --output_dir NeuralPlumber/experiments/structural_fitness/
# Tiempo estimado: 15-20 min
```

---

## Estructura del proyecto

```
NeuralPlumber/
├── README.md
├── TODO.md                        <- tareas pendientes
├── proyecto_inf398.md             <- documentacion completa del proyecto
├── requirements.txt
├── .gitignore
├── .gitattributes
|
├── src/
│   ├── data/
│   │   └── vglc_parser.py         <- lee .txt del VGLC -> dataset_full.json (2518 ventanas)
│   ├── metrics/
│   │   ├── structural.py          <- pipe_completeness, ground_continuity, etc.
│   │   └── diversity.py           <- diversidad entre niveles generados
│   ├── fitness/
│   │   └── f3_combined.py         <- F3 estatico + F3 completo + F1/F2 originales
│   ├── evolution/
│   │   └── cmaes_runner.py        <- CMA-ES sobre espacio latente del GAN
│   ├── baseline/
│   │   └── reproduce_volz.py      <- genera N niveles, reporta metricas (--model_path)
│   └── visualization/
│       ├── level_renderer.py      <- renderiza niveles con colores por tile
│       └── metrics_plotter.py     <- graficas comparativas A vs B vs C
│
├── models/                        <- .pth van aqui (ignorados por git)
│   └── netG_15levels.pth          <- modelo re-entrenado con 15 niveles VGLC
├── data/
│   └── dataset_full.json          <- generado con vglc_parser.py (ignorado por git)
└── experiments/
    ├── baseline/                  <- Exp A: metricas, 10 niveles de ejemplo, imagen
    ├── expanded_data/             <- Exp B: metricas, log de entrenamiento, curvas
    └── structural_fitness/        <- Exp C: resultados CMA-ES, mejores niveles, evolucion
```

---

## Fix de compatibilidad PyTorch 2.x

El archivo `clone/DagstuhlGAN/pytorch/models/dcgan.py` fue modificado para compatibilidad con PyTorch 2.x:

- **Puntos en nombres de módulos** → guiones bajos (`initial.32-256.convt` → `initial_32-256_convt`)
- **`load_compatible()`** → remapea automáticamente las claves del `.pth` guardado con la versión antigua
- **`input.is_cuda`** → reemplaza `isinstance(input.data, torch.cuda.FloatTensor)`
- **`nn.Softmax(dim=1)`** → fix del warning de dimensión implícita
- **`weights_only=False`** → silencia el FutureWarning de `torch.load`

Usar `load_compatible()` en lugar de `load_state_dict(torch.load(...))` directamente.

---

## Experimentos

| Experimento | Modelo | Fitness | Objetivo |
|---|---|---|---|
| **A — Baseline** | `netG_epoch_5000.pth` (1 nivel) | ninguno (muestreo random) | Reproducir Volz et al. |
| **B — Datos** | `netG_15levels.pth` (15 niveles) | ninguno (muestreo random) | Aislar efecto del dataset |
| **C — Full** | `netG_15levels.pth` (15 niveles) | F3 estatico (CMA-ES) | Propuesta completa |

---

## Métricas estructurales

| Métrica | Rango | Mejor | Descripción |
|---|---|---|---|
| `pipe_completeness` | [0,1] | Alto | % de tuberias estructuralmente validas |
| `ground_continuity` | [0,1] | Alto | % de tiles solidos en fila inferior |
| `gap_traversability` | [0,1] | Alto | % de huecos <= 4 tiles de ancho |
| `enemy_placement` | [0,1] | Alto | % de enemigos sobre superficie solida |
| `structural_avg` | [0,1] | Alto | Promedio de las 4 anteriores |

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

## Reproducir desde cero

### Estructura de carpetas requerida

```
ML/Proyecto/
├── NeuralPlumber/          <- este repositorio
├── clone/
│   └── DagstuhlGAN/        <- clonar aquí
└── Datasets/
    └── TheVGLC/            <- clonar aquí
```

Clonar dependencias externas:

```bash
cd "ML/Proyecto"
mkdir clone && cd clone
git clone https://github.com/CIGbalance/DagstuhlGAN
cd ..
mkdir -p Datasets && cd Datasets
git clone https://github.com/TheVGLC/TheVGLC
cd ..
```

---

### Entorno Python

```bash
# Con conda (recomendado)
conda create -n neuralplumber python=3.12
conda activate neuralplumber
pip install -r NeuralPlumber/requirements.txt

# O con pip directo
pip install -r NeuralPlumber/requirements.txt
```

Verificar instalación:

```bash
python -c "import torch, cma, numpy, matplotlib; print('OK')"
```

---

### Verificar pipeline completo

Desde `ML/Proyecto/`:

```bash
python -c "
import sys
sys.path.insert(0, 'clone/DagstuhlGAN/pytorch')
sys.path.insert(0, 'NeuralPlumber/src')
import torch
from models.dcgan import DCGAN_G, load_compatible
from metrics.structural import structural_score
gen = DCGAN_G(32, 32, 10, 64, 0, 0)
load_compatible(gen, 'clone/DagstuhlGAN/pytorch/netG_epoch_5000.pth')
gen.eval()
with torch.no_grad():
    out = gen(torch.randn(1, 32, 1, 1))
level = out[0, :, :14, :28].argmax(0).numpy()
print(structural_score(level))
"
```

---

### Experimento A — Baseline

```bash
# Desde ML/Proyecto/
python NeuralPlumber/src/baseline/reproduce_volz.py --n_samples 100
# Output: NeuralPlumber/experiments/baseline/metrics_baseline.json
```

---

### Experimento B — Dataset expandido + re-entrenamiento

```bash
# 1. Construir dataset
python NeuralPlumber/src/data/vglc_parser.py \
  --input_dir "Datasets/TheVGLC/Super Mario Bros/Processed/" \
  --output NeuralPlumber/data/dataset_full.json
# Resultado esperado: Total ventanas: 2518

# 2. Reemplazar dataset en DagstuhlGAN
cp clone/DagstuhlGAN/pytorch/example.json clone/DagstuhlGAN/pytorch/example_original.json
cp NeuralPlumber/data/dataset_full.json clone/DagstuhlGAN/pytorch/example.json

# 3. Re-entrenar (~50 min CPU, ~10 min GPU)
cd clone/DagstuhlGAN/pytorch
python main.py --nz 32 --ngf 64 --ndf 64 --batchSize 32 --niter 5000
cd ../../..

# 4. Copiar modelo
cp clone/DagstuhlGAN/pytorch/netG_epoch_5000.pth NeuralPlumber/models/netG_15levels.pth

# 5. Medir metricas del modelo re-entrenado
python NeuralPlumber/src/baseline/reproduce_volz.py \
  --n_samples 100 \
  --model_path NeuralPlumber/models/netG_15levels.pth \
  --output_dir NeuralPlumber/experiments/expanded_data/
```

---

### Experimento C — CMA-ES con F3

```bash
# Desde ML/Proyecto/ (~15-20 min)
python NeuralPlumber/src/evolution/cmaes_runner.py \
  --model NeuralPlumber/models/netG_15levels.pth \
  --fitness f3_static \
  --runs 40 \
  --evals 1000 \
  --output_dir NeuralPlumber/experiments/structural_fitness/

# Para prueba rapida:
python NeuralPlumber/src/evolution/cmaes_runner.py \
  --model NeuralPlumber/models/netG_15levels.pth \
  --fitness f3_static \
  --runs 3 \
  --evals 100
```

---

### Visualizar niveles

```bash
python -c "
import sys, json, numpy as np
sys.path.insert(0, 'NeuralPlumber/src')
from visualization.level_renderer import render_comparison

# Cambiar segun que quieras ver:
# Baseline: NeuralPlumber/experiments/baseline/sample_levels.json
# CMA-ES:   NeuralPlumber/experiments/structural_fitness/best_levels_netG_15levels_f3_static.json
levels = json.load(open('NeuralPlumber/experiments/baseline/sample_levels.json'))
render_comparison(
    [np.array(l) for l in levels[:6]],
    titles=[f'Nivel {i+1}' for i in range(6)],
    save_path='visualizacion.png'
)
print('Imagen guardada: visualizacion.png')
"
```

---

### Script automatico (Windows)

Para correr todos los experimentos de una vez desde `ML/Proyecto/`:

```bat
run_experiments.bat
```

El archivo `run_experiments.bat` incluido en el repositorio ejecuta los pasos A, B y C en orden. El re-entrenamiento del GAN (paso más largo) tarda ~50 min en CPU.

---

## Referencias

- Volz et al. "Evolving Mario Levels in the Latent Space of a DCGAN." GECCO 2018.
- Shu et al. "Experience-Driven PCG via Reinforcement Learning." CoG 2021.
- Summerville et al. "The VGLC: The Video Game Level Corpus." PCG Workshop 2016.
- Vapnik. "An Overview of Statistical Learning Theory." IEEE TNN 1999.
