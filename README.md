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
| Visualización | Funcionando |
| Baseline estadístico (100 niveles) | Pendiente |
| Dataset expandido (15 niveles) | Pendiente |
| Re-entrenamiento GAN | Pendiente |
| CMA-ES con F3 | Pendiente |

---

## Requisitos

### Python

```bash
conda create -n neuralplumber python=3.8
conda activate neuralplumber
pip install -r requirements.txt
```

**Versión probada:** Python 3.12 · PyTorch 2.5.1+cu121

### Java

```
Java 8+
Apache Ant (para compilar marioaiDagstuhl, solo necesario para Exp. C)
```

### Estructura de dependencias externas

Este proyecto requiere dos recursos en la carpeta padre (`../`):

```
ML/Proyecto/
├── NeuralPlumber/               ← este repositorio
├── clone/DagstuhlGAN/           ← clonar desde https://github.com/CIGbalance/DagstuhlGAN
└── Datasets/TheVGLC/            ← clonar desde https://github.com/TheVGLC/TheVGLC
```

---

## Instalación

```bash
# 1. Instalar dependencias Python
pip install -r requirements.txt

# 2. Aplicar fix de compatibilidad PyTorch 2.x (ya incluido en el repo)
#    El archivo clone/DagstuhlGAN/pytorch/models/dcgan.py ya fue parcheado.

# 3. (Solo para Exp. C) Compilar el framework Java de Mario
cd ../clone/DagstuhlGAN/marioaiDagstuhl/
ant build
echo "$(which python)" > src/basicMap/my_python_path.txt
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
cd ..
python NeuralPlumber/src/baseline/reproduce_volz.py --n_samples 100
# Output: NeuralPlumber/experiments/baseline/metrics_baseline.json
```

### Construir dataset expandido

```bash
python NeuralPlumber/src/data/vglc_parser.py \
  --input_dir "Datasets/TheVGLC/Super Mario Bros/Processed/" \
  --output NeuralPlumber/data/dataset_full.json
# Output: ~2500 ventanas de 15 niveles de Mario
```

### Re-entrenar el GAN (Experimento B)

```bash
cd clone/DagstuhlGAN/pytorch/
cp example.json example_original.json
cp ../../../NeuralPlumber/data/dataset_full.json example.json
python main.py --nz 32 --ngf 64 --ndf 64 --batchSize 32 --niter 5000
# Output: netG_epoch_5000.pth (modelo re-entrenado con 15 niveles)
```

### CMA-ES con F3 (Experimento C)

```bash
cd ../../../NeuralPlumber/
python src/evolution/cmaes_runner.py \
  --model ../clone/DagstuhlGAN/pytorch/netG_epoch_5000.pth \
  --fitness f3 --runs 40 --evals 1000 \
  --output experiments/structural_fitness/
```

---

## Estructura del proyecto

```
NeuralPlumber/
├── README.md
├── TODO.md                        ← tareas pendientes
├── proyecto_inf398.md             ← documentación completa del proyecto
├── requirements.txt
├── .gitignore
├── .gitattributes
│
├── src/
│   ├── data/
│   │   └── vglc_parser.py         ← lee .txt del VGLC → dataset_full.json
│   ├── metrics/
│   │   ├── structural.py          ← pipe_completeness, ground_continuity, etc.
│   │   └── diversity.py           ← diversidad entre niveles generados
│   ├── fitness/
│   │   └── f3_combined.py         ← F3 + F1/F2 originales
│   ├── evolution/
│   │   └── cmaes_runner.py        ← CMA-ES con fitness configurable (pendiente)
│   ├── baseline/
│   │   └── reproduce_volz.py      ← genera 100 niveles, reporta métricas baseline
│   └── visualization/
│       ├── level_renderer.py      ← renderiza nivel con colores por tile
│       └── metrics_plotter.py     ← gráficas comparativas A vs B vs C
│
├── models/                        ← .pth van aquí (ignorados por git)
├── data/
│   └── dataset_full.json          ← generado con vglc_parser.py (ignorado por git)
├── experiments/
│   ├── baseline/
│   ├── expanded_data/
│   └── structural_fitness/
└── notebooks/
    ├── 01_baseline_analysis.ipynb
    ├── 02_dataset_expansion.ipynb
    ├── 03_structural_metrics.ipynb
    └── 04_results_comparison.ipynb
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
| **A — Baseline** | `netG_epoch_5000.pth` (1 nivel) | F1/F2 original | Reproducir Volz et al. |
| **B — Datos** | `netG_15levels.pth` (15 niveles) | F1/F2 original | Aislar efecto del dataset |
| **C — Full** | `netG_15levels.pth` (15 niveles) | F3 combinado | Propuesta completa |

---

## Métricas estructurales

| Métrica | Rango | Mejor | Descripción |
|---|---|---|---|
| `pipe_completeness` | [0,1] | Alto | % de tuberías estructuralmente válidas |
| `ground_continuity` | [0,1] | Alto | % de tiles sólidos en fila inferior |
| `gap_traversability` | [0,1] | Alto | % de huecos ≤ 4 tiles de ancho |
| `enemy_placement` | [0,1] | Alto | % de enemigos sobre superficie sólida |
| `structural_avg` | [0,1] | Alto | Promedio de las 4 anteriores |

**Resultado baseline medido (1 sample):** `structural_avg = 0.506`

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
