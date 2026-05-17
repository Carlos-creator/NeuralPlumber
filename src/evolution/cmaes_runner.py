"""
CMA-ES sobre el espacio latente del GAN para optimizar coherencia estructural.

Uso basico:
    python src/evolution/cmaes_runner.py --fitness f3_static --runs 20 --evals 500

Uso con modelo re-entrenado:
    python src/evolution/cmaes_runner.py \
        --model models/netG_15levels.pth \
        --fitness f3_static --runs 40 --evals 1000 \
        --output_dir experiments/structural_fitness/
"""
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path

DAGSTUHL_PYTORCH = Path(__file__).resolve().parents[3] / "clone" / "DagstuhlGAN" / "pytorch"
sys.path.insert(0, str(DAGSTUHL_PYTORCH))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.dcgan import DCGAN_G, load_compatible
from metrics.structural import structural_score
from fitness.f3_combined import f3_static

DEFAULT_MODEL  = DAGSTUHL_PYTORCH / "netG_epoch_5000.pth"
OUTPUT_DIR     = Path(__file__).resolve().parents[2] / "experiments" / "structural_fitness"
NZ             = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_generator(model_path: Path) -> DCGAN_G:
    gen = DCGAN_G(32, NZ, 10, 64, 0, 0)
    load_compatible(gen, str(model_path))
    gen.eval()
    return gen


def z_to_level(generator: DCGAN_G, z: np.ndarray) -> np.ndarray:
    """Convierte vector latente z -> nivel (14, 28)."""
    z_tensor = torch.FloatTensor(z).view(1, NZ, 1, 1)
    with torch.no_grad():
        out = generator(z_tensor)           # (1, 10, 32, 32)
    return out[0, :, :14, :28].argmax(dim=0).numpy()


def make_fitness_fn(generator: DCGAN_G, fitness_name: str):
    """
    Retorna funcion de fitness para CMA-ES (CMA-ES minimiza).
    - f3_static ya retorna negativo (mas negativo = mejor), se pasa directo.
    - structural_avg retorna [0,1] (mayor = mejor), se niega.
    En ambos casos gen_best_score = -fitness da el score positivo real.
    """
    def fitness(z: np.ndarray) -> float:
        level = z_to_level(generator, z)
        if fitness_name == "f3_static":
            return float(f3_static(level))          # ya es negativo
        else:
            return -float(structural_score(level)["structural_avg"])
    return fitness


# ---------------------------------------------------------------------------
# CMA-ES
# ---------------------------------------------------------------------------

def run_cmaes(fitness_fn, nz: int = NZ, sigma0: float = 1.0,
              maxfevals: int = 500) -> tuple:
    """
    Un run de CMA-ES.
    Retorna (best_z, best_score, fitness_history).
    best_score esta en [0,1] (sin negar).
    """
    try:
        import cma
    except ImportError:
        raise ImportError("Instala cma: pip install cma")

    z0 = np.random.randn(nz)
    es = cma.CMAEvolutionStrategy(z0, sigma0, {
        "maxfevals": maxfevals,
        "verbose":   -9,        # silencioso
        "tolx":      1e-6,
        "tolfun":    1e-7,
    })

    fitness_history = []
    best_z     = z0.copy()
    best_score = -np.inf

    while not es.stop():
        solutions  = es.ask()
        fitnesses  = [fitness_fn(z) for z in solutions]
        es.tell(solutions, fitnesses)

        # mejor de esta generacion (fitness negado -> menor = mejor)
        gen_best_idx   = int(np.argmin(fitnesses))
        gen_best_score = -fitnesses[gen_best_idx]

        if gen_best_score > best_score:
            best_score = gen_best_score
            best_z     = solutions[gen_best_idx].copy()

        fitness_history.append(gen_best_score)

    return best_z, best_score, fitness_history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CMA-ES en espacio latente del GAN")
    parser.add_argument("--model",      default=str(DEFAULT_MODEL),
                        help="Ruta al .pth del generador")
    parser.add_argument("--fitness",    default="f3_static",
                        choices=["f3_static", "structural_avg"],
                        help="Funcion de fitness a usar")
    parser.add_argument("--runs",       type=int,   default=20,
                        help="Numero de runs independientes")
    parser.add_argument("--evals",      type=int,   default=500,
                        help="Evaluaciones maximas por run")
    parser.add_argument("--sigma0",     type=float, default=1.0,
                        help="Sigma inicial de CMA-ES")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR),
                        help="Directorio de salida")
    args = parser.parse_args()

    model_path = Path(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"ERROR: No se encontro el modelo en {model_path}")
        return

    print(f"Modelo:  {model_path}")
    print(f"Fitness: {args.fitness}")
    print(f"Runs:    {args.runs}  |  Evals/run: {args.evals}")
    print("-" * 50)

    gen        = load_generator(model_path)
    fitness_fn = make_fitness_fn(gen, args.fitness)

    all_results     = []
    all_best_levels = []

    for run_i in range(args.runs):
        best_z, best_score, history = run_cmaes(
            fitness_fn, sigma0=args.sigma0, maxfevals=args.evals
        )
        best_level = z_to_level(gen, best_z)
        metrics    = structural_score(best_level)

        result = {
            "run":             run_i,
            "best_score":      float(best_score),
            "n_evals":         len(history),
            "fitness_history": [float(x) for x in history],
            "metrics":         {k: float(v) for k, v in metrics.items()},
        }
        all_results.append(result)
        all_best_levels.append(best_level.tolist())

        print(f"  Run {run_i + 1:2d}/{args.runs} "
              f"| best={best_score:.4f} "
              f"| structural_avg={metrics['structural_avg']:.4f} "
              f"| pipe={metrics['pipe_completeness']:.4f} "
              f"| ground={metrics['ground_continuity']:.4f}")

    # Metricas promedio sobre todos los mejores niveles
    avg_metrics = {
        k: float(np.mean([r["metrics"][k] for r in all_results]))
        for k in all_results[0]["metrics"]
    }

    print("\n=== Metricas promedio CMA-ES ({} runs) ===".format(args.runs))
    for k, v in avg_metrics.items():
        print(f"  {k:25s}: {v:.4f}")

    # Guardar
    summary = {
        "config":      vars(args),
        "n_runs":      args.runs,
        "avg_metrics": avg_metrics,
        "runs":        all_results,
    }

    suffix = f"_{Path(args.model).stem}_{args.fitness}"
    results_file = output_dir / f"cmaes_results{suffix}.json"
    levels_file  = output_dir / f"best_levels{suffix}.json"

    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)
    with open(levels_file, "w") as f:
        json.dump(all_best_levels, f)

    print(f"\nResultados -> {results_file}")
    print(f"Niveles    -> {levels_file}")


if __name__ == "__main__":
    main()
