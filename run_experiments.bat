@echo off
:: ============================================================
:: run_experiments.bat
:: Ejecutar desde ML/Proyecto/
:: Reproduce los experimentos A, B y C del proyecto NeuralPlumber
:: ============================================================

echo ============================================================
echo  NeuralPlumber - Pipeline de experimentos
echo ============================================================
echo.

:: --- Experimento A: Baseline ---
echo [1/6] Experimento A - Baseline (100 niveles, modelo original)...
python NeuralPlumber/src/baseline/reproduce_volz.py --n_samples 100
echo.

:: --- Dataset expandido ---
echo [2/6] Construyendo dataset expandido (15 niveles VGLC)...
python NeuralPlumber/src/data/vglc_parser.py ^
  --input_dir "Datasets/TheVGLC/Super Mario Bros/Processed/" ^
  --output NeuralPlumber/data/dataset_full.json
echo.

:: --- Re-entrenamiento GAN ---
echo [3/6] Copiando dataset al directorio DagstuhlGAN...
copy clone\DagstuhlGAN\pytorch\example.json clone\DagstuhlGAN\pytorch\example_original.json
copy NeuralPlumber\data\dataset_full.json clone\DagstuhlGAN\pytorch\example.json
echo.

echo [4/6] Re-entrenando GAN (5000 epocas, ~50 min en CPU)...
cd clone\DagstuhlGAN\pytorch
python main.py --nz 32 --ngf 64 --ndf 64 --batchSize 32 --niter 5000
cd ..\..\..
echo.

echo [5/6] Copiando modelo re-entrenado...
copy clone\DagstuhlGAN\pytorch\netG_epoch_5000.pth NeuralPlumber\models\netG_15levels.pth

echo [5/6] Midiendo metricas del modelo re-entrenado (Experimento B)...
python NeuralPlumber/src/baseline/reproduce_volz.py ^
  --n_samples 100 ^
  --model_path NeuralPlumber/models/netG_15levels.pth ^
  --output_dir NeuralPlumber/experiments/expanded_data/
echo.

:: --- Experimento C: CMA-ES + F3 ---
echo [6/6] Experimento C - CMA-ES con F3 (40 runs, 1000 evals, ~20 min)...
python NeuralPlumber/src/evolution/cmaes_runner.py ^
  --model NeuralPlumber/models/netG_15levels.pth ^
  --fitness f3_static ^
  --runs 40 ^
  --evals 1000 ^
  --output_dir NeuralPlumber/experiments/structural_fitness/
echo.

echo ============================================================
echo  Pipeline completado. Resultados en NeuralPlumber/experiments/
echo ============================================================
pause
