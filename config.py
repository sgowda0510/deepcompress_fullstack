# config.py — Central configuration
import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")
UPLOAD_DIR  = os.path.join(BASE_DIR, "uploads")

BATCH_SIZE       = 128
NUM_EPOCHS       = 3
LEARNING_RATE    = 1e-3
NUM_WORKERS      = 0
DEVICE           = "cpu"
NUM_CLASSES      = 10
INPUT_CHANNELS   = 3
PRUNING_AMOUNTS  = [0.2, 0.4]
NUM_INFERENCE_RUNS = 100
RESULTS_CSV      = os.path.join(RESULTS_DIR, "experiment_results.csv")
