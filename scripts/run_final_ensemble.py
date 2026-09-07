"""CLI последовательного final-fit ансамбля s62."""

from pathlib import Path

from uzner.training.final_ensemble import run_queue

if __name__ == "__main__":
    run_queue(Path.cwd())
