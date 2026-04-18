# post_generation (simplified)

This subproject is now organized around a small set of clear entrypoints and shared utilities.

## Folder layout

- app/
  - train_evader_app.py: main evader training implementation
  - train_detector_app.py: main detector training implementation
- core/
  - runtime.py: path/bootstrap helpers for script execution
  - logging_utils.py: shared logging configuration
- attack/
  - attack recipes, methods, baselines, and smoke scripts
- train_evader.py
  - backward-compatible wrapper that calls app/train_evader_app.py
- train_detector.py
  - backward-compatible wrapper that calls app/train_detector_app.py
- main.py
  - unified command entrypoint for train and attack workflows

## Common commands

From workspace root:

- python post_generation/train_evader.py --help
- python post_generation/train_detector.py --help
- python post_generation/main.py train-evader --help
- python post_generation/main.py train-detector --help
- python post_generation/main.py attack --help
- python post_generation/main.py smoke-phase2 --help
- python post_generation/main.py smoke-semantic --help

From post_generation directory:

- python train_evader.py --help
- python train_detector.py --help
- python main.py train-evader --help
- python main.py train-detector --help
