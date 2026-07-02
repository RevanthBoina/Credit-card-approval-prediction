# TODO

- [x] Step 1: Create `src/` directory with `__init__.py`, and implement `src/app.py`, `src/model.py`, `src/utils.py` as the new required entrypoints.

- [ ] Step 2: Move code from `extracted/Credit-card-approval-prediction/` into the new layout (templates/static/models/preprocessing/utils/preprocessing/config) and remove `extracted/`.
- [ ] Step 3: Update all imports so training and Flask use `src` modules.
- [ ] Step 4: Implement artifact paths to use `saved_models/` and dataset paths to use `data/raw` + `data/processed`.
- [ ] Step 5: Update `.gitignore`, `README.md`, `Procfile`, and `manifest.yml` for new entrypoints.
- [ ] Step 6: Run smoke tests: `python -m src.model` and `python -c "from src.app import create_app; app=create_app()"`.

