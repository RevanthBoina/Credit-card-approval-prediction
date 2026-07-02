# TODO - Move extracted project files to repo root

- [ ] Verify current layout (confirm only `extracted/` exists at workspace root)
- [ ] Plan file move strategy: move `extracted/Credit-card-approval-prediction/*` to repo root, then remove `extracted/`
- [ ] Implement move via shell commands (robust: handle hidden files and directories)
- [ ] Update/verify any paths if needed (e.g., references to `extracted/`)
- [ ] Run a quick sanity check: `python -m models.train` (optional) or `python -c "import app"`
- [ ] Confirm repo shows files directly at root (as user expects in Codespaces/GitHub)

