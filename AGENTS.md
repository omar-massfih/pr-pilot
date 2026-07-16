# Repository guidance

- This project supports Python 3.11 and newer and intentionally has no runtime dependencies.
- Keep provider-specific CLI behavior inside `src/pr_pilot/providers.py`.
- Keep Git and GitHub mutations in their respective adapter modules.
- Never put API tokens in prompts, logs, configuration examples, or persisted run state.
- Run `python -m unittest discover -s tests -v` after changes.
