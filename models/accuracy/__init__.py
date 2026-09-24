"""The accuracy model: TinyLlama perplexity with the recipe's arithmetic in every linear layer.

accuracy.py is the model (available, key, run, line, dry_run); rules.py names which layers are patched;
_worker.py is the per-GPU subprocess that loads the model and sums the loss. `python -m models.accuracy` runs
it alone.
"""
from .accuracy import MODEL_ID, available, dry_run, key, line, run  # noqa: F401
