import functools

from typing import Optional, Tuple


@functools.lru_cache(maxsize=None)
def load_spacy_model(model_names: Tuple[str, ...], disable: Optional[Tuple[str, ...]] = None):
    try:
        import spacy
    except ImportError as e:
        raise ImportError("Please install spacy. pip install spacy") from e

    if isinstance(model_names, str):
        model_names = (model_names,)

    if not isinstance(model_names, tuple):
        model_names = tuple(model_names)

    if disable is None:
        disable = []

    for model_name in model_names:  # Try all models except the last one
        try:
            return spacy.load(model_name, disable=disable)
        except OSError:
            print(f"{model_name} not found")

    # If none of the models worked, raise a clear error instead of silently downloading (which might fail offline).
    last_model = model_names[-1]
    raise RuntimeError(
        f"spaCy model '{last_model}' not found. Please install it manually, e.g. `python -m spacy download {last_model}`."
    )
