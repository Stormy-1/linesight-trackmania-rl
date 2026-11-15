"""Load ``.joblib`` files without letting them run code.

``joblib.load`` is pickle underneath: a crafted file can call any Python
function while it loads, so opening a stats file someone else sent you would
run their code. The project only ever stores plain data in these files
(numbers, strings, lists, dicts, ``defaultdict`` and numpy arrays), so
``safe_joblib_load`` refuses every other class or function a file names.

joblib (1.6+) routes the nested pickle of object-dtype arrays through the
same ``find_class``, so the allowlist covers those too.
"""

import pickle
from pathlib import Path
from typing import Any

from joblib import numpy_pickle

_NUMPY_RECONSTRUCTORS = frozenset({"_reconstruct", "scalar"})

ALLOWED_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        *(("builtins", name) for name in ("int", "float", "complex", "bool", "str", "bytes",
                                          "bytearray", "list", "dict", "tuple", "set", "frozenset")),
        ("collections", "defaultdict"),
        ("collections", "OrderedDict"),
        ("collections", "deque"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        *(("numpy.core.multiarray", name) for name in _NUMPY_RECONSTRUCTORS),
        *(("numpy._core.multiarray", name) for name in _NUMPY_RECONSTRUCTORS),
        ("joblib.numpy_pickle", "NumpyArrayWrapper"),
    }
)


def _is_allowed(module: str, name: str) -> bool:
    if (module, name) in ALLOWED_GLOBALS:
        return True
    # numpy 2 pickles some dtypes by their class, e.g. numpy.dtypes.Float64DType.
    return module == "numpy.dtypes" and name.endswith("DType")


class _RestrictedNumpyUnpickler(numpy_pickle.NumpyUnpickler):
    def find_class(self, module: str, name: str) -> Any:
        if not _is_allowed(module, name):
            raise pickle.UnpicklingError(f"Refusing to load {module}.{name}: only plain data is allowed")
        return super().find_class(module, name)


def safe_joblib_load(path: Path | str) -> Any:
    """``joblib.load`` for plain-data files that rejects anything able to run code."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"joblib file not found: {path}")
    filename = str(path)
    with open(path, "rb") as raw:
        # Private joblib helper: it unwraps compressed files the same way joblib.load does.
        with numpy_pickle._validate_fileobject_and_memmap(raw, filename, None) as (fobj, _):
            if isinstance(fobj, str):
                raise pickle.UnpicklingError(f"{path} uses the pre-0.10 joblib format, which is not supported")
            unpickler = _RestrictedNumpyUnpickler(filename, fobj, ensure_native_byte_order="auto")
            return unpickler.load()
