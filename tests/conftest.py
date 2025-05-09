import numpy as np

# Monkeypatch for NumPy 2.x compatibility with libraries expecting np.float_, np.int_, np.uint
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64
if not hasattr(np, "uint"):
    np.uint = np.uint64
