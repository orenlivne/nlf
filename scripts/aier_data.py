"""aier_data.py — resolve the consolidated aier data hub via the $AIER_DATA env var,
with a fallback to the package's own committed ``data/`` so a fresh ``git clone``
(e.g. a SISC reviewer's machine, with no hub and no $AIER_DATA) still finds the
vendored / fetched data.

Canonical copy lives in ~/code/aier/data (this file); an identical copy is shipped
in each package's scripts/ dir::

    from aier_data import aier_corpus
    df = pd.read_csv(aier_corpus("mtx_sizes.csv"))

Resolution order for aier_corpus/aier_ssl/...:
  1. $AIER_DATA/<subdir>/<parts>   (the hub; default ~/code/aier/data)
  2. <repo>/data/<parts>           (in-repo committed/fetched data — reviewer path)
  3. the hub path                  (so a missing file yields a clear error)
"""
from __future__ import annotations

import os


def aier_data_root() -> str:
    """Root of the data hub: ``$AIER_DATA`` or ``~/code/aier/data``."""
    return os.environ.get("AIER_DATA", os.path.expanduser("~/code/aier/data"))


def _inrepo() -> str:
    """The package's own committed data dir (this file lives in scripts/, so ../data)."""
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    )


def _aier(sub: str, *parts: str) -> str:
    hub = os.path.join(aier_data_root(), sub, *parts)
    if os.path.exists(hub):
        return hub
    if not parts:
        return _inrepo() if os.path.isdir(_inrepo()) else hub
    inrepo = os.path.join(_inrepo(), *parts)
    return inrepo if os.path.exists(inrepo) else hub


def aier_data(*parts: str) -> str:
    """Join ``parts`` under the data-hub root (no in-repo fallback)."""
    return os.path.join(aier_data_root(), *parts)


def aier_corpus(*parts: str) -> str:
    """SuiteSparse / SNAP ``.mtx`` corpus (hub ``suitesparse/``, else in-repo ``data/``)."""
    return _aier("suitesparse", *parts)


def aier_ssl(*parts: str) -> str:
    """MNIST / FashionMNIST for NP (hub ``ssl/``, else in-repo ``data/``)."""
    return _aier("ssl", *parts)


def aier_spe10(*parts: str) -> str:
    """SPE10 pressure-Laplacian matrix (hub ``spe10/``, else in-repo ``data/``)."""
    return _aier("spe10", *parts)
