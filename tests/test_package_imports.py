"""Smoke tests: every package/module is importable without extra installs.

In particular, ``data.featurize`` must stay importable even though RDKit
is not yet installed in the target conda env, because RDKit is imported
lazily inside its functions rather than at module scope.
"""

import aidd_generative_docking
from aidd_generative_docking import data, diffusion, evaluation, models, training
from aidd_generative_docking.data import featurize, pdbbind
from aidd_generative_docking.diffusion import flow_matching
from aidd_generative_docking.evaluation import metrics
from aidd_generative_docking.models import equivariant
from aidd_generative_docking.training import train


def test_version_string() -> None:
    assert aidd_generative_docking.__version__ == "0.0.1"


def test_submodules_are_modules() -> None:
    for module in (data, diffusion, evaluation, models, training):
        assert module is not None


def test_leaf_modules_importable() -> None:
    for module in (featurize, pdbbind, flow_matching, metrics, equivariant, train):
        assert module is not None
