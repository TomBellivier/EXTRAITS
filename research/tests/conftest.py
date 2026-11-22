"""Fixtures of the research tests: the same as tests/pose/ (see extraits/pose/testing/)."""

import pytest

from extraits.pose.testing.fake_ultralytics import *  # noqa: F403
from extraits.pose.testing.fixtures import *  # noqa: F403


@pytest.fixture(autouse=True)
def _isolated_workspace(tmp_path_factory, monkeypatch):
    """Never write into the user's real models folder ($EXTRAITS_MODELS)."""
    monkeypatch.delenv("EXTRAITS_MODELS", raising=False)
    monkeypatch.setenv("EXTRAITS_WORKSPACE", str(tmp_path_factory.mktemp("workspace")))
