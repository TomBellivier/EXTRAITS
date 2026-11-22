"""Safety net of the whole test suite.

The user's real models folder can be set in the environment ($EXTRAITS_MODELS), e.g.
to run the end-to-end test of `extraits process`. A command that WRITES models
(`extraits train ...`, `extraits finetune`) run by a test with the default paths would
then overwrite it. So every test runs in a throwaway workspace, without
$EXTRAITS_MODELS, except the tests marked `models`, which only read the models.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_workspace(request, tmp_path_factory, monkeypatch):
    if request.node.get_closest_marker("models") is None:
        monkeypatch.delenv("EXTRAITS_MODELS", raising=False)
        monkeypatch.setenv("EXTRAITS_WORKSPACE", str(tmp_path_factory.mktemp("workspace")))
