"""Deterministic smoke test for the package scaffold."""

import deltavision


def test_package_import_and_version_are_deterministic() -> None:
    assert deltavision.__version__ == "0.1.0"
