import pytest

def test_import():
    assert True

def test_requirements():
    import importlib
    for pkg in ["torch", "torchvision"]:
        importlib.import_module(pkg)
