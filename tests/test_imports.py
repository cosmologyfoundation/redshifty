"""Test that all packages import correctly."""


def test_src_imports():
    """Smoke test: verify src subpackages can be imported."""
    import src
    import src.tokenizers
    import src.models
    import src.training
    import src.evaluation
    import src.utils
    assert src is not None
