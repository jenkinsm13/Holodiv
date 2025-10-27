def test_init_imports():
    """Test that all expected names are imported."""
    import holodiv as hd
    expected_imports = [
        'DimensionalArray',
        'array',
        'get_registry',
    ]
    for name in expected_imports:
        assert hasattr(hd, name)
