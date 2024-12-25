"""Test suite for error registry system."""

import pytest
import numpy as np
from dividebyzero.registry import ErrorRegistry, ErrorData
from dividebyzero.exceptions import RegistryError
from . import generate_test_array

class TestErrorRegistry:
    """Test suite for ErrorRegistry class."""
    
    @pytest.fixture
    def registry(self):
        """Provide clean registry for each test."""
        return ErrorRegistry()
    
    @pytest.fixture
    def sample_error_data(self):
        """Provide sample error data for testing."""
        return ErrorData(
            original_shape=(3, 3),
            error_tensor=generate_test_array((3, 3)),
            reduction_type='complete',
            mask=None
        )

    def test_registry_creation(self, registry):
        """Test registry initialization."""
        assert len(registry) == 0
        assert isinstance(registry._errors, dict)

    def test_error_storage(self, registry, sample_error_data):
        """Test storing error information."""
        # Store error data
        error_id = registry.store(sample_error_data)
        
        # Verify storage
        assert len(registry) == 1
        assert error_id in registry
        
        # Check data integrity
        stored_data = registry.retrieve(error_id)
        assert stored_data == sample_error_data
        assert stored_data.original_shape == (3, 3)
        assert stored_data.reduction_type == 'complete'

    def test_error_retrieval(self, registry, sample_error_data):
        """Test retrieving error information."""
        # Store and retrieve
        error_id = registry.store(sample_error_data)
        retrieved_data = registry.retrieve(error_id)
        
        # Check data
        assert retrieved_data.original_shape == sample_error_data.original_shape
        np.testing.assert_array_equal(
            retrieved_data.error_tensor,
            sample_error_data.error_tensor
        )
        
        # Test non-existent ID
        assert registry.retrieve('non-existent') is None

    def test_registry_cleanup(self, registry, sample_error_data):
        """Test garbage collection behavior."""
        import gc
        import weakref
        
        # Create a reference-counting test class
        class TestRef:
            def __init__(self, data):
                self.data = data
        
        # Store with weak reference
        test_obj = TestRef(sample_error_data)
        weak_ref = weakref.ref(test_obj)
        error_id = registry.store(sample_error_data)
        
        # Delete original reference
        del test_obj
        gc.collect()
        
        # Verify weak reference behavior
        assert weak_ref() is None
        assert registry.retrieve(error_id) is not None  # Original data persists

    def test_partial_error_data(self, registry):
        """Test storing partial reduction data."""
        # Create partial reduction data
        mask = np.array([[True, False], [False, True]])
        error_data = ErrorData(
            original_shape=(2, 2),
            error_tensor=np.ones((2, 2)),
            reduction_type='partial',
            mask=mask
        )
        
        # Store and retrieve
        error_id = registry.store(error_data)
        retrieved = registry.retrieve(error_id)
        
        # Verify mask and data integrity
        assert retrieved.reduction_type == 'partial'
        np.testing.assert_array_equal(retrieved.mask, mask)
        np.testing.assert_array_equal(retrieved.error_tensor, np.ones((2, 2)))

    def test_registry_clear(self, registry, sample_error_data):
        """Test clearing registry."""
        # Store multiple error entries
        error_ids = [registry.store(sample_error_data) for _ in range(3)]
        assert len(registry) == 3
        
        # Clear registry
        registry.clear()
        
        # Verify complete cleanup
        assert len(registry) == 0
        for error_id in error_ids:
            assert registry.retrieve(error_id) is None

    def test_concurrent_access(self, registry, sample_error_data):
        """Test concurrent registry access."""
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        access_lock = threading.Lock()
        stored_ids = []
        
        def store_and_retrieve():
            with access_lock:
                error_id = registry.store(sample_error_data)
                stored_ids.append(error_id)
            
            # Attempt retrieval
            retrieved = registry.retrieve(error_id)
            return retrieved is not None
        
        # Test with multiple threads
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _: store_and_retrieve(), range(10)))
        
        # Verify all operations succeeded and data integrity
        assert all(results)
        assert len(registry) == len(stored_ids)
        for error_id in stored_ids:
            retrieved = registry.retrieve(error_id)
            assert retrieved == sample_error_data

    def test_error_data_immutability(self, registry, sample_error_data):
        """Test immutability of stored error data."""
        error_id = registry.store(sample_error_data)
        retrieved = registry.retrieve(error_id)
        
        # Attempt to modify retrieved data in various ways
        with pytest.raises(AttributeError):
            retrieved.error_tensor = np.zeros_like(retrieved.error_tensor)
            
        with pytest.raises(AttributeError):
            retrieved.original_shape = (4, 4)
            
        with pytest.raises(AttributeError):
            retrieved.reduction_type = 'modified'
            
        # Verify data remains unchanged
        fresh_retrieval = registry.retrieve(error_id)
        assert fresh_retrieval == sample_error_data
        np.testing.assert_array_equal(
            fresh_retrieval.error_tensor,
            sample_error_data.error_tensor
        )

    def test_registry_persistence(self, registry, sample_error_data):
        """Test registry data persistence across operations."""
        # Store initial data
        error_id = registry.store(sample_error_data)
        
        # Perform multiple store/retrieve operations
        for _ in range(10):
            temp_id = registry.store(sample_error_data)
            registry.retrieve(temp_id)
            
        # Verify original data persists unchanged
        final_retrieval = registry.retrieve(error_id)
        assert final_retrieval == sample_error_data
        
    def test_error_data_validation(self, registry):
        """Test validation of error data structure."""
        # Test with invalid shape
        with pytest.raises(ValueError):
            ErrorData(
                original_shape=(),  # Empty shape
                error_tensor=np.array([1, 2, 3]),
                reduction_type='complete'
            )
        
        # Test with mismatched dimensions
        with pytest.raises(ValueError):
            ErrorData(
                original_shape=(2, 2),
                error_tensor=np.array([1, 2, 3]),  # Wrong shape
                reduction_type='complete'
            )
        
        # Test with invalid reduction type
        with pytest.raises(ValueError):
            ErrorData(
                original_shape=(2, 2),
                error_tensor=np.ones((2, 2)),
                reduction_type='invalid_type'  # Invalid type
            )