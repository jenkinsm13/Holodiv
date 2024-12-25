"""Custom exceptions for the dividebyzero package."""

class DimensionalError(Exception):
    """Error in dimensional operations."""
    pass

class ReconstructionError(Exception):
    """Exception raised for errors in the reconstruction process."""
    pass

class RegistryError(Exception):
    """Error in error registry operations."""
    pass