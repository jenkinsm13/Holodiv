import dividebyzero as dbz

# Test 1: Basic division by zero
print("Test 1: Basic division by zero")
a = dbz.array([1.0, 2.0, 3.0])
b = dbz.array([1.0, 0.0, 1.0])
result = a / b
print(f"a/b = {result.array}")

# Test 2: Matrix division by zero with SVD reduction
print("\nTest 2: Matrix division by zero")
matrix = dbz.array([[1.0, 2.0], [3.0, 4.0]])
zero_div = dbz.array([[1.0, 0.0], [0.0, 1.0]])
result = matrix / zero_div
print(f"Matrix/zero_div = {result.array}")

# Test 3: Verify numpy operations still work
print("\nTest 3: Numpy operations")
c = dbz.array([1.0, 2.0, 3.0])
d = dbz.sin(c)  # Should use numpy's sin function
print(f"sin(c) = {d.array}")

# Test 4: Verify error registry is working
print("\nTest 4: Error registry")
result = a / 0
print(f"Error ID: {result._error_id}")
error_data = dbz.get_registry().retrieve(result._error_id)
print(f"Error data exists: {error_data is not None}")

# Test 5: Test elevation
print("\nTest 5: Elevation")
elevated = result.elevate()
print(f"Elevated result shape: {elevated.shape}") 
print(elevated)