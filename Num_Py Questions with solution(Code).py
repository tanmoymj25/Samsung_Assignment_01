import numpy as np

# 1. Create a 5×5 matrix with values 1,2,3,…,25
matrix_5x5 = np.arange(1, 26).reshape(5, 5)
print("Q1. 5x5 Matrix:\n", matrix_5x5)

# 2. Generate a 4×4 identity matrix
identity_4x4 = np.eye(4)
print("\nQ2. 4x4 Identity Matrix:\n", identity_4x4)

# 3. Create a 1D array of numbers from 100 to 200 with step size 10
array_step = np.arange(100, 201, 10)
print("\nQ3. Array 100–200 step 10:\n", array_step)

# 4. Generate a random 3×3 matrix and find its determinant
random_matrix_3x3 = np.random.rand(3, 3)
det_matrix = np.linalg.det(random_matrix_3x3)
print("\nQ4. Random 3x3 Matrix:\n", random_matrix_3x3)
print("Determinant:", det_matrix)

# 5. Create a NumPy array of 10 random integers between 1 and 100
random_integers = np.random.randint(1, 101, 10)
print("\nQ5. 10 Random Integers:\n", random_integers)


# 6. Reshape a 1D array of size 12 into a 3×4 matrix
array_1d = np.arange(12)
reshaped_array = array_1d.reshape(3, 4)
print("Q6. Reshaped 3x4 Matrix:\n", reshaped_array)

# 7. Create two 3×3 matrices and perform matrix multiplication
mat1 = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])
mat2 = np.array([[9, 8, 7],
                 [6, 5, 4],
                 [3, 2, 1]])
matrix_multiplication = np.dot(mat1, mat2)
print("\nQ7. Matrix Multiplication:\n", matrix_multiplication)

# 8. Find eigenvalues and eigenvectors of a given 2×2 matrix
matrix_2x2 = np.array([[4, -2],
                       [1,  1]])
eigenvalues, eigenvectors = np.linalg.eig(matrix_2x2)
print("\nQ8. Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# 9. Create a 5×5 matrix with random values and extract its diagonal elements
random_matrix_5x5 = np.random.rand(5, 5)
diagonal_elements = np.diag(random_matrix_5x5)
print("\nQ9. Random 5x5 Matrix:\n", random_matrix_5x5)
print("Diagonal Elements:", diagonal_elements)

# 10. Generate a 1D array and normalize it (scale values between 0 and 1)
arr = np.random.randint(1, 100, 10)
normalized_arr = (arr - arr.min()) / (arr.max() - arr.min())
print("\nQ10. Original Array:\n", arr)
print("Normalized Array:\n", normalized_arr)


# 11. Sort a NumPy array by row and column
arr_sort = np.array([[12, 5, 7],
                     [3, 8, 1],
                     [9, 4, 6]])
sorted_by_row = np.sort(arr_sort, axis=1)
sorted_by_col = np.sort(arr_sort, axis=0)
print("Q11. Original Array:\n", arr_sort)
print("Sorted by Row:\n", sorted_by_row)
print("Sorted by Column:\n", sorted_by_col)

# 12. Indices of maximum and minimum values in a NumPy array
arr_minmax = np.array([12, 45, 7, 89, 3, 56])
max_index = np.argmax(arr_minmax)
min_index = np.argmin(arr_minmax)
print("\nQ12. Array:", arr_minmax)
print("Max Index:", max_index, "Value:", arr_minmax[max_index])
print("Min Index:", min_index, "Value:", arr_minmax[min_index])

# 13. Flatten a 2D array
arr2d = np.array([[1, 2],
                  [3, 4]])
ravel_flat = arr2d.ravel()
flatten_flat = arr2d.flatten()
print("\nQ13. Original 2D Array:\n", arr2d)
print("Using ravel():", ravel_flat)
print("Using flatten():", flatten_flat)

# 14. Inverse of a 3×3 matrix
matrix_3x3 = np.array([[1, 2, 3],
                       [0, 1, 4],
                       [5, 6, 0]])
inverse_matrix = np.linalg.inv(matrix_3x3)
print("\nQ14. Original 3x3 Matrix:\n", matrix_3x3)
print("Inverse Matrix:\n", inverse_matrix)

# 15. Random permutation of numbers 1 to 10
random_perm = np.random.permutation(np.arange(1, 11))
print("\nQ15. Random Permutation of 1–10:\n", random_perm)


# 16. Replace even numbers with -1
arr_even = np.arange(21)   # values 0 to 20
arr_even[arr_even % 2 == 0] = -1
print("Q16. Replace evens with -1:\n", arr_even)

# 17. Dot product of two arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot_product = np.dot(a, b)
print("\nQ17. Dot Product of", a, "and", b, "=", dot_product)

# 18. Trace of a 5×5 random matrix
matrix_5x5 = np.random.randint(1, 10, (5, 5))
trace_val = np.trace(matrix_5x5)
print("\nQ18. Random 5x5 Matrix:\n", matrix_5x5)
print("Trace of Matrix =", trace_val)

# 19. Split a 1D array into 3 equal parts
arr_split = np.arange(1, 13)   # 12 elements
split_parts = np.split(arr_split, 3)
print("\nQ19. Original Array:", arr_split)
print("Split into 3 parts:", split_parts)

# 20. 3D array mean across axis=0
arr3d = np.random.randint(1, 10, (3, 3, 3))
mean_axis0 = arr3d.mean(axis=0)
print("\nQ20. 3D Array:\n", arr3d)
print("Mean across axis=0:\n", mean_axis0)
# 21. Cumulative sum of a NumPy array
arr_cumsum = np.array([1, 2, 3, 4, 5])
cumsum_result = np.cumsum(arr_cumsum)
print("Q21. Array:", arr_cumsum)
print("Cumulative Sum:", cumsum_result)

# 22. Upper triangular of a 4×4 random matrix
matrix_4x4 = np.random.randint(1, 10, (4, 4))
upper_tri = np.triu(matrix_4x4)
print("\nQ22. Original 4x4 Matrix:\n", matrix_4x4)
print("Upper Triangular Matrix:\n", upper_tri)

# 23. 6×6 checkerboard pattern (0,1)
checkerboard = np.zeros((6, 6), dtype=int)
checkerboard[1::2, ::2] = 1
checkerboard[::2, 1::2] = 1
print("\nQ23. 6x6 Checkerboard Pattern:\n", checkerboard)

# 24. Element-wise square root of a random 3×3 matrix
matrix_sqrt = np.random.randint(1, 10, (3, 3))
sqrt_result = np.sqrt(matrix_sqrt)
print("\nQ24. Original 3x3 Matrix:\n", matrix_sqrt)
print("Square Root Matrix:\n", sqrt_result)

# 25. Reverse a 1D array of 20 elements without slicing
arr_reverse = np.arange(1, 21)
reversed_array = np.flip(arr_reverse)
print("\nQ25. Original Array:", arr_reverse)
print("Reversed Array:", reversed_array)

# 26. Merge two arrays vertically and horizontally
arr1 = np.array([[1, 2],
                 [3, 4]])
arr2 = np.array([[5, 6],
                 [7, 8]])
merge_vert = np.vstack((arr1, arr2))
merge_horiz = np.hstack((arr1, arr2))
print("Q26. Vertical Merge:\n", merge_vert)
print("Horizontal Merge:\n", merge_horiz)

# 27. Row-wise and column-wise sum of a 2D array
arr2d_sum = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
row_sum = arr2d_sum.sum(axis=1)
col_sum = arr2d_sum.sum(axis=0)
print("\nQ27. Array:\n", arr2d_sum)
print("Row-wise Sum:", row_sum)
print("Column-wise Sum:", col_sum)

# 28. Replace NaN values with column mean
arr_nan = np.array([[1, 2, np.nan],
                    [4, np.nan, 6],
                    [7, 8, 9]])
col_mean = np.nanmean(arr_nan, axis=0)
inds = np.where(np.isnan(arr_nan))
arr_nan[inds] = np.take(col_mean, inds[1])
print("\nQ28. After replacing NaN with column mean:\n", arr_nan)

# 29. Cosine similarity between two 1D arrays
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
cos_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
print("\nQ29. Cosine Similarity =", cos_sim)

# 30. Rotate a 4×4 array by 90 degrees
arr_rotate = np.arange(1, 17).reshape(4, 4)
rotated = np.rot90(arr_rotate)
print("\nQ30. Original 4x4 Array:\n", arr_rotate)
print("Rotated 90 degrees:\n", rotated)

# 31. Structured array with fields (name, age, marks)
students = np.array([("Alice", 20, 85.5),
                     ("Bob", 22, 90.0),
                     ("Charlie", 21, 78.0)],
                    dtype=[("name", "U10"), ("age", "i4"), ("marks", "f4")])
print("Q31. Structured Array:\n", students)
print("Names:", students["name"])
print("Ages:", students["age"])
print("Marks:", students["marks"])

# 32. Rank of a random 3×3 matrix
matrix_rank = np.random.randint(1, 10, (3, 3))
rank = np.linalg.matrix_rank(matrix_rank)
print("\nQ32. Random 3x3 Matrix:\n", matrix_rank)
print("Rank of Matrix:", rank)

# 33. Normalize each row of a 5×5 random matrix to unit length
matrix_norm = np.random.randint(1, 10, (5, 5))
row_norms = np.linalg.norm(matrix_norm, axis=1, keepdims=True)
normalized_matrix = matrix_norm / row_norms
print("\nQ33. Original Matrix:\n", matrix_norm)
print("Row-normalized Matrix:\n", normalized_matrix)

# 34. Check whether two arrays are equal element-wise
arr_a = np.array([1, 2, 3])
arr_b = np.array([1, 2, 4])
equal_check = np.array_equal(arr_a, arr_b)
elementwise_check = arr_a == arr_b
print("\nQ34. Array A:", arr_a)
print("Array B:", arr_b)
print("Are equal? ->", equal_check)
print("Element-wise equality:", elementwise_check)

# 35. Histogram of 1000 random numbers
data = np.random.randn(1000)  # standard normal distribution
hist, bins = np.histogram(data, bins=10)
print("\nQ35. Histogram Counts:", hist)
print("Bin Edges:", bins)

# 36. Broadcasting: add 1D array to 2D array
arr2d_broadcast = np.array([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]])
arr1d_broadcast = np.array([10, 20, 30])
result_broadcast = arr2d_broadcast + arr1d_broadcast
print("Q36. 2D Array:\n", arr2d_broadcast)
print("1D Array:", arr1d_broadcast)
print("After Broadcasting:\n", result_broadcast)

# 37. Unique values and counts
arr_unique = np.array([1, 2, 2, 3, 4, 4, 4, 5])
unique_vals, counts = np.unique(arr_unique, return_counts=True)
print("\nQ37. Array:", arr_unique)
print("Unique Values:", unique_vals)
print("Counts:", counts)

# 38. Pearson correlation coefficient
x = np.array([1, 2, 3, 4, 5])
y = np.array([10, 20, 30, 40, 50])
pearson_corr = np.corrcoef(x, y)[0, 1]
print("\nQ38. Arrays:\nX:", x, "\nY:", y)
print("Pearson Correlation Coefficient:", pearson_corr)

# 39. Numerical gradient of a 1D array
arr_grad = np.array([1, 2, 4, 7, 11])
gradient = np.gradient(arr_grad)
print("\nQ39. Array:", arr_grad)
print("Gradient:", gradient)

# 40. Singular Value Decomposition (SVD)
matrix_svd = np.random.randint(1, 10, (3, 3))
U, S, Vt = np.linalg.svd(matrix_svd)
print("\nQ40. Original 3x3 Matrix:\n", matrix_svd)
print("U Matrix:\n", U)
print("Singular Values:", S)
print("V Transpose:\n", Vt)
