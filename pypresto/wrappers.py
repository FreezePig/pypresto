"""
Python wrappers for matrix operations with comprehensive error handling.

This module provides high-level Python interfaces for C++ matrix operations,
including input validation, type checking, and proper error handling.
"""
import sys
import os
from pathlib import Path
import ctypes
import numpy as np
from typing import List

try:
    from . import matrix_module as mm
except ImportError as e:
    raise ImportError(
        "Failed to import C++ extension 'matrix_module'. "
        "Please ensure the package is installed correctly (not just cloned)."
    ) from e

integer_dtypes = {
    np.dtype('uint8'), np.dtype('uint16'), np.dtype('uint32'), np.dtype('uint64'),
    np.dtype('int8'), np.dtype('int16'), np.dtype('int32'), np.dtype('int64')
}

def sumGroups_csc(
    x: np.ndarray,
    p: np.ndarray,
    i: np.ndarray,
    nrow: int,
    ncol: int,
    groups: np.ndarray,
    ngroups: int,
    nthreads: int = 1
) -> np.ndarray:
    """
    Sum groups for CSC (Compressed Sparse Column) sparse matrix by column.
    
    This function performs group-wise summation of a sparse matrix stored in CSC format
    Each row belongs to a group, and the function sums all values within each group
    for each column.
    
    Parameters
    ----------
    x : np.ndarray
        1D array of non-zero values in the sparse matrix (dtype: float64).
    p : np.ndarray
        1D array of column pointers indicating start/end indices for each column
        in arrays x and i (dtype: size_t/uint64). Length should be ncol + 1.
    i : np.ndarray
        1D array of row indices corresponding to each non-zero value in x
        (dtype: size_t/uint64).
    ncol : int
        Number of columns in the sparse matrix. Must be positive.
    nrow: int
        Number of rows in the sparse matrix. Must be positive. 
        Just for input tests, not used for calc.
    groups : np.ndarray
        1D array indicating which group each row belongs to (dtype: size_t/uint64).
        Values should be in range [0, ngroups-1].
    ngroups : int
        Total number of groups. Must be positive.
    nthreads: int, optional
        Number of threads to use for computation. Default is 1.
        If nthread <= 0 or >= largest threads, uses all available threads.
        
    Returns
    -------
    np.ndarray
        2D array of shape (ngroups, ncol) containing the sum of values for each
        group and column combination (dtype: float64).
    """

    # Type validation
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Parameter 'x' must be a numpy array, got {type(x)}")
    if not isinstance(p, np.ndarray):
        raise TypeError(f"Parameter 'p' must be a numpy array, got {type(p)}")
    if not isinstance(i, np.ndarray):
        raise TypeError(f"Parameter 'i' must be a numpy array, got {type(i)}")
    if not isinstance(groups, np.ndarray):
        raise TypeError(f"Parameter 'groups' must be a numpy array, got {type(groups)}")
    
    # Scalar parameter validation
    if not isinstance(ncol, int):
        raise TypeError(f"Parameter 'ncol' must be an integer, got {type(ncol)}")
    if not isinstance(nrow, int):
        raise TypeError(f"Parameter 'nrow' must be an integer, got {type(nrow)}")
    if not isinstance(ngroups, int):
        raise TypeError(f"Parameter 'ngroups' must be an integer, got {type(ngroups)}")
    if not isinstance(nthreads, int):
        raise TypeError(f"Parameter 'nthreads' must be an integer, got {type(nthreads)}")
    
    if ncol <= 0:
        raise ValueError(f"Parameter 'ncol' must be positive, got {ncol}")
    if nrow <= 0:
        raise ValueError(f"Parameter 'nrow' must be positive, got {nrow}")
    if ngroups <= 0:
        raise ValueError(f"Parameter 'ngroups' must be positive, got {ngroups}")
    if nthreads <= 0:
        print("Warning: nthreads <= 0, using all available threads")
    
    # Array dimension validation
    if x.ndim != 1:
        raise ValueError(f"Parameter 'x' must be 1-dimensional, got {x.ndim}D")
    if p.ndim != 1:
        raise ValueError(f"Parameter 'p' must be 1-dimensional, got {p.ndim}D")
    if i.ndim != 1:
        raise ValueError(f"Parameter 'i' must be 1-dimensional, got {i.ndim}D")
    if groups.ndim != 1:
        raise ValueError(f"Parameter 'groups' must be 1-dimensional, got {groups.ndim}D")
    
    # Array length validation
    if len(x) != len(i):
        raise ValueError(f"Arrays 'x' and 'i' must have the same length, "
                        f"got x: {len(x)}, i: {len(i)}")
    
    if len(p) != ncol + 1:
        raise ValueError(f"Array 'p' must have length ncol + 1 = {ncol + 1}, "
                        f"got {len(p)}")
    
    if len(x) == 0:
        raise ValueError("Array 'x' cannot be empty")
    
    # Data type validation for integer arrays
    if x.dtype != np.float64:
        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError(f"Input array must be floating-point, got {x.dtype}")
        x = x.astype(np.float64)

    # Check integer dtypes (allow uint64, int64, uint32, int32)
    if p.dtype not in integer_dtypes:
        raise TypeError(f"Array 'p' must have integer dtype (uint64/int64/uint32/int32), got {p.dtype}")
    if i.dtype not in integer_dtypes:
        raise TypeError(f"Array 'i' must have integer dtype (uint64/int64/uint32/int32), got {i.dtype}")
    if groups.dtype not in integer_dtypes:
        raise TypeError(f"Array 'groups' must have integer dtype (uint64/int64/uint32/int32), got {groups.dtype}")
    p, i, groups = p.astype(np.int32), i.astype(np.int32), groups.astype(np.int32)

    # Ensure x are positive
    if np.any(x <= 0):
        # Find the minimum value for better error message
        min_val = np.min(x)
        raise ValueError(f"All elements in 'x' must be positive (greater than 0), "
                         f"but found minimum value: {min_val}")
    
    # Value range validation
    if len(groups) != nrow:
        raise ValueError(f"Array 'groups' must have length equal to number of rows ({nrow}), "
                        f"got {len(groups)}")
    if len(groups) > 0:
        max_group = np.max(groups)
        if max_group >= ngroups:
            raise ValueError(f"Group indices must be in range [0, {ngroups-1}], "
                             f"found maximum group index: {max_group}")
    
    if len(i) > 0:
        max_row = np.max(i)
        if max_row >= len(groups):
            raise ValueError(f"Row indices in 'i' must be in range [0, {len(groups)-1}], "
                             f"found maximum row index: {max_row}")
    
    # Validate CSC format constraints
    if len(p) > 0:
        if p[0] != 0:
            raise ValueError(f"First element of 'p' must be 0, got {p[0]}")
        
        if p[-1] != len(x):
            raise ValueError(f"Last element of 'p' must equal length of 'x' ({len(x)}), "
                           f"got {p[-1]}")
        
        # Check that p is non-decreasing
        if not np.all(np.diff(p) >= 0):
            raise ValueError("Array 'p' must be non-decreasing (valid CSC format)")
    
    # Call the C++ function
    try:
        result = mm.cpp_sumGroups_csc(x, p, i, ncol, groups, ngroups, nthreads)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    
def sumGroups_csr(
    x: np.ndarray,
    p: np.ndarray,
    i: np.ndarray,
    nrow: int,
    ncol: int,
    groups: np.ndarray,
    ngroups: int,
    nthreads: int = 1
) -> np.ndarray:
    """
    Sum groups for CSR (Compressed Sparse Row) sparse matrix by column.
    
    This function performs group-wise summation of a sparse matrix stored in CSR format.
    Each row belongs to a group, and the function sums all values within each group
    for each column.
    
    Parameters
    ----------
    x : np.ndarray
        1D array of non-zero values in the sparse matrix (dtype: float64). 
        All values must be positive (greater than 0).
    p : np.ndarray
        1D array of row pointers indicating start/end indices for each row
        in arrays x and i (dtype: size_t/uint64). Length should be nrow + 1.
    i : np.ndarray
        1D array of column indices corresponding to each non-zero value in x
        (dtype: size_t/uint64).
    ncol : int
        Number of columns in the sparse matrix. Must be positive.
    nrow : int
        Number of rows in the sparse matrix. Must be positive.
    groups : np.ndarray
        1D array indicating which group each row belongs to (dtype: size_t/uint64).
        Values should be in range [0, ngroups-1]. Length should be nrow.
    ngroups : int
        Total number of groups. Must be positive.
    nthreads: int, optional
        Number of threads to use for computation. Must be positive. Default is 1.
        
    Returns
    -------
    np.ndarray
        2D array of shape (ngroups, ncol) containing the sum of values for each
        group and column combination (dtype: float64).
    """

    # Type validation
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Parameter 'x' must be a numpy array, got {type(x)}")
    if not isinstance(p, np.ndarray):
        raise TypeError(f"Parameter 'p' must be a numpy array, got {type(p)}")
    if not isinstance(i, np.ndarray):
        raise TypeError(f"Parameter 'i' must be a numpy array, got {type(i)}")
    if not isinstance(groups, np.ndarray):
        raise TypeError(f"Parameter 'groups' must be a numpy array, got {type(groups)}")
    
    # Scalar parameter validation
    if not isinstance(ncol, int):
        raise TypeError(f"Parameter 'ncol' must be an integer, got {type(ncol)}")
    if not isinstance(nrow, int):
        raise TypeError(f"Parameter 'nrow' must be an integer, got {type(nrow)}")
    if not isinstance(ngroups, int):
        raise TypeError(f"Parameter 'ngroups' must be an integer, got {type(ngroups)}")
    if not isinstance(nthreads, int):
        raise TypeError(f"Parameter 'nthreads' must be an integer, got {type(nthreads)}")
    
    if ncol <= 0:
        raise ValueError(f"Parameter 'ncol' must be positive, got {ncol}")
    if nrow <= 0:
        raise ValueError(f"Parameter 'nrow' must be positive, got {nrow}")
    if ngroups <= 0:
        raise ValueError(f"Parameter 'ngroups' must be positive, got {ngroups}")
    if nthreads <= 0:
        print("Warning: nthreads <= 0, using all available threads")

    # Array dimension validation
    if x.ndim != 1:
        raise ValueError(f"Parameter 'x' must be 1-dimensional, got {x.ndim}D")
    if p.ndim != 1:
        raise ValueError(f"Parameter 'p' must be 1-dimensional, got {p.ndim}D")
    if i.ndim != 1:
        raise ValueError(f"Parameter 'i' must be 1-dimensional, got {i.ndim}D")
    if groups.ndim != 1:
        raise ValueError(f"Parameter 'groups' must be 1-dimensional, got {groups.ndim}D")
    
    # Array length validation
    if len(x) != len(i):
        raise ValueError(f"Arrays 'x' and 'i' must have the same length, "
                        f"got x: {len(x)}, i: {len(i)}")
    
    if len(p) != nrow + 1:
        raise ValueError(f"Array 'p' must have length nrow + 1 = {nrow + 1}, "
                        f"got {len(p)}")
    
    if len(groups) != nrow:
        raise ValueError(f"Array 'groups' must have length nrow = {nrow}, "
                        f"got {len(groups)}")
    
    if len(x) == 0:
        raise ValueError("Array 'x' cannot be empty")
    
    # Data type validation for integer arrays
    if x.dtype != np.float64:
        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError(f"Input array must be floating-point, got {x.dtype}")
        x = x.astype(np.float64)

    # Check integer dtypes (allow uint64, int64, uint32, int32)
    if p.dtype not in integer_dtypes:
        raise TypeError(f"Array 'p' must have dtype in {integer_dtypes}, got {p.dtype}")
    if i.dtype not in integer_dtypes:
        raise TypeError(f"Array 'i' must have dtype in {integer_dtypes}, got {i.dtype}")
    if groups.dtype not in integer_dtypes:
        raise TypeError(f"Array 'groups' must have dtype in {integer_dtypes}, got {groups.dtype}")
    p, i, groups = p.astype(np.int32), i.astype(np.int32), groups.astype(np.int32)

    # Ensure x are positive
    if np.any(x <= 0):
        # Find the minimum value for better error message
        min_val = np.min(x)
        raise ValueError(f"All elements in 'x' must be positive (greater than 0), "
                         f"but found minimum value: {min_val}")
    
    # Value range validation
    if len(groups) > 0:
        max_group = np.max(groups)
        if max_group >= ngroups:
            raise ValueError(f"Group indices must be in range [0, {ngroups-1}], "
                             f"found maximum group index: {max_group}")
    if len(i) > 0:
        max_col = np.max(i)
        if max_col >= ncol:
            raise ValueError(f"Column indices in 'i' must be in range [0, {ncol-1}], "
                             f"found maximum column index: {max_col}")
    if len(p) > 0:
        if p[0] != 0:
            raise ValueError(f"First element of 'p' must be 0, got {p[0]}")
        
        if p[-1] != len(x):
            raise ValueError(f"Last element of 'p' must equal length of 'x' ({len(x)}), "
                           f"got {p[-1]}")
        # Check that p is non-decreasing
        if not np.all(np.diff(p) >= 0):
            raise ValueError("Array 'p' must be non-decreasing (valid CSR format)")
    
    # Call the C++ function
    try:
        result = mm.cpp_sumGroups_csr(x, p, i, ncol, nrow, groups, ngroups, nthreads)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    
def sumGroups_csc_T(
    x: np.ndarray,
    p: np.ndarray,
    i: np.ndarray,
    nrow: int,
    ncol: int,
    groups: np.ndarray,
    ngroups: int,
    nthreads: int = 1
) -> np.ndarray:
    """
    Sum groups for transposed CSC (Compressed Sparse Column) sparse matrix by row.
    
    This function performs group-wise summation of a sparse matrix stored in CSC format,
    but groups are applied to COLUMNS instead of rows. Each column belongs to a group,
    and the function sums all values within each group for each row.
    
    This is equivalent to summing groups on the transpose of the original matrix.
    
    Parameters
    ----------
    x : np.ndarray
        1D array of non-zero values in the sparse matrix (dtype: float64). 
        All values must be positive (greater than 0).
    p : np.ndarray
        1D array of column pointers indicating start/end indices for each column
        in arrays x and i (dtype: size_t/uint64). Length should be ncol + 1.
    i : np.ndarray
        1D array of row indices corresponding to each non-zero value in x
        (dtype: size_t/uint64).
    ncol : int
        Number of columns in the sparse matrix. Must be positive.
    nrow : int
        Number of rows in the sparse matrix. Must be positive.
    groups : np.ndarray
        1D array indicating which group each COLUMN belongs to (dtype: size_t/uint64).
        Values should be in range [0, ngroups-1]. Length should be ncol.
    ngroups : int
        Total number of groups. Must be positive.
    nthreads: int, optional
        Number of threads to use for computation. Must be positive. Default is 1.
        
    Returns
    -------
    np.ndarray
        2D array of shape (ngroups, nrow) containing the sum of values for each
        group and row combination (dtype: float64).
    """

    # Type validation
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Parameter 'x' must be a numpy array, got {type(x)}")
    if not isinstance(p, np.ndarray):
        raise TypeError(f"Parameter 'p' must be a numpy array, got {type(p)}")
    if not isinstance(i, np.ndarray):
        raise TypeError(f"Parameter 'i' must be a numpy array, got {type(i)}")
    if not isinstance(groups, np.ndarray):
        raise TypeError(f"Parameter 'groups' must be a numpy array, got {type(groups)}")
    
    # Scalar parameter validation
    if not isinstance(ncol, int):
        raise TypeError(f"Parameter 'ncol' must be an integer, got {type(ncol)}")
    if not isinstance(nrow, int):
        raise TypeError(f"Parameter 'nrow' must be an integer, got {type(nrow)}")
    if not isinstance(ngroups, int):
        raise TypeError(f"Parameter 'ngroups' must be an integer, got {type(ngroups)}")
    if not isinstance(nthreads, int):
        raise TypeError(f"Parameter 'nthreads' must be an integer, got {type(nthreads)}")
    
    if ncol <= 0:
        raise ValueError(f"Parameter 'ncol' must be positive, got {ncol}")
    if nrow <= 0:
        raise ValueError(f"Parameter 'nrow' must be positive, got {nrow}")
    if ngroups <= 0:
        raise ValueError(f"Parameter 'ngroups' must be positive, got {ngroups}")
    if nthreads <= 0:
        print("Warning: nthreads <= 0, using all available threads")
    
    # Array dimension validation
    if x.ndim != 1:
        raise ValueError(f"Parameter 'x' must be 1-dimensional, got {x.ndim}D")
    if p.ndim != 1:
        raise ValueError(f"Parameter 'p' must be 1-dimensional, got {p.ndim}D")
    if i.ndim != 1:
        raise ValueError(f"Parameter 'i' must be 1-dimensional, got {i.ndim}D")
    if groups.ndim != 1:
        raise ValueError(f"Parameter 'groups' must be 1-dimensional, got {groups.ndim}D")
    
    # Array length validation
    if len(x) != len(i):
        raise ValueError(f"Arrays 'x' and 'i' must have the same length, "
                        f"got x: {len(x)}, i: {len(i)}")
    
    if len(p) != ncol + 1:
        raise ValueError(f"Array 'p' must have length ncol + 1 = {ncol + 1}, "
                        f"got {len(p)}")
    
    if len(groups) != ncol:
        raise ValueError(f"Array 'groups' must have length ncol = {ncol}, "
                        f"got {len(groups)}")
    
    if len(x) == 0:
        raise ValueError("Array 'x' cannot be empty")
    
    # Data type validation for integer arrays
    if x.dtype != np.float64:
        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError(f"Input array must be floating-point, got {x.dtype}")
        x = x.astype(np.float64)

    # Check integer dtypes (allow uint64, int64, uint32, int32)
    if p.dtype not in integer_dtypes:
        raise TypeError(f"Array 'p' must have integer dtype (uint64/int64/uint32/int32), got {p.dtype}")
    if i.dtype not in integer_dtypes:
        raise TypeError(f"Array 'i' must have integer dtype (uint64/int64/uint32/int32), got {i.dtype}")
    if groups.dtype not in integer_dtypes:
        raise TypeError(f"Array 'groups' must have integer dtype (uint64/int64/uint32/int32), got {groups.dtype}")
    p, i, groups = p.astype(np.int32), i.astype(np.int32), groups.astype(np.int32)

    # Ensure all elements in x are positive (greater than 0)
    if np.any(x <= 0):
        # Find the minimum value for better error message
        min_val = np.min(x)
        raise ValueError(f"All elements in 'x' must be positive (greater than 0), "
                        f"but found minimum value: {min_val}")
    
    # Value range validation
    if len(groups) > 0:
        max_group = np.max(groups)
        if max_group >= ngroups:
            raise ValueError(f"Group indices must be in range [0, {ngroups-1}], "
                           f"found maximum group index: {max_group}")
    
    if len(i) > 0:
        max_row = np.max(i)
        if max_row >= nrow:
            raise ValueError(f"Row indices in 'i' must be in range [0, {nrow-1}], "
                           f"found maximum row index: {max_row}")
    
    # Validate CSC format constraints
    if len(p) > 0:
        if p[0] != 0:
            raise ValueError(f"First element of 'p' must be 0, got {p[0]}")
        
        if p[-1] != len(x):
            raise ValueError(f"Last element of 'p' must equal length of 'x' ({len(x)}), "
                           f"got {p[-1]}")
        
        # Check that p is non-decreasing
        if not np.all(np.diff(p) >= 0):
            raise ValueError("Array 'p' must be non-decreasing (valid CSC format)")
    
    # Call the C++ function
    try:
        result = mm.cpp_sumGroups_csc_T(x, p, i, ncol, nrow, groups, ngroups, nthreads)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    

def sumGroups_csr_T(
    x: np.ndarray,
    p: np.ndarray,
    i: np.ndarray,
    nrow: int,
    ncol: int,
    groups: np.ndarray,
    ngroups: int,
    nthreads: int = 1
) -> np.ndarray:
    """
    Sum groups for transposed CSR (Compressed Sparse Row) sparse matrix by rows.
    
    This function performs group-wise summation of a sparse matrix stored in CSR format,
    but groups are applied to COLUMNS instead of rows. Each column belongs to a group,
    and the function sums all values within each group for each row.
    
    This is equivalent to summing groups on the transpose of the original matrix.
    
    Parameters
    ----------
    x : np.ndarray
        1D array of non-zero values in the sparse matrix (dtype: float64). 
        All values must be positive (greater than 0).
    p : np.ndarray
        1D array of row pointers indicating start/end indices for each row
        in arrays x and i (dtype: size_t/uint64). Length should be nrow + 1.
    i : np.ndarray
        1D array of column indices corresponding to each non-zero value in x
        (dtype: size_t/uint64).
    ncol: int
        Number of columns in the sparse matrix. Must be positive.
        Just for input tests, not used for calc.
    nrow : int
        Number of rows in the sparse matrix. Must be positive.
    groups : np.ndarray
        1D array indicating which group each COLUMN belongs to (dtype: size_t/uint64).
        Values should be in range [0, ngroups-1]. Length should match the total number of columns.
    ngroups : int
        Total number of groups. Must be positive.
    nthreads: int, optional
        Number of threads to use for computation. Must be positive. Default is 1.
        
    Returns
    -------
    np.ndarray
        2D array of shape (ngroups, nrow) containing the sum of values for each
        group and row combination (dtype: float64).
    """

    # Type validation
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Parameter 'x' must be a numpy array, got {type(x)}")
    if not isinstance(p, np.ndarray):
        raise TypeError(f"Parameter 'p' must be a numpy array, got {type(p)}")
    if not isinstance(i, np.ndarray):
        raise TypeError(f"Parameter 'i' must be a numpy array, got {type(i)}")
    if not isinstance(groups, np.ndarray):
        raise TypeError(f"Parameter 'groups' must be a numpy array, got {type(groups)}")
    
    # Scalar parameter validation
    if not isinstance(ncol, int):
        raise TypeError(f"Parameter 'ncol' must be an integer, got {type(ncol)}")
    if not isinstance(nrow, int):
        raise TypeError(f"Parameter 'nrow' must be an integer, got {type(nrow)}")
    if not isinstance(ngroups, int):
        raise TypeError(f"Parameter 'ngroups' must be an integer, got {type(ngroups)}")
    if not isinstance(nthreads, int):
        raise TypeError(f"Parameter 'nthreads' must be an integer, got {type(nthreads)}")
    
    if ncol <= 0:
        raise ValueError(f"Parameter 'ncol' must be positive, got {ncol}")
    if nrow <= 0:
        raise ValueError(f"Parameter 'nrow' must be positive, got {nrow}")
    if ngroups <= 0:
        raise ValueError(f"Parameter 'ngroups' must be positive, got {ngroups}")
    if nthreads <= 0:
        print("Warning: nthreads <= 0, using all available threads")
    
    # Array dimension validation
    if x.ndim != 1:
        raise ValueError(f"Parameter 'x' must be 1-dimensional, got {x.ndim}D")
    if p.ndim != 1:
        raise ValueError(f"Parameter 'p' must be 1-dimensional, got {p.ndim}D")
    if i.ndim != 1:
        raise ValueError(f"Parameter 'i' must be 1-dimensional, got {i.ndim}D")
    if groups.ndim != 1:
        raise ValueError(f"Parameter 'groups' must be 1-dimensional, got {groups.ndim}D")
    
    # Array length validation
    if len(x) != len(i):
        raise ValueError(f"Arrays 'x' and 'i' must have the same length, "
                        f"got x: {len(x)}, i: {len(i)}")
    if len(p) != nrow + 1:
        raise ValueError(f"Array 'p' must have length nrow + 1 = {nrow + 1}, "
                        f"got {len(p)}")
    if len(x) == 0:
        raise ValueError("Array 'x' cannot be empty")
    
    # Data type validation for integer arrays
    if x.dtype != np.float64:
        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError(f"Input array must be floating-point, got {x.dtype}")
        x = x.astype(np.float64)

    # Check integer dtypes (allow uint64, int64, uint32, int32)
    if p.dtype not in integer_dtypes:
        raise TypeError(f"Array 'p' must have integer dtype (uint64/int64/uint32/int32), got {p.dtype}")
    if i.dtype not in integer_dtypes:
        raise TypeError(f"Array 'i' must have integer dtype (uint64/int64/uint32/int32), got {i.dtype}")
    if groups.dtype not in integer_dtypes:
        raise TypeError(f"Array 'groups' must have integer dtype (uint64/int64/uint32/int32), got {groups.dtype}")
    p, i, groups = p.astype(np.int32), i.astype(np.int32), groups.astype(np.int32)

    # Ensure all elements in x are positive (greater than 0)
    if np.any(x <= 0):
        # Find the minimum value for better error message
        min_val = np.min(x)
        raise ValueError(f"All elements in 'x' must be positive (greater than 0), "
                        f"but found minimum value: {min_val}")
    
    # Value range validation
    if len(groups) != ncol:
        raise ValueError(f"Array 'groups' must have length equal to number of columns ({ncol}), "
                        f"got {len(groups)}")
    if len(groups) > 0:
        max_groups = np.max(groups)
        if max_groups >= ngroups:
            raise ValueError(f"Group indices must be in range [0, {ngroups-1}], "
                           f"found maximum group index: {max_groups}")
    if len(i) > 0:
        max_col = np.max(i)
        if max_col >= ncol:
            raise ValueError(f"Column indices in 'i' must be in range [0, {ncol-1}], "
                           f"found maximum column index: {max_col}")
    
    # Validate CSR format constraints
    if len(p) > 0:
        if p[0] != 0:
            raise ValueError(f"First element of 'p' must be 0, got {p[0]}")
        
        if p[-1] != len(x):
            raise ValueError(f"Last element of 'p' must equal length of 'x' ({len(x)}), "
                           f"got {p[-1]}")
        
        # Check that p is non-decreasing
        if not np.all(np.diff(p) >= 0):
            raise ValueError("Array 'p' must be non-decreasing (valid CSR format)")
    
    # Call the C++ function
    try:
        result = mm.cpp_sumGroups_csr_T(x, p, i, nrow, groups, ngroups, nthreads)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    
def sumGroups_dense(
    x: np.ndarray,
    groups: np.ndarray,
    ngroups: int
) -> np.ndarray:
    """
    Sum groups for dense matrix by columns.
    
    This function performs group-wise summation of a dense matrix.
    Each row belongs to a group, and the function sums all values within each group
    for each column.
    
    Parameters
    ----------
    x : np.ndarray
        2D dense matrix (dtype: float64). All values must be non-negative.
    groups : np.ndarray
        1D array indicating which group each row belongs to (dtype: size_t/uint64).
        Values should be in range [0, ngroups-1]. Length should equal number of rows in x.
    ngroups : int
        Total number of groups. Must be positive.
        
    Returns
    -------
    np.ndarray
        2D array of shape (ngroups, ncols) containing the sum of values for each
        group and column combination (dtype: float64).
    """
    # Type validation
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Parameter 'x' must be a numpy array, got {type(x)}")
    if not isinstance(groups, np.ndarray):
        raise TypeError(f"Parameter 'groups' must be a numpy array, got {type(groups)}")
    
    # calar parameter validation
    if not isinstance(ngroups, int):
        raise TypeError(f"Parameter 'ngroups' must be an integer, got {type(ngroups)}")
    if ngroups <= 0:
        raise ValueError(f"Parameter 'ngroups' must be positive, got {ngroups}")
    
    # Array dimension validation
    if x.ndim != 2:
        raise ValueError(f"Parameter 'x' must be 2-dimensional, got {x.ndim}D")
    if groups.ndim != 1:
        raise ValueError(f"Parameter 'groups' must be 1-dimensional, got {groups.ndim}D")
    
    nrows, ncols = x.shape
    if nrows == 0 or ncols == 0:
        raise ValueError("Input matrix 'x' cannot be empty")
    
    # Array length validation
    if len(groups) != nrows:
        raise ValueError(f"Array 'groups' must have length equal to number of rows ({nrows}), "
                        f"got {len(groups)}")

    # Data type validation
    if x.dtype != np.float64:
        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError(f"Input array must be floating-point, got {x.dtype}")
        x = x.astype(np.float64)
        
    # Check integer dtypes for groups (allow uint64, int64, uint32, int32)
    if groups.dtype not in integer_dtypes:
        raise TypeError(f"Array 'groups' must have integer dtype (uint64/int64/uint32/int32), got {groups.dtype}")
    groups = groups.astype(np.int32)

    # Ensure all elements in x are non-negative
    if np.any(x < 0):
        # Find the minimum value for better error message
        min_val = np.min(x)
        raise ValueError(f"All elements in 'x' must be non-negative, "
                        f"but found minimum value: {min_val}")
    
    # Value range validation
    if len(groups) > 0:
        max_group = np.max(groups)
        if max_group >= ngroups:
            raise ValueError(f"Group indices must be in range [0, {ngroups-1}], "
                           f"found maximum group index: {max_group}")
        
    # Call the C++ function
    try:
        result = mm.cpp_sumGroups_dense(x, groups, ngroups)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    

def sumGroups_dense_T(
    x: np.ndarray,
    groups: np.ndarray,
    ngroups: int
) -> np.ndarray:
    """
    Sum groups for transposed dense matrix by rows.
    
    This function performs group-wise summation of a dense matrix,
    but groups are applied to COLUMNS instead of rows. Each column belongs to a group,
    and the function sums all values within each group for each row.
    
    This is equivalent to summing groups on the transpose of the original matrix.
    
    Parameters
    ----------
    x : np.ndarray
        2D dense matrix (dtype: float64). All values must be positive (greater than 0).
    groups : np.ndarray
        1D array indicating which group each COLUMN belongs to (dtype: size_t/uint64).
        Values should be in range [0, ngroups-1]. Length should equal number of columns in x.
    ngroups : int
        Total number of groups. Must be positive.
        
    Returns
    -------
    np.ndarray
        2D array of shape (ngroups, nrows) containing the sum of values for each
        group and row combination (dtype: float64).
    """

    # Type validation
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Parameter 'x' must be a numpy array, got {type(x)}")
    if not isinstance(groups, np.ndarray):
        raise TypeError(f"Parameter 'groups' must be a numpy array, got {type(groups)}")
    
    # Scalar parameter validation
    if not isinstance(ngroups, int):
        raise TypeError(f"Parameter 'ngroups' must be an integer, got {type(ngroups)}")
    if ngroups <= 0:
        raise ValueError(f"Parameter 'ngroups' must be positive, got {ngroups}")
    
    # Array dimension validation
    if x.ndim != 2:
        raise ValueError(f"Parameter 'x' must be 2-dimensional, got {x.ndim}D")
    if groups.ndim != 1:
        raise ValueError(f"Parameter 'groups' must be 1-dimensional, got {groups.ndim}D")
    
    nrows, ncols = x.shape
    if nrows == 0 or ncols == 0:
        raise ValueError("Input matrix 'x' cannot be empty")
    
    # Array length validation
    if len(groups) != ncols:
        raise ValueError(f"Array 'groups' must have length equal to number of columns ({ncols}), "
                        f"got {len(groups)}")
    
    # Data type validation
    if x.dtype != np.float64:
        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError(f"Input array must be floating-point, got {x.dtype}")
        x = x.astype(np.float64)

    # Check integer dtypes for groups (allow uint64, int64, uint32, int32)
    if groups.dtype not in integer_dtypes:
        raise TypeError(f"Array 'groups' must have integer dtype (uint64/int64/uint32/int32), got {groups.dtype}")
    groups = groups.astype(np.int32)

    # Ensure all elements in x are non-negative
    if np.any(x < 0):
        # Find the minimum value for better error message
        min_val = np.min(x)
        raise ValueError(f"All elements in 'x' must be non-negative, "
                        f"but found minimum value: {min_val}")
    
    # Value range validation
    if len(groups) > 0:
        max_group = np.max(groups)
        if max_group >= ngroups:
            raise ValueError(f"Group indices must be in range [0, {ngroups-1}], "
                           f"found maximum group index: {max_group}")
        
    # Call the C++ function
    try:
        result = mm.cpp_sumGroups_dense_T(x, groups, ngroups)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    
def nnzeroGroups_csc(
    p: np.ndarray,
    i: np.ndarray,
    nrow: int,
    ncol: int,
    groups: np.ndarray,
    ngroups: int,
    nthreads: int = 1
) -> np.ndarray:
    """
    Count non-zero elements for CSC (Compressed Sparse Column) sparse matrix by groups.
    
    This function counts the number of non-zero elements in a sparse matrix stored in CSC format.
    Each row belongs to a group, and the function counts non-zero elements within each group.
    for each column.
    
    Parameters
    ----------
    p : np.ndarray
        1D array of column pointers indicating start/end indices for each column
        in array i (dtype: size_t/uint64). Length should be ncol + 1.
    i : np.ndarray
        1D array of row indices corresponding to each non-zero element
        (dtype: size_t/uint64).
    ncol : int
        Number of columns in the sparse matrix. Must be positive.
    nrow : int
        Number of rows in the sparse matrix. Must be positive.
        Just for input tests, not used for calc.
    groups : np.ndarray
        1D array indicating which group each row belongs to (dtype: size_t/uint64).
        Values should be in range [0, ngroups-1].
    ngroups : int
        Total number of groups. Must be positive.
    nthreads: int, optional
        Number of threads to use for computation. Must be positive. Default is 1.
        
    Returns
    -------
    np.ndarray
        2D array of shape (ngroups, ncol) containing the count of non-zero elements for each
        group and column combination (dtype: int32).
    """

    # Type validation
    if not isinstance(p, np.ndarray):
        raise TypeError(f"Parameter 'p' must be a numpy array, got {type(p)}")
    if not isinstance(i, np.ndarray):
        raise TypeError(f"Parameter 'i' must be a numpy array, got {type(i)}")
    if not isinstance(groups, np.ndarray):
        raise TypeError(f"Parameter 'groups' must be a numpy array, got {type(groups)}")
    
    # Scalar parameter validation
    if not isinstance(ncol, int):
        raise TypeError(f"Parameter 'ncol' must be an integer, got {type(ncol)}")
    if not isinstance(nrow, int):
        raise TypeError(f"Parameter 'nrow' must be an integer, got {type(nrow)}")
    if not isinstance(ngroups, int):
        raise TypeError(f"Parameter 'ngroups' must be an integer, got {type(ngroups)}")
    if not isinstance(nthreads, int):
        raise TypeError(f"Parameter 'nthreads' must be an integer, got {type(nthreads)}")
    
    if ncol <= 0:
        raise ValueError(f"Parameter 'ncol' must be positive, got {ncol}")
    if nrow <= 0:
        raise ValueError(f"Parameter 'nrow' must be positive, got {nrow}")
    if ngroups <= 0:
        raise ValueError(f"Parameter 'ngroups' must be positive, got {ngroups}")
    if nthreads <= 0:
        print("Warning: nthreads <= 0, using all available threads")
    
    # Array dimension validation
    if p.ndim != 1:
        raise ValueError(f"Parameter 'p' must be 1-dimensional, got {p.ndim}D")
    if i.ndim != 1:
        raise ValueError(f"Parameter 'i' must be 1-dimensional, got {i.ndim}D")
    if groups.ndim != 1:
        raise ValueError(f"Parameter 'groups' must be 1-dimensional, got {groups.ndim}D")
    
    # Array length validation
    if len(p) != ncol + 1:
        raise ValueError(f"Array 'p' must have length ncol + 1 = {ncol + 1}, "
                        f"got {len(p)}")
    if len(groups) != nrow:
        raise ValueError(f"Array 'groups' must have length equal to number of rows ({nrow}), "
                         f"got {len(groups)}")
    # Get the number of non-zero elements
    nnz = p[-1] if len(p) > 0 else 0
    if len(i) != nnz:
        raise ValueError(f"Array 'i' must have length equal to number of non-zero elements ({nnz}), "
                        f"got {len(i)}")
    
    # Data type validation for integer arrays
    if p.dtype not in integer_dtypes:
        raise TypeError(f"Array 'p' must have integer dtype (uint64/int64/uint32/int32), got {p.dtype}")
    if i.dtype not in integer_dtypes:
        raise TypeError(f"Array 'i' must have integer dtype (uint64/int64/uint32/int32), got {i.dtype}")
    if groups.dtype not in integer_dtypes:
        raise TypeError(f"Array 'groups' must have integer dtype (uint64/int64/uint32/int32), got {groups.dtype}")
    p, i, groups = p.astype(np.int32), i.astype(np.int32), groups.astype(np.int32)

    # Value range validation
    if len(groups) > 0:
        max_group = np.max(groups)
        if max_group >= ngroups:
            raise ValueError(f"Group indices must be in range [0, {ngroups-1}], "
                           f"found maximum group index: {max_group}")
    if len(i) > 0:
        max_row = np.max(i)
        if max_row >= len(groups):
            raise ValueError(f"Row indices in 'i' must be in range [0, {len(groups)-1}], "
                           f"found maximum row index: {max_row}")
    
    # Validate CSC format constraints
    if len(p) > 0:
        if p[0] != 0:
            raise ValueError(f"First element of 'p' must be 0, got {p[0]}")
        
        # Check that p is non-decreasing
        if not np.all(np.diff(p) >= 0):
            raise ValueError("Array 'p' must be non-decreasing (valid CSC format)")

    # Call the C++ function
    try:
        result = mm.cpp_nnzeroGroups_csc(p, i, ncol, groups, ngroups, nthreads)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    
def nnzeroGroups_csr(
    p: np.ndarray,
    i: np.ndarray,
    nrow: int,
    ncol: int,
    groups: np.ndarray,
    ngroups: int,
    nthreads: int = 1
) -> np.ndarray:
    """
    Count non-zero elements for CSR (Compressed Sparse Row) sparse matrix by groups.
    
    This function counts the number of non-zero elements in a sparse matrix stored in CSR format.
    Each row belongs to a group, and the function counts non-zero elements within each group
    for each column.
    
    Parameters
    ----------
    p : np.ndarray
        1D array of row pointers indicating start/end indices for each row
        in array i (dtype: size_t/uint64). Length should be nrow + 1.
    i : np.ndarray
        1D array of column indices corresponding to each non-zero element
        (dtype: size_t/uint64).
    ncol : int
        Number of columns in the sparse matrix. Must be positive.
    nrow : int
        Number of rows in the sparse matrix. Must be positive.
    groups : np.ndarray
        1D array indicating which group each row belongs to (dtype: size_t/uint64).
        Values should be in range [0, ngroups-1].
    ngroups : int
        Total number of groups. Must be positive.
    nthreads: int, optional
        Number of threads to use for computation. Must be positive. Default is 1.
    """

    # Type validation
    if not isinstance(p, np.ndarray):
        raise TypeError(f"Parameter 'p' must be a numpy array, got {type(p)}")
    if not isinstance(i, np.ndarray):
        raise TypeError(f"Parameter 'i' must be a numpy array, got {type(i)}")
    if not isinstance(groups, np.ndarray):
        raise TypeError(f"Parameter 'groups' must be a numpy array, got {type(groups)}")
    
    # Scalar parameter validation
    if not isinstance(ncol, int):
        raise TypeError(f"Parameter 'ncol' must be an integer, got {type(ncol)}")
    if not isinstance(nrow, int):
        raise TypeError(f"Parameter 'nrow' must be an integer, got {type(nrow)}")
    if not isinstance(ngroups, int):
        raise TypeError(f"Parameter 'ngroups' must be an integer, got {type(ngroups)}")
    if not isinstance(nthreads, int):
        raise TypeError(f"Parameter 'nthreads' must be an integer, got {type(nthreads)}")
    
    # Scalar value validation
    if ncol <= 0:
        raise ValueError(f"Parameter 'ncol' must be positive, got {ncol}")
    if nrow <= 0:
        raise ValueError(f"Parameter 'nrow' must be positive, got {nrow}")
    if ngroups <= 0:
        raise ValueError(f"Parameter 'ngroups' must be positive, got {ngroups}")
    if nthreads <= 0:
        print("Warning: nthreads <= 0, using all available threads")

    # Array dimension validation
    if p.ndim != 1:
        raise ValueError(f"Parameter 'p' must be 1-dimensional, got {p.ndim}D")
    if i.ndim != 1:
        raise ValueError(f"Parameter 'i' must be 1-dimensional, got {i.ndim}D")
    if groups.ndim != 1:
        raise ValueError(f"Parameter 'groups' must be 1-dimensional, got {groups.ndim}D")
    
    # Array length validation
    if len(p) != nrow + 1:
        raise ValueError(f"Array 'p' must have length nrow + 1 = {nrow + 1}, "
                        f"got {len(p)}")
    if len(groups) != nrow:
        raise ValueError(f"Array 'groups' must have length nrow = {nrow}, "
                        f"got {len(groups)}")
    # Get the number of non-zero elements
    nnz = p[-1] if len(p) > 0 else 0
    if len(i) != nnz:
        raise ValueError(f"Array 'i' must have length equal to number of non-zero elements ({nnz}), "
                        f"got {len(i)}")
    
    # Data type validation for integer arrays
    if p.dtype not in integer_dtypes:
        raise TypeError(f"Array 'p' must have dtype in {integer_dtypes}, got {p.dtype}")
    if i.dtype not in integer_dtypes:
        raise TypeError(f"Array 'i' must have dtype in {integer_dtypes}, got {i.dtype}")
    if groups.dtype not in integer_dtypes:
        raise TypeError(f"Array 'groups' must have dtype in {integer_dtypes}, got {groups.dtype}")
    p, i, groups = p.astype(np.int32), i.astype(np.int32), groups.astype(np.int32)

    # Value range validation
    if len(groups) > 0:
        max_group = np.max(groups)
        if max_group >= ngroups:
            raise ValueError(f"Group indices must be in range [0, {ngroups-1}], "
                             f"found maximum group index: {max_group}")
    if len(i) > 0:
        max_col = np.max(i)
        if max_col >= ncol:
            raise ValueError(f"Column indices in 'i' must be in range [0, {ncol-1}], "
                             f"found maximum column index: {max_col}")
    if len(p) > 0:
        if p[0] != 0:
            raise ValueError(f"First element of 'p' must be 0, got {p[0]}")
        # Check that p is non-decreasing
        if not np.all(np.diff(p) >= 0):
            raise ValueError("Array 'p' must be non-decreasing (valid CSR format)")
    
    # Call the C++ function
    try:
        result = mm.cpp_nnzeroGroups_csr(p, i, ncol, nrow, groups, ngroups, nthreads)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    
def nnzeroGroups_csc_T(
    p: np.ndarray,
    i: np.ndarray,
    nrow: int,
    ncol: int,
    groups: np.ndarray,
    ngroups: int,
    nthreads: int = 1
) -> np.ndarray:
    """
    Count non-zero elements for transposed CSC (Compressed Sparse Column) sparse matrix by groups.
    
    This function counts the number of non-zero elements in a sparse matrix stored in CSC format,
    but groups are applied to COLUMNS instead of rows. Each column belongs to a group,
    and the function counts non-zero elements within each group for each row.
    
    This is equivalent to counting non-zeros on the transpose of the original matrix.
    
    Parameters
    ----------
    p : np.ndarray
        1D array of column pointers indicating start/end indices for each column
        in array i (dtype: size_t/uint64). Length should be ncol + 1.
    i : np.ndarray
        1D array of row indices corresponding to each non-zero element
        (dtype: size_t/uint64).
    ncol : int
        Number of columns in the sparse matrix. Must be positive.
    nrow : int
        Number of rows in the sparse matrix. Must be positive.
    groups : np.ndarray
        1D array indicating which group each COLUMN belongs to (dtype: size_t/uint64).
        Values should be in range [0, ngroups-1]. Length should be ncol.
    ngroups : int
        Total number of groups. Must be positive.
    nthreads: int, optional
        Number of threads to use for computation. Must be positive. Default is 1.
        
    Returns
    -------
    np.ndarray
        2D array of shape (ngroups, nrow) containing the count of non-zero elements for each
        group and row combination (dtype: int32).
    """
    # Type validation
    if not isinstance(p, np.ndarray):
        raise TypeError(f"Parameter 'p' must be a numpy array, got {type(p)}")
    if not isinstance(i, np.ndarray):
        raise TypeError(f"Parameter 'i' must be a numpy array, got {type(i)}")
    if not isinstance(groups, np.ndarray):
        raise TypeError(f"Parameter 'groups' must be a numpy array, got {type(groups)}")
    
    # Scalar parameter validation
    if not isinstance(ncol, int):
        raise TypeError(f"Parameter 'ncol' must be an integer, got {type(ncol)}")
    if not isinstance(nrow, int):
        raise TypeError(f"Parameter 'nrow' must be an integer, got {type(nrow)}")
    if not isinstance(ngroups, int):
        raise TypeError(f"Parameter 'ngroups' must be an integer, got {type(ngroups)}")
    if not isinstance(nthreads, int):
        raise TypeError(f"Parameter 'nthreads' must be an integer, got {type(nthreads)}")
    
    if ncol <= 0:
        raise ValueError(f"Parameter 'ncol' must be positive, got {ncol}")
    if nrow <= 0:
        raise ValueError(f"Parameter 'nrow' must be positive, got {nrow}")
    if ngroups <= 0:
        raise ValueError(f"Parameter 'ngroups' must be positive, got {ngroups}")
    if nthreads <= 0:
        print("Warning: nthreads <= 0, using all available threads")
    
    # Array dimension validation
    if p.ndim != 1:
        raise ValueError(f"Parameter 'p' must be 1-dimensional, got {p.ndim}D")
    if i.ndim != 1:
        raise ValueError(f"Parameter 'i' must be 1-dimensional, got {i.ndim}D")
    if groups.ndim != 1:
        raise ValueError(f"Parameter 'groups' must be 1-dimensional, got {groups.ndim}D")
    
    # Array length validation
    if len(p) != ncol + 1:
        raise ValueError(f"Array 'p' must have length ncol + 1 = {ncol + 1}, "
                        f"got {len(p)}")
    # Get the number of non-zero elements
    nnz = p[-1] if len(p) > 0 else 0
    if len(i) != nnz:
        raise ValueError(f"Array 'i' must have length equal to number of non-zero elements ({nnz}), "
                        f"got {len(i)}")
    if len(groups) != ncol:
        raise ValueError(f"Array 'groups' must have length equal to number of columns ({ncol}), "
                        f"got {len(groups)}")
    
    # Data type validation for integer arrays
    if p.dtype not in integer_dtypes:
        raise TypeError(f"Array 'p' must have integer dtype (uint64/int64/uint32/int32), got {p.dtype}")
    if i.dtype not in integer_dtypes:
        raise TypeError(f"Array 'i' must have integer dtype (uint64/int64/uint32/int32), got {i.dtype}")
    if groups.dtype not in integer_dtypes:
        raise TypeError(f"Array 'groups' must have integer dtype (uint64/int64/uint32/int32), got {groups.dtype}")
    p, i, groups = p.astype(np.int32), i.astype(np.int32), groups.astype(np.int32)

    # Value range validation
    if len(groups) > 0:
        max_group = np.max(groups)
        if max_group >= ngroups:
            raise ValueError(f"Group indices must be in range [0, {ngroups-1}], "
                           f"found maximum group index: {max_group}")
    if len(i) > 0:
        max_row = np.max(i)
        if max_row >= nrow:
            raise ValueError(f"Row indices in 'i' must be in range [0, {nrow-1}], "
                           f"found maximum row index: {max_row}")
    
    # Validate CSC format constraints
    if len(p) > 0:
        if p[0] != 0:
            raise ValueError(f"First element of 'p' must be 0, got {p[0]}")
        if not np.all(np.diff(p) >= 0):
            raise ValueError("Array 'p' must be non-decreasing (valid CSC format)")
        
    # Call the C++ function
    try:
        result = mm.cpp_nnzeroGroups_csc_T(p, i, ncol, nrow, groups, ngroups, nthreads)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")

def nnzeroGroups_csr_T(
    p: np.ndarray,
    i: np.ndarray,
    nrow: int,
    ncol: int,
    groups: np.ndarray,
    ngroups: int,
    nthreads: int = 1
) -> np.ndarray:
    """
    Count non-zero elements for transposed CSR (Compressed Sparse Row) sparse matrix by groups.
    
    This function counts the number of non-zero elements in a sparse matrix stored in CSR format,
    but groups are applied to COLUMNS instead of rows. Each column belongs to a group,
    and the function counts non-zero elements within each group for each row.
    
    This is equivalent to counting non-zeros on the transpose of the original matrix.
    
    Parameters
    ----------
    p : np.ndarray
        1D array of row pointers indicating start/end indices for each row
        in array i (dtype: size_t/uint64). Length should be nrow + 1.
    i : np.ndarray
        1D array of column indices corresponding to each non-zero element
        (dtype: size_t/uint64).
    ncol : int
        Number of columns in the sparse matrix. Must be positive.
        Used for validation only (not passed to C++ function).
    nrow : int
        Number of rows in the sparse matrix. Must be positive.
    groups : np.ndarray
        1D array indicating which group each COLUMN belongs to (dtype: size_t/uint64).
        Values should be in range [0, ngroups-1]. Length should match the actual number of columns.
    ngroups : int
        Total number of groups. Must be positive.
    nthreads: int, optional
        Number of threads to use for computation. Must be positive. Default is 1.
        
    Returns
    -------
    np.ndarray
        2D array of shape (ngroups, nrow) containing the count of non-zero elements for each
        group and row combination (dtype: int32).
    """

    # Type validation
    if not isinstance(p, np.ndarray):
        raise TypeError(f"Parameter 'p' must be a numpy array, got {type(p)}")
    if not isinstance(i, np.ndarray):
        raise TypeError(f"Parameter 'i' must be a numpy array, got {type(i)}")
    if not isinstance(groups, np.ndarray):
        raise TypeError(f"Parameter 'groups' must be a numpy array, got {type(groups)}")
    
    # Scalar parameter validation
    if not isinstance(ncol, int):
        raise TypeError(f"Parameter 'ncol' must be an integer, got {type(ncol)}")
    if not isinstance(nrow, int):
        raise TypeError(f"Parameter 'nrow' must be an integer, got {type(nrow)}")
    if not isinstance(ngroups, int):
        raise TypeError(f"Parameter 'ngroups' must be an integer, got {type(ngroups)}")
    if not isinstance(nthreads, int):
        raise TypeError(f"Parameter 'nthreads' must be an integer, got {type(nthreads)}")
    
    if ncol <= 0:
        raise ValueError(f"Parameter 'ncol' must be positive, got {ncol}")
    if nrow <= 0:
        raise ValueError(f"Parameter 'nrow' must be positive, got {nrow}")
    if ngroups <= 0:
        raise ValueError(f"Parameter 'ngroups' must be positive, got {ngroups}")
    if nthreads <= 0:
        print("Warning: nthreads <= 0, using all available threads")

    # Array dimension validation
    if p.ndim != 1:
        raise ValueError(f"Parameter 'p' must be 1-dimensional, got {p.ndim}D")
    if i.ndim != 1:
        raise ValueError(f"Parameter 'i' must be 1-dimensional, got {i.ndim}D")
    if groups.ndim != 1:
        raise ValueError(f"Parameter 'groups' must be 1-dimensional, got {groups.ndim}D")
    
    # Array length validation
    nnz = p[-1] if len(p) > 0 else 0
    if len(i) != nnz:
        raise ValueError(f"Array 'i' must have length equal to number of non-zero elements ({nnz}), "
                        f"got {len(i)}")
    if len(p) != nrow + 1:
        raise ValueError(f"Array 'p' must have length nrow + 1 = {nrow + 1}, "
                        f"got {len(p)}")
    if len(groups) != ncol:
        raise ValueError(f"Array 'groups' must have length equal to number of columns ({ncol}), "
                        f"got {len(groups)}")
    
    # Data type validation for integer arrays
    if p.dtype not in integer_dtypes:
        raise TypeError(f"Array 'p' must have integer dtype (uint64/int64/uint32/int32), got {p.dtype}")
    if i.dtype not in integer_dtypes:
        raise TypeError(f"Array 'i' must have integer dtype (uint64/int64/uint32/int32), got {i.dtype}")
    if groups.dtype not in integer_dtypes:
        raise TypeError(f"Array 'groups' must have integer dtype (uint64/int64/uint32/int32), got {groups.dtype}")
    p, i, groups = p.astype(np.int32), i.astype(np.int32), groups.astype(np.int32)

    # Value range validation
    if len(groups) > 0:
        max_group = np.max(groups)
        if max_group >= ngroups:
            raise ValueError(f"Group indices must be in range [0, {ngroups-1}], "
                           f"found maximum group index: {max_group}")
    if len(i) > 0:
        max_col = np.max(i)
        if max_col >= ncol:
            raise ValueError(f"Row indices in 'i' must be in range [0, {ncol-1}], "
                           f"found maximum row index: {max_col}")
        
    # Validate CSR format constraints
    if len(p) > 0:
        if p[0] != 0:
            raise ValueError(f"First element of 'p' must be 0, got {p[0]}")
        if not np.all(np.diff(p) >= 0):
            raise ValueError("Array 'p' must be non-decreasing (valid CSR format)")
        
    # Call the C++ function
    try:
        result = mm.cpp_nnzeroGroups_csr_T(p, i, ncol, nrow, groups, ngroups, nthreads)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    
def nnzeroGroups_dense(
    x: np.ndarray,
    groups: np.ndarray,
    ngroups: int,
    nthreads: int = 1
) -> np.ndarray:
    
    """
    Count non-zero elements in a dense matrix by row groups, per column.

    This function counts, for each group and each column, how many rows in that group
    have a non-zero entry in the given column.

    Parameters
    ----------
    x : np.ndarray
        2D dense input matrix (dtype: float64). Shape: (nrows, ncols).
    groups : np.ndarray
        1D array assigning each row to a group (dtype: any integer type).
        Length must be equal to number of rows in `x`.
        Group indices must be in range [0, ngroups - 1].
    ngroups : int
        Total number of groups. Must be positive.
    nthreads : int, optional
        Number of OpenMP threads to use. If <= 0, uses all available threads.
        Default is 1.

    Returns
    -------
    np.ndarray
        2D integer array of shape (ngroups, ncols), where result[g, c]
        is the count of non-zero entries in column `c` among rows assigned to group `g`.
    """

    # --- Type validation ---
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Parameter 'x' must be a numpy array, got {type(x)}")
    if not isinstance(groups, np.ndarray):
        raise TypeError(f"Parameter 'groups' must be a numpy array, got {type(groups)}")
    if not isinstance(ngroups, int):
        raise TypeError(f"Parameter 'ngroups' must be an integer, got {type(ngroups)}")
    if not isinstance(nthreads, int):
        raise TypeError(f"Parameter 'nthreads' must be an integer, got {type(nthreads)}")
    
    # --- Scalar validation ---
    if ngroups <= 0:
        raise ValueError(f"Parameter 'ngroups' must be positive, got {ngroups}")
    if nthreads <= 0:
        print("Warning: nthreads <= 0, using all available threads")

    # --- Array dimension validation ---
    if x.ndim != 2:
        raise ValueError(f"Parameter 'x' must be 2-dimensional, got {x.ndim}D")
    if groups.ndim != 1:
        raise ValueError(f"Parameter 'groups' must be 1-dimensional, got {groups.ndim}D")
    
    nrows, ncols = x.shape
    if nrows == 0 or ncols == 0:
        raise ValueError(f"Input matrix 'x' must have at least one row and one column, got shape {x.shape}")

    if len(groups) != nrows:
        raise ValueError(f"Length of 'groups' ({len(groups)}) must equal number of rows in 'x' ({nrows})")
    
    # --- Data type validation ---
    if x.dtype != np.float64:
        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError(f"Input array must be floating-point, got {x.dtype}")
        x = x.astype(np.float64)

    if groups.dtype not in integer_dtypes:
        raise TypeError(f"Array 'groups' must have an integer dtype in {integer_dtypes}, got {groups.dtype}")
    groups = groups.astype(np.int32)

    # --- Ensure all elements in x are non-negative ---
    if np.any(x < 0):
        min_val = np.min(x)
        raise ValueError(f"All elements in 'x' must be non-negative, "
                         f"but found minimum value: {min_val}")
    # Value range validation
    if len(groups) > 0:
        max_group = np.max(groups)
        if max_group >= ngroups:
            raise ValueError(f"Group indices in 'groups' must be in range [0, {ngroups-1}], "
                             f"found maximum group index: {max_group}")
    # Call the C++ function
    try:
        result = mm.cpp_nnzeroGroups_dense(x, groups, ngroups, nthreads)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    
def nnzeroGroups_dense_T(
    x: np.ndarray,
    groups: np.ndarray,
    ngroups: int,
    nthreads: int = 1
) -> np.ndarray:
    """
    Count non-zero elements in a dense matrix by column groups, per row.

    This function transposes the logic of `nnzeroGroups_dense`: each *column* belongs to a group,
    and for each group and each *row*, it counts how many columns in that group have a non-zero
    entry in the given row.

    Parameters
    ----------
    x : np.ndarray
        2D dense input matrix (dtype: float64). Shape: (nrows, ncols).
    groups : np.ndarray
        1D array assigning each *column* to a group (dtype: any integer type).
        Length must be equal to number of columns in `x`.
        Group indices must be in range [0, ngroups - 1].
    ngroups : int
        Total number of groups. Must be positive.
    nthreads : int, optional
        Number of OpenMP threads to use. If <= 0, uses all available threads.
        Default is 1.

    Returns
    -------
    np.ndarray
        2D integer array of shape (ngroups, nrows), where result[g, r]
        is the count of non-zero entries in row `r` among columns assigned to group `g`.
    """

    # Type validation
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Parameter 'x' must be a numpy array, got {type(x)}")
    if not isinstance(groups, np.ndarray):
        raise TypeError(f"Parameter 'groups' must be a numpy array, got {type(groups)}")
    if not isinstance(ngroups, int):
        raise TypeError(f"Parameter 'ngroups' must be an integer, got {type(ngroups)}")
    if not isinstance(nthreads, int):
        raise TypeError(f"Parameter 'nthreads' must be an integer, got {type(nthreads)}")
    
    # Scalar parameter validation
    if ngroups <= 0:
        raise ValueError(f"Parameter 'ngroups' must be positive, got {ngroups}")
    if nthreads <= 0:
        print("Warning: nthreads <= 0, using all available threads")
    
    # --- Array dimension validation ---
    if x.ndim != 2:
        raise ValueError(f"Parameter 'x' must be 2-dimensional, got {x.ndim}D")
    if groups.ndim != 1:
        raise ValueError(f"Parameter 'groups' must be 1-dimensional, got {groups.ndim}D")
    
    nrows, ncols = x.shape
    if nrows == 0 or ncols == 0:
        raise ValueError(f"Input matrix 'x' must have at least one row and one column, got shape {x.shape}")

    if len(groups) != ncols:
        raise ValueError(f"Length of 'groups' ({len(groups)}) must equal number of columns in 'x' ({ncols})")
    
    # --- Data type validation ---
    if x.dtype != np.float64:
        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError(f"Input array must be floating-point, got {x.dtype}")
        x = x.astype(np.float64)

    if groups.dtype not in integer_dtypes:
        raise TypeError(f"Array 'groups' must have an integer dtype in {integer_dtypes}, got {groups.dtype}")
    groups = groups.astype(np.int32)

    # --- Ensure all elements in x are non-negative ---
    if np.any(x < 0):
        min_val = np.min(x)
        raise ValueError(f"All elements in 'x' must be non-negative, "
                         f"but found minimum value: {min_val}")
    # Value range validation
    if len(groups) > 0:
        max_group = np.max(groups)
        if max_group >= ngroups:
            raise ValueError(f"Group indices in 'groups' must be in range [0, {ngroups-1}], "
                             f"found maximum group index: {max_group}")
        
    # --- Call C++ backend ---
    try:
        result = mm.cpp_nnzeroGroups_dense_T(x, groups, ngroups, nthreads)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    
def rank_matrix_csc(
    x: np.ndarray,
    p: np.ndarray,
    rank_data_out: np.ndarray,
    nrow: int,
    ncol: int,
    nthreads: int = 1 
) -> List[List[float]]:
    """
    Compute average ranks for non-zero elements in a CSC-formatted sparse matrix, column-wise.

    This function assigns ranks to non-zero values within each column of a sparse matrix
    stored in CSC format. Tied values receive the average of the ranks they would have
    occupied. Zeros are implicitly treated as the smallest values and are not stored,
    but their count is used to shift the ranks of non-zero entries upward.

    The function modifies the input array `x` in-place by replacing non-zero values
    with their computed ranks (adjusted for implicit zeros).

    Parameters
    ----------
    x : np.ndarray
        1D array of non-zero values in CSC format (dtype: float64).
        Length must equal the number of non-zero elements (nnz).
    p : np.ndarray
        1D array of column pointers in CSC format (dtype: int32 or int64).
        Length must be `ncol + 1`, with `p[0] == 0` and non-decreasing.
    nrow : int
        Number of rows in the matrix. Must be positive.
    ncol : int
        Number of columns in the matrix. Must be positive.
    nthreads : int, optional
        Number of OpenMP threads to use. If <= 0, uses all available threads.
        Default is 1.

    Returns
    -------
    list[list[float]]
        A list of length `ncol`. Each element is a list containing:
        - The sizes of tie groups (number of tied elements) for that column (only for ties with size > 1),
        - Followed by the number of implicit zero elements in that column (as a float).

        Example: `[[2.0, 3.0], [0.0], [4.0, 1.0]]` means:
          - Column 0: one tie of size 2, and 3 zeros.
          - Column 1: no ties, and 0 zeros.
          - Column 2: one tie of size 4, and 1 zero.

    Notes
    -----
    - **This function modifies `x` in-place**: non-zero entries are replaced with their ranks.
    - Ranks start at 1 (i.e., smallest non-zero gets rank = (#zeros + 1)).
    - Input arrays must be contiguous and writable.
    """

    # --- Type validation ---
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Parameter 'x' must be a numpy array, got {type(x)}")
    if not isinstance(p, np.ndarray):
        raise TypeError(f"Parameter 'p' must be a numpy array, got {type(p)}")
    if not isinstance(rank_data_out, np.ndarray):
        raise TypeError(f"Parameter 'rank_data_out' must be a numpy array, got {type(rank_data_out)}")
    
    # Scalar parameter validation
    if not isinstance(nrow, int):
        raise TypeError(f"Parameter 'nrow' must be an integer, got {type(nrow)}")
    if not isinstance(ncol, int):
        raise TypeError(f"Parameter 'ncol' must be an integer, got {type(ncol)}")
    if not isinstance(nthreads, int):
        raise TypeError(f"Parameter 'nthreads' must be an integer, got {type(nthreads)}")
    
    # --- Scalar validation ---
    if nrow <= 0:
        raise ValueError(f"Parameter 'nrow' must be positive, got {nrow}")
    if ncol <= 0:
        raise ValueError(f"Parameter 'ncol' must be positive, got {ncol}")
    if nthreads <= 0:
        print("Warning: nthreads <= 0, using all available threads")
    
    # ---Array dimension validation ---
    if x.ndim != 1:
        raise ValueError(f"Parameter 'x' must be 1-dimensional, got {x.ndim}D")
    if rank_data_out.ndim != 1:
        raise ValueError(f"Parameter 'rank_data_out' must be 1-dimensional, got {rank_data_out.ndim}D")
    if p.ndim != 1:
        raise ValueError(f"Parameter 'p' must be 1-dimensional, got {p.ndim}D")
    
    # Array length validation
    if len(p) != ncol + 1:
        raise ValueError(f"Array 'p' must have length ncol + 1 = {ncol + 1}, "
                        f"got {len(p)}")
    
    # ---Data type validation ---
    if x.dtype != np.float64:
        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError(f"Input array must be floating-point, got {x.dtype}")
        x = x.astype(np.float64)

    if rank_data_out.dtype != np.float64:
        raise TypeError(f"Array 'rank_data_out' must have dtype float64, got {rank_data_out.dtype}")
    
    if p.dtype not in integer_dtypes:
        raise TypeError(f"Array 'p' must have dtype int32 or int64, got {p.dtype}")
    p = p.astype(np.int32)

    # --- Contiguity and writability ---
    if not rank_data_out.flags['C_CONTIGUOUS']:
        raise ValueError("Array 'rank_data_out' must be C-contiguous")
    if not rank_data_out.flags['WRITEABLE']:
        raise ValueError("Array 'rank_data_out' must be writable (will be modified in-place)")
    
    # Array length validation
    if len(x) != len(rank_data_out):
        raise ValueError(f"Array 'rank_data_out' must have the same length as 'x' ({len(x)}), "
                         f"got {len(rank_data_out)}")
    
    # Ensure x are positive
    if np.any(x <= 0):
        # Find the minimum value for better error message
        min_val = np.min(x)
        raise ValueError(f"All elements in 'x' must be positive (greater than 0), "
                         f"but found minimum value: {min_val}")
    
    # Validate CSC format constraints
    if len(p) > 0 and p[0] != 0:
        raise ValueError(f"First element of 'p' must be 0, got {p[0]}")
    
    if len(p) > 0 and p[-1] != len(x):
        raise ValueError(f"Array 'x' must have length equal to number of non-zeros ({p[-1]}), "
                         f"got {len(x)}")
    
    if not np.all(np.diff(p) >= 0):
        raise ValueError("Array 'p' must be non-decreasing (valid CSC format)")
    
    # Call the C++ function
    try:
        result = mm.cpp_rank_matrix_csc(x, p, rank_data_out, nrow, ncol, nthreads)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    
def rank_matrix_csr(
    x: np.ndarray,
    p: np.ndarray,
    i: np.ndarray,
    rank_data_out: np.ndarray,
    nrow: int,
    ncol: int,
    nthreads: int = 1
) -> dict:
    """
    Compute average ranks for non-zero elements in a CSR-formatted sparse matrix, column-wise.

    This function processes a sparse matrix in CSR format and computes average ranks
    of non-zero values **within each column**. Tied values receive the average of the
    ranks they would have occupied. Implicit zeros are treated as smallest values,
    and their count is used to shift non-zero ranks upward.

    Unlike `rank_matrix_csc`, this function **does not modify the input `data`**.
    Instead, it writes the resulting rank values into a pre-allocated output array
    `rank_data_out`, which must have the same length as `data`.
    Parameters
    ----------
    data : np.ndarray
        1D array of non-zero values in CSR format (dtype: float64).
        Length must equal the number of non-zero elements (nnz).
    p : np.ndarray
        1D array of row pointers (dtype: int32 or int64). Length = nrow + 1.
    i : np.ndarray
        1D array of column indices corresponding to each non-zero (dtype: int32 or int64).
        Length = nnz.
    rank_data_out : np.ndarray
        1D output array to store computed rank values (dtype: float64).
        Must be writable, C-contiguous, and have same length as `data`.
    nrow : int
        Number of rows in the matrix. Must be positive.
    ncol : int
        Number of columns in the matrix. Must be positive.
    nthreads : int, optional
        Number of OpenMP threads to use. If <= 0, uses all available threads.
        Default is 1.

    Returns
    -------
    dict
        A dictionary with three keys:
        - 'ties': list of list of float.
                  Length = number of columns. Each sublist contains the sizes
                  (as float) of tie groups for that column (only groups with size > 1) and zero numbers.
        - 'indptr': indptr for csc format, list of int.
                    Length = number of columns + 1.
        - 'indices': indices for csc format, list of int.
                     Length = number of non-zeros.

    Notes
    -----
    - Input arrays `data`, `p`, `i` are **not modified**.
    - `rank_data_out` is **overwritten** with rank values in an internal CSC-like order
      (grouped by column, but original per-column entry order is preserved via indexing).
    - The order of values in `rank_data_out` **does not match** the original `data` order.
      It is ordered column by column (like CSC `data`), not row by row (like CSR `data`).
    """

    # Type validation
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Parameter 'x' must be a numpy array, got {type(x)}")
    if not isinstance(p, np.ndarray):
        raise TypeError(f"Parameter 'p' must be a numpy array, got {type(p)}")
    if not isinstance(i, np.ndarray):
        raise TypeError(f"Parameter 'i' must be a numpy array, got {type(i)}")
    if not isinstance(rank_data_out, np.ndarray):
        raise TypeError(f"Parameter 'rank_data_out' must be a numpy array, got {type(rank_data_out)}")

    # Scalar parameter validation
    if not isinstance(nrow, int):
        raise TypeError(f"Parameter 'nrow' must be an integer, got {type(nrow)}")
    if not isinstance(ncol, int):
        raise TypeError(f"Parameter 'ncol' must be an integer, got {type(ncol)}")
    if not isinstance(nthreads, int):
        raise TypeError(f"Parameter 'nthreads' must be an integer, got {type(nthreads)}")
    
    # Scalar value validation
    if nrow <= 0:
        raise ValueError(f"Parameter 'nrow' must be positive, got {nrow}")
    if ncol <= 0:
        raise ValueError(f"Parameter 'ncol' must be positive, got {ncol}")
    if nthreads <= 0:
        print("Warning: nthreads <= 0, using all available threads")

    # Array dimension validation
    if x.ndim != 1:
        raise ValueError(f"Parameter 'x' must be 1-dimensional, got {x.ndim}D")
    if rank_data_out.ndim != 1:
        raise ValueError(f"Parameter 'rank_data_out' must be 1-dimensional, got {rank_data_out.ndim}D")
    if p.ndim != 1:
        raise ValueError(f"Parameter 'p' must be 1-dimensional, got {p.ndim}D")
    if i.ndim != 1:
        raise ValueError(f"Parameter 'i' must be 1-dimensional, got {i.ndim}D")
    
    # Data type validation
    if x.dtype != np.float64:
        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError(f"Input array must be floating-point, got {x.dtype}")
        x = x.astype(np.float64)
        
    if rank_data_out.dtype != np.float64:
        raise TypeError(f"Array 'rank_data_out' must have dtype float64, got {rank_data_out.dtype}")

    if p.dtype not in integer_dtypes:
        raise TypeError(f"Array 'p' must have dtype int32 or int64, got {p.dtype}")
    if i.dtype not in integer_dtypes:
        raise TypeError(f"Array 'i' must have dtype int32 or int64, got {i.dtype}")
    p, i = p.astype(np.int32), i.astype(np.int32)

    # Contiguity and writability of output
    if not rank_data_out.flags['C_CONTIGUOUS']:
        raise ValueError("Array 'rank_data_out' must be C-contiguous")
    if not rank_data_out.flags['WRITEABLE']:
        raise ValueError("Array 'rank_data_out' must be writable (will be modified in-place)")
    
    # Array length validation
    if len(x) != len(rank_data_out):
        raise ValueError(f"Array 'rank_data_out' must have the same length as 'x' ({len(x)}), "
                         f"got {len(rank_data_out)}")
    if len(x) != len(i):
        raise ValueError(f"Array 'i' must have the same length as 'x' ({len(x)}), "
                         f"got {len(i)}")
    if len(p) != nrow + 1:
        raise ValueError(f"Array 'p' must have length nrow + 1 = {nrow + 1}, "
                         f"got {len(p)}")
    if len(x) == 0:
        raise ValueError("Array 'x' must not be empty")
    
    # Ensure x are positive
    if np.any(x <= 0):
        # Find the minimum value for better error message
        min_val = np.min(x)
        raise ValueError(f"All elements in 'x' must be positive (greater than 0), "
                         f"but found minimum value: {min_val}")
    # Value range validation
    if len(i) > 0:
        max_col = np.max(i)
        if max_col >= ncol:
            raise ValueError(f"Column indices in 'i' must be in range [0, {ncol-1}], "
                             f"found maximum column index: {max_col}")
    # Validate CSR format constraints
    if len(p) > 0:
        if p[0] != 0:
            raise ValueError(f"First element of 'p' must be 0, got {p[0]}")
        if p[-1] != len(x):
            raise ValueError(f"Last element of 'p' must equal length of 'x' ({len(x)}), "
                           f"got {p[-1]}")
        # Check that p is non-decreasing
        if not np.all(np.diff(p) >= 0):
            raise ValueError("Array 'p' must be non-decreasing (valid CSR format)")
    
    # Call the C++ function
    try:
        result_dict = mm.cpp_rank_matrix_csr(x, p, i, rank_data_out, nrow, ncol, nthreads)
        return result_dict
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")

def rank_matrix_csr_(
    x: np.ndarray,
    p: np.ndarray,
    i: np.ndarray,
    rank_data_out: np.ndarray,
    nrow: int,
    ncol: int,
    nthreads: int = 1
) -> dict:
    """
    Compute average ranks for non-zero elements in a CSR-formatted sparse matrix, column-wise.

    This function processes a sparse matrix in CSR format and computes average ranks
    of non-zero values **within each column**. Tied values receive the average of the
    ranks they would have occupied. Implicit zeros are treated as smallest values,
    and their count is used to shift non-zero ranks upward.

    Unlike `rank_matrix_csc`, this function **does not modify the input `data`**.
    Instead, it writes the resulting rank values into a pre-allocated output array
    `rank_data_out`, which must have the same length as `data`.
    Parameters
    ----------
    data : np.ndarray
        1D array of non-zero values in CSR format (dtype: float64).
        Length must equal the number of non-zero elements (nnz).
    p : np.ndarray
        1D array of row pointers (dtype: int32 or int64). Length = nrow + 1.
    i : np.ndarray
        1D array of column indices corresponding to each non-zero (dtype: int32 or int64).
        Length = nnz.
    rank_data_out : np.ndarray
        1D output array to store computed rank values (dtype: float64).
        Must be writable, C-contiguous, and have same length as `data`.
    nrow : int
        Number of rows in the matrix. Must be positive.
    ncol : int
        Number of columns in the matrix. Must be positive.
    nthreads : int, optional
        Number of OpenMP threads to use. If <= 0, uses all available threads.
        Default is 1.

    Returns
    -------
    dict
        A dictionary with three keys:
        - 'ties': list of list of float.
                  Length = number of columns. Each sublist contains the sizes
                  (as float) of tie groups for that column (only groups with size > 1) and zero numbers.
        - 'indptr': indptr for csc format, list of int.
                    Length = number of columns + 1.
        - 'indices': indices for csc format, list of int.
                     Length = number of non-zeros.

    Notes
    -----
    - Input arrays `data`, `p`, `i` are **not modified**.
    - `rank_data_out` is **overwritten** with rank values in an internal CSC-like order
      (grouped by column, but original per-column entry order is preserved via indexing).
    - The order of values in `rank_data_out` **does not match** the original `data` order.
      It is ordered column by column (like CSC `data`), not row by row (like CSR `data`).
    """

    # Type validation
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Parameter 'x' must be a numpy array, got {type(x)}")
    if not isinstance(p, np.ndarray):
        raise TypeError(f"Parameter 'p' must be a numpy array, got {type(p)}")
    if not isinstance(i, np.ndarray):
        raise TypeError(f"Parameter 'i' must be a numpy array, got {type(i)}")
    if not isinstance(rank_data_out, np.ndarray):
        raise TypeError(f"Parameter 'rank_data_out' must be a numpy array, got {type(rank_data_out)}")

    # Scalar parameter validation
    if not isinstance(nrow, int):
        raise TypeError(f"Parameter 'nrow' must be an integer, got {type(nrow)}")
    if not isinstance(ncol, int):
        raise TypeError(f"Parameter 'ncol' must be an integer, got {type(ncol)}")
    if not isinstance(nthreads, int):
        raise TypeError(f"Parameter 'nthreads' must be an integer, got {type(nthreads)}")
    
    # Scalar value validation
    if nrow <= 0:
        raise ValueError(f"Parameter 'nrow' must be positive, got {nrow}")
    if ncol <= 0:
        raise ValueError(f"Parameter 'ncol' must be positive, got {ncol}")
    if nthreads <= 0:
        print("Warning: nthreads <= 0, using all available threads")

    # Array dimension validation
    if x.ndim != 1:
        raise ValueError(f"Parameter 'x' must be 1-dimensional, got {x.ndim}D")
    if rank_data_out.ndim != 1:
        raise ValueError(f"Parameter 'rank_data_out' must be 1-dimensional, got {rank_data_out.ndim}D")
    if p.ndim != 1:
        raise ValueError(f"Parameter 'p' must be 1-dimensional, got {p.ndim}D")
    if i.ndim != 1:
        raise ValueError(f"Parameter 'i' must be 1-dimensional, got {i.ndim}D")
    
    # Data type validation
    if x.dtype != np.float64:
        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError(f"Input array must be floating-point, got {x.dtype}")
        x = x.astype(np.float64)
        
    if rank_data_out.dtype != np.float64:
        raise TypeError(f"Array 'rank_data_out' must have dtype float64, got {rank_data_out.dtype}")

    if p.dtype not in integer_dtypes:
        raise TypeError(f"Array 'p' must have dtype int32 or int64, got {p.dtype}")
    if i.dtype not in integer_dtypes:
        raise TypeError(f"Array 'i' must have dtype int32 or int64, got {i.dtype}")
    p, i = p.astype(np.int32), i.astype(np.int32)

    # Contiguity and writability of output
    if not rank_data_out.flags['C_CONTIGUOUS']:
        raise ValueError("Array 'rank_data_out' must be C-contiguous")
    if not rank_data_out.flags['WRITEABLE']:
        raise ValueError("Array 'rank_data_out' must be writable (will be modified in-place)")
    
    # Array length validation
    if len(x) != len(rank_data_out):
        raise ValueError(f"Array 'rank_data_out' must have the same length as 'x' ({len(x)}), "
                         f"got {len(rank_data_out)}")
    if len(x) != len(i):
        raise ValueError(f"Array 'i' must have the same length as 'x' ({len(x)}), "
                         f"got {len(i)}")
    if len(p) != nrow + 1:
        raise ValueError(f"Array 'p' must have length nrow + 1 = {nrow + 1}, "
                         f"got {len(p)}")
    if len(x) == 0:
        raise ValueError("Array 'x' must not be empty")
    
    # Ensure x are positive
    if np.any(x <= 0):
        # Find the minimum value for better error message
        min_val = np.min(x)
        raise ValueError(f"All elements in 'x' must be positive (greater than 0), "
                         f"but found minimum value: {min_val}")
    # Value range validation
    if len(i) > 0:
        max_col = np.max(i)
        if max_col >= ncol:
            raise ValueError(f"Column indices in 'i' must be in range [0, {ncol-1}], "
                             f"found maximum column index: {max_col}")
    # Validate CSR format constraints
    if len(p) > 0:
        if p[0] != 0:
            raise ValueError(f"First element of 'p' must be 0, got {p[0]}")
        if p[-1] != len(x):
            raise ValueError(f"Last element of 'p' must equal length of 'x' ({len(x)}), "
                           f"got {p[-1]}")
        # Check that p is non-decreasing
        if not np.all(np.diff(p) >= 0):
            raise ValueError("Array 'p' must be non-decreasing (valid CSR format)")
    
    # Call the C++ function
    try:
        result = mm.cpp_rank_matrix_csr_(x, p, i, rank_data_out, nrow, ncol, nthreads)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    
def rank_matrix_dense(
    x: np.ndarray,
    nthreads: int = 1
) -> dict:
    """
    Compute average ranks for each column of a dense matrix.

    This function replaces each element in the input matrix with its average rank
    within its column. Tied values receive the average of the ranks they would have
    occupied. Ranks start at 1 (smallest value gets rank 1).

    Parameters
    ----------
    x : np.ndarray
        2D dense input matrix (dtype: float64). Shape: (nrows, ncols).
        Must be non-empty (at least one row and one column).
    nthreads : int, optional
        Number of OpenMP threads to use. If <= 0, uses all available threads.
        Default is 1.

    Returns
    -------
    dict
        A dictionary with two keys:
        - 'X_ranked': np.ndarray of shape (nrows, ncols), dtype float64.
                      Same shape as input, with values replaced by column-wise ranks.
        - 'ties': list of list of float.
                  Length = number of columns. Each sublist contains the sizes
                  (as float) of tie groups for that column (only groups with size > 1).

    Notes
    -----
    - Input matrix `x` is **not modified**; a new array is returned.
    - Ranking is **column-wise and independent**.
    """

    # Type validation
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Parameter 'x' must be a numpy array, got {type(x)}")
    if not isinstance(nthreads, int):
        raise TypeError(f"Parameter 'nthreads' must be an integer, got {type(nthreads)}")
    
    # Array dimension validation
    if x.ndim != 2:
        raise ValueError(f"Parameter 'x' must be 2-dimensional, got {x.ndim}D")
    
    # Scaler value validation
    nrows, ncols = x.shape
    if nrows == 0 or ncols == 0:
        raise ValueError(f"Input matrix must be non-empty, got shape {x.shape}")
    if nthreads <= 0:
        print("Warning: nthreads <= 0, using all available threads")

    # Data type validation
    if x.dtype != np.float64:
        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError(f"Input array must be floating-point, got {x.dtype}")
        x = x.astype(np.float64)
    
    # non-negativity check
    if np.any(x < 0):
        min_val = np.min(x)
        raise ValueError(f"All elements in 'x' must be non-negative, "
                         f"but found minimum value: {min_val}")
    
    # Call C++ function
    try:
        result_dict = mm.cpp_rank_matrix_dense(x, nthreads)
        return result_dict
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")

def group_rank_csc(
    x: np.ndarray,
    p: np.ndarray,
    i: np.ndarray,
    rank_data_out: np.ndarray,
    groups: np.ndarray,
    nrow: int,
    ncol: int,
    nthreads: int = 1
) -> None:
    """
    Compute average ranks for non-zero elements in a CSC-formatted sparse matrix,
    grouped by specified column groups.

    This function assigns ranks to non-zero values within each group of columns
    in a sparse matrix stored in CSC format. Zeros are implicitly treated as the smallest
    values and are not stored, the rank starts from 1.

    The function modifies the input array `rank_data_out` in-place by replacing non-zero values
    with their computed ranks (adjusted for implicit zeros).

    Parameters
    ----------
    x : np.ndarray
        1D array of non-zero values in CSC format (dtype: float64).
        Length must equal the number of non-zero elements (nnz).
    p : np.ndarray
        1D array of column pointers in CSC format (dtype: int32 or int64).
        Length must be `ncol + 1`, with `p[0] == 0` and non-decreasing.
    i : np.ndarray
        1D array of row indices corresponding to each non-zero (dtype: int32 or int64).
        Length must equal the number of non-zero elements (nnz).
    rank_data_out : np.ndarray
        1D output array to store computed rank values (dtype: float64).
        Must be writable, contiguous, and have same length as `x`.
    groups : np.ndarray
        1D array assigning each row to a group (dtype: any integer type).
        Length must be equal to number of rows in the matrix.
        Group indices must be in range [0, ngroups - 1].
    ncol : int
        Number of columns in the matrix. Must be positive.
    nrow : int
        Number of rows in the matrix. Must be positive.
        Just for input tests, not used for calc.
    nthreads : int, optional
        Number of OpenMP threads to use. If <= 0, uses all available threads.
        Default is 1.
    Returns
    -------
    None
        The function modifies `rank_data_out` in-place; no return value.
    """
    # Type validation
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Parameter 'x' must be a numpy array, got {type(x)}")
    if not isinstance(p, np.ndarray):
        raise TypeError(f"Parameter 'p' must be a numpy array, got {type(p)}")
    if not isinstance(i, np.ndarray):
        raise TypeError(f"Parameter 'i' must be a numpy array, got {type(i)}")
    if not isinstance(rank_data_out, np.ndarray):
        raise TypeError(f"Parameter 'rank_data_out' must be a numpy array, got {type(rank_data_out)}")
    if not isinstance(groups, np.ndarray):
        raise TypeError(f"Parameter 'groups' must be a numpy array, got {type(groups)}")
    if not isinstance(ncol, int):
        raise TypeError(f"Parameter 'ncol' must be an integer, got {type(ncol)}")
    if not isinstance(nthreads, int):
        raise TypeError(f"Parameter 'nthreads' must be an integer, got {type(nthreads)}")
    
    # Scalar value validation
    if ncol <= 0:
        raise ValueError(f"Parameter 'ncol' must be positive, got {ncol}")
    if nrow <= 0:
        raise ValueError(f"Parameter 'nrow' must be positive, got {nrow}")
    if nthreads <= 0:
        print("Warning: nthreads <= 0, using all available threads")
    
    # Array dimension validation
    if x.ndim != 1:
        raise ValueError(f"Parameter 'x' must be 1-dimensional, got {x.ndim}D")
    if rank_data_out.ndim != 1:
        raise ValueError(f"Parameter 'rank_data_out' must be 1-dimensional, got {rank_data_out.ndim}D")
    if p.ndim != 1:
        raise ValueError(f"Parameter 'p' must be 1-dimensional, got {p.ndim}D")
    if i.ndim != 1:
        raise ValueError(f"Parameter 'i' must be 1-dimensional, got {i.ndim}D")
    if groups.ndim != 1:
        raise ValueError(f"Parameter 'groups' must be 1-dimensional, got {groups.ndim}D")
    
    # Data type validation
    if x.dtype != np.float64:
        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError(f"Input array must be floating-point, got {x.dtype}")
        x = x.astype(np.float64)
    
    if rank_data_out.dtype != np.float64:
        raise TypeError(f"Array 'rank_data_out' must have dtype float64, got {rank_data_out.dtype}")

    if p.dtype not in integer_dtypes:
        raise TypeError(f"Array 'p' must have dtype int32 or int64, got {p.dtype}")
    if i.dtype not in integer_dtypes:
        raise TypeError(f"Array 'i' must have dtype int32 or int64, got {i.dtype}")
    if groups.dtype not in integer_dtypes:
        raise TypeError(f"Array 'groups' must have dtype int32 or int64, got {groups.dtype}")
    p, i, groups = p.astype(np.int32), i.astype(np.int32), groups.astype(np.int32)

    # Contiguity and writability of output
    if not rank_data_out.flags['C_CONTIGUOUS']:
        raise ValueError("Array 'rank_data_out' must be C-contiguous")
    if not rank_data_out.flags['WRITEABLE']:
        raise ValueError("Array 'rank_data_out' must be writable (will be modified in-place)")
    
    # Array length validation
    if len(x) != len(rank_data_out):
        raise ValueError(f"Array 'rank_data_out' must have the same length as 'x' ({len(x)}), "
                         f"got {len(rank_data_out)}")
    if len(x) != len(i):
        raise ValueError(f"Array 'i' must have the same length as 'x' ({len(x)}), "
                         f"got {len(i)}")
    if len(p) != ncol + 1:
        raise ValueError(f"Array 'p' must have length ncol + 1 = {ncol + 1}, "
                         f"got {len(p)}")
    if len(groups) != nrow:
        raise ValueError(f"Array 'groups' must have length equal to number of rows ({nrow}), "
                         f"got {len(groups)}")
    if len(x) == 0:
        raise ValueError("Array 'x' must not be empty")

    
    # Ensure x are positive
    if np.any(x <= 0):
        # Find the minimum value for better error message
        min_val = np.min(x)
        raise ValueError(f"All elements in 'x' must be positive (greater than 0), "
                         f"but found minimum value: {min_val}")
    # Value range validation
    if len(i) > 0:
        max_row = np.max(i)
        if max_row >= nrow:
            raise ValueError(f"Row indices in 'i' must be in range [0, {nrow-1}], "
                             f"found maximum row index: {max_row}")
    
    # Validate CSC format constraints
    if len(p) > 0 and p[0] != 0:
        raise ValueError(f"First element of 'p' must be 0, got {p[0]}")
    if len(p) > 0 and p[-1] != len(x):
        raise ValueError(f"Array 'x' must have length equal to number of non-zeros ({p[-1]}), "
                         f"got {len(x)}")
    
    if not np.all(np.diff(p) >= 0):
        raise ValueError("Array 'p' must be non-decreasing (valid CSC format)")
    
    # Call the C++ function
    try:
        mm.cpp_group_rank_csc(x, rank_data_out, p, i, groups, ncol, nthreads)
        return rank_data_out
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")

def group_rank_dense(
    x: np.ndarray,
    rank_data_out: np.ndarray,
    groups: np.ndarray,
    nthreads: int = 1
) -> None:
    """
    Compute average ranks for non-zero elements in a dense matrix.

    This function assigns ranks to non-zero values within each group of columns
    in a dense matrix. Zeros are implicitly treated as the smallest values and 
    are not stored, the rank started from 1.

    The function modifies the input array `rank_data_out` in-place by replacing non-zero values
    with their computed ranks (adjusted for implicit zeros).

    Parameters
    ----------
    x : np.ndarray
        2D dense input matrix (dtype: float64). Shape: (nrows, ncols).
    rank_data_out : np.ndarray
        2D output array to store computed rank values (dtype: float64).
        Must be writable, contiguous, and have same shape as `x`.
    groups : np.ndarray
        1D array assigning each row to a group (dtype: any integer type).
        Length must be equal to number of rows in the matrix.
        Group indices must be in range [0, ngroups - 1].
    nthreads : int, optional
        Number of OpenMP threads to use. If <= 0, uses all available threads.
        Default is 1.
    Returns
    -------
    None
        The function modifies `rank_data_out` in-place; no return value.
    """
    # Type validation
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Parameter 'x' must be a numpy array, got {type(x)}")
    if not isinstance(rank_data_out, np.ndarray):
        raise TypeError(f"Parameter 'rank_data_out' must be a numpy array, got {type(rank_data_out)}")
    if not isinstance(groups, np.ndarray):
        raise TypeError(f"Parameter 'groups' must be a numpy array, got {type(groups)}")
    if not isinstance(nthreads, int):
        raise TypeError(f"Parameter 'nthreads' must be an integer, got {type(nthreads)}")
    
    # Array dimension validation
    if x.ndim != 2:
        raise ValueError(f"Parameter 'x' must be 2-dimensional, got {x.ndim}D")
    if rank_data_out.ndim != 2:
        raise ValueError(f"Parameter 'rank_data_out' must be 2-dimensional, got {rank_data_out.ndim}D")
    if groups.ndim != 1:
        raise ValueError(f"Parameter 'groups' must be 1-dimensional, got {groups.ndim}D")
    
    nrow, ncol = x.shape

    # Scalar value validation
    if nrow == 0 or ncol == 0:
        raise ValueError(f"Input matrix must be non-empty, got shape {x.shape}")
    if len(groups) != nrow:
        raise ValueError(f"Array 'groups' must have length equal to number of rows ({nrow}), "
                         f"got {len(groups)}")
    if nthreads <= 0:
        print("Warning: nthreads <= 0, using all available threads")
    
    # Data type validation
    if x.dtype != np.float64:
        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError(f"Input array must be floating-point, got {x.dtype}")
        x = x.astype(np.float64)
    
    if rank_data_out.dtype != np.float64:
        raise TypeError(f"Array 'rank_data_out' must have dtype float64, got {rank_data_out.dtype}")

    if groups.dtype not in integer_dtypes:
        raise TypeError(f"Array 'groups' must have dtype int32 or int64, got {groups.dtype}")
    groups = groups.astype(np.int32)

    # non-negativity check
    if np.any(x < 0):
        min_val = np.min(x)
        raise ValueError(f"All elements in 'x' must be non-negative, "
                         f"but found minimum value: {min_val}")
    

    # call c++ function
    try:
        mm.cpp_group_rank_dense(x, rank_data_out, groups, nthreads)
        return rank_data_out
    except Exception as e:
        print(f"C++ function failed: {e}")