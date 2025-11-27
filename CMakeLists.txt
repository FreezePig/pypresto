cmake_minimum_required(VERSION 3.15)
project(pypresto LANGUAGES CXX)

# --- 1. basic set ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 2. find python environment ---
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# --- 3. find openmp ---
if(APPLE)
    find_program(BREW_CMD brew)
    set(LIBOMP_PREFIX "")
    if(BREW_CMD)
        execute_process(COMMAND ${BREW_CMD} --prefix libomp
                OUTPUT_VARIABLE LIBOMP_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
    if(NOT LIBOMP_PREFIX)
        if(EXISTS "/opt/homebrew/opt/libomp")     # Apple Silicon
            set(LIBOMP_PREFIX "/opt/homebrew/opt/libomp")
        elseif(EXISTS "/usr/local/opt/libomp")    # Intel Mac
            set(LIBOMP_PREFIX "/usr/local/opt/libomp")
        endif()
    endif()
    if(LIBOMP_PREFIX)
        message(STATUS ">>> Found macOS OpenMP prefix: ${LIBOMP_PREFIX}")
        include_directories("${LIBOMP_PREFIX}/include")
        link_directories("${LIBOMP_PREFIX}/lib")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    else()
        message(WARNING ">>> Could NOT find libomp prefix! OpenMP linking might fail.")
    endif()
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found. Parallel processing enabled.")
else()
    message(WARNING "OpenMP NOT found. Will build single-threaded.")
endif()

# ---4. OpenBLAS setup ---
if(WIN32)
    set(OPENBLAS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/OpenBLAS")
    
    if(NOT EXISTS "${OPENBLAS_ROOT}")
        message(FATAL_ERROR "Windows build requires external/OpenBLAS at: ${OPENBLAS_ROOT}")
    endif()

    include_directories("${OPENBLAS_ROOT}/include")
    link_directories("${OPENBLAS_ROOT}/lib")
    
    # Try linking the static library libopenblas.a (if it exists) to avoid DLL dependencies.
    # If only .lib or .dll.a files are available, you will need to use delvewheel to fix them after packaging.
    set(BLAS_LIBRARIES libopenblas)
else()

    # 1. Find any BLAS library (OpenBLAS/MKL/Atlas etc.)
    # Using CMake FindBLAS.cmake
    find_package(BLAS REQUIRED)

    # 2. Find header cblas.h
    if(NOT BLAS_INCLUDE_DIRS)
        find_path(BLAS_INCLUDE_DIRS cblas.h
                PATHS
                /usr/include/openblas
                /usr/local/include/openblas
                /opt/homebrew/include
                /opt/homebrew/opt/openblas/include
                /usr/include
                /usr/local/include
        )
    endif()

    if(BLAS_INCLUDE_DIRS)
        include_directories(${BLAS_INCLUDE_DIRS})
        message(STATUS "Found BLAS headers in: ${BLAS_INCLUDE_DIRS}")
    else()
        message(WARNING "Could NOT find cblas.h. Compilation might fail if headers are missing.")
    endif()
    message(STATUS "Found BLAS libraries: ${BLAS_LIBRARIES}")

endif()

# --- 5. define target module ---
pybind11_add_module(matrix_module main.cpp)

# Header file path
target_include_directories(matrix_module PRIVATE
    ${NUMPY_INCLUDE_DIR}  # find pybind11/numpy.h
)

# link library
target_link_libraries(matrix_module PRIVATE ${BLAS_LIBRARIES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(matrix_module PRIVATE OpenMP::OpenMP_CXX)
    if(APPLE)
        target_link_libraries(matrix_module PRIVATE omp)
    endif()
endif()

# compilation options
if(MSVC)
    target_compile_options(matrix_module PRIVATE /W3 /O2)
else()
    target_compile_options(matrix_module PRIVATE -O3 -Wall -fvisibility=hidden)
endif()

install(TARGETS matrix_module DESTINATION .)