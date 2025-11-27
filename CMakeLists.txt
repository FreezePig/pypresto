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
    if(EXISTS "/opt/homebrew/opt/libomp")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    elseif(EXISTS "/usr/local/opt/libomp")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")
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
    # Linux and macOS use standard CMake lookup.
    find_package(PkgConfig QUIET)
    if (PKG_CONFIG_FOUND)
        pkg_check_modules(OpenBLAS openblas)
    endif()

    if(NOT OpenBLAS_FOUND)
        find_package(OpenBLAS REQUIRED)
    endif()

    if(OpenBLAS_INCLUDE_DIRS)
        include_directories(${OpenBLAS_INCLUDE_DIRS})
    else()
        find_path(CBLAS_INCLUDE_DIR cblas.h
                PATHS /usr/include/openblas /usr/local/include/openblas /opt/homebrew/include /usr/include
        )
        if(CBLAS_INCLUDE_DIR)
            include_directories(${CBLAS_INCLUDE_DIR})
        endif()
    endif()
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
endif()

# compilation options
if(MSVC)
    target_compile_options(matrix_module PRIVATE /W3 /O2)
else()
    target_compile_options(matrix_module PRIVATE -O3 -Wall -fvisibility=hidden)
endif()

install(TARGETS matrix_module DESTINATION .)