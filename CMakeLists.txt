cmake_minimum_required(VERSION 3.15)
project(pypresto LANGUAGES CXX)

# --- 1. Basic Set ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- 2. Find Python Environment ---
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# --- 3. Find Openmp ---
#set(OMP_FOUND FALSE)
#
#if(APPLE)
#    find_program(BREW_CMD brew)
#    set(LIBOMP_PREFIX "")
#    if(BREW_CMD)
#        execute_process(COMMAND ${BREW_CMD} --prefix libomp
#                OUTPUT_VARIABLE LIBOMP_PREFIX
#                OUTPUT_STRIP_TRAILING_WHITESPACE)
#    endif()
#
#    if(NOT LIBOMP_PREFIX)
#        if(EXISTS "/opt/homebrew/opt/libomp")
#            set(LIBOMP_PREFIX "/opt/homebrew/opt/libomp")
#        elseif(EXISTS "/usr/local/opt/libomp")
#            set(LIBOMP_PREFIX "/usr/local/opt/libomp")
#        endif()
#    endif()
#    if(LIBOMP_PREFIX AND EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
#        message(STATUS "Configuring Custom OpenMP from: ${LIBOMP_PREFIX}")
#
#        if(NOT TARGET OpenMP::OpenMP_CXX)
#            add_library(OpenMP::OpenMP_CXX SHARED IMPORTED)
#            set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
#                    IMPORTED_LOCATION "${LIBOMP_PREFIX}/lib/libomp.dylib"
#                    INTERFACE_INCLUDE_DIRECTORIES "${LIBOMP_PREFIX}/include"
#                    INTERFACE_COMPILE_OPTIONS "-Xpreprocessor;-fopenmp"
#                    INTERFACE_LINK_LIBRARIES "${LIBOMP_PREFIX}/lib/libomp.dylib"
#            )
#        endif()
#        set(OMP_FOUND TRUE)
#    endif()
#endif()
#
#if(NOT OMP_FOUND)
#    find_package(OpenMP REQUIRED)
#    message(STATUS "OpenMP Found via standard package. Version: ${OpenMP_CXX_VERSION}")
#endif()
find_package(OpenMP REQUIRED)

if(OpenMP_FOUND)
    message(STATUS "OpenMP Found. Version: ${OpenMP_CXX_VERSION}")
    message(STATUS "OpenMP Libs: ${OpenMP_CXX_LIBRARIES}")
else()
    message(FATAL_ERROR "Could NOT find OpenMP. Please ensure libomp is installed and paths are set.")
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

# NumPy headers (only if needed)
target_include_directories(matrix_module PRIVATE ${NUMPY_INCLUDE_DIR})

# BLAS
if(WIN32)
    target_include_directories(matrix_module PRIVATE "${OPENBLAS_ROOT}/include")
    target_link_libraries(matrix_module PRIVATE "${OPENBLAS_ROOT}/lib/libopenblas.dll.a")
else()
    target_include_directories(matrix_module PRIVATE ${BLAS_INCLUDE_DIRS})
    target_link_libraries(matrix_module PRIVATE ${BLAS_LIBRARIES})
endif()

# OpenMP
target_link_libraries(matrix_module PRIVATE OpenMP::OpenMP_CXX)

# compilation options
if(MSVC)
    target_compile_options(matrix_module PRIVATE /W3 /O2)
else()
    target_compile_options(matrix_module PRIVATE -O3 -Wall -fvisibility=hidden)
endif()

# macOS RPATH
if(APPLE)
    set_target_properties(matrix_module PROPERTIES
            INSTALL_RPATH "@loader_path"
            BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

install(TARGETS matrix_module DESTINATION .)
