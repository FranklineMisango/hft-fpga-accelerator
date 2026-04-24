cmake_minimum_required(VERSION 3.20)
project(VeriTrade-Dashboard VERSION 1.0.0 LANGUAGES CXX CUDA)

# ============================================================================
# Configuration
# ============================================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)  # RTX 40, A100, H100

# Performance flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -use_fast_math")

# Institutional settings
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS ON)

# ============================================================================
# Dependencies
# ============================================================================

find_package(OpenGL REQUIRED)
find_package(GLFW3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(CUB REQUIRED)  # NVIDIA CUB library for reduction kernels

# GLM (header-only math library)
find_package(glm REQUIRED)

# ImGui (build from source)
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dashboards/imgui")
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# ============================================================================
# Main Dashboard Executable
# ============================================================================

add_executable(imgui_dashboard
    dashboards/imgui_dashboard.cpp
    ${IMGUI_SOURCES}
)

target_include_directories(imgui_dashboard PRIVATE
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

target_link_libraries(imgui_dashboard PRIVATE
    OpenGL::GL
    glfw
    glm::glm
)

# Performance profiling
if(ENABLE_PROFILING)
    target_compile_definitions(imgui_dashboard PRIVATE ENABLE_PROFILING=1)
    target_link_libraries(imgui_dashboard PRIVATE gperftools)
endif()

# ============================================================================
# GPU Market Data Processor (CUDA)
# ============================================================================

add_library(market_data_gpu SHARED
    backends/gpu_market_processor.cu
)

target_include_directories(market_data_gpu PRIVATE
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(market_data_gpu PRIVATE
    ${CUDA_LIBRARIES}
    CUDA::cublas
    cub::cub
)

# ============================================================================
# Performance Benchmarks
# ============================================================================

enable_testing()

add_executable(benchmark_dashboard
    dashboards/benchmark_dashboard.cpp
)

target_link_libraries(benchmark_dashboard PRIVATE
    market_data_gpu
    OpenGL::GL
    glfw
)

add_test(NAME dashboard_benchmark COMMAND benchmark_dashboard)

# ============================================================================
# Installation
# ============================================================================

install(TARGETS imgui_dashboard market_data_gpu
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

install(FILES dashboards/imgui_dashboard.cpp
    DESTINATION include
)
