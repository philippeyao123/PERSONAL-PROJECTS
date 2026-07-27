cmake_minimum_required(VERSION 3.20)

project(qf_rates VERSION 0.1.0 LANGUAGES CXX)

option(QF_BUILD_TESTS "Build the unit and property tests" ON)
option(QF_BUILD_EXAMPLES "Build examples" ON)
option(QF_BUILD_BENCHMARKS "Build micro-benchmarks" OFF)
option(QF_BUILD_PYTHON "Build optional pybind11 bindings" OFF)
option(QF_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(QF_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(FetchContent)

add_library(qf_core
    src/core/numerics.cpp
    src/core/random.cpp
)
add_library(qf::core ALIAS qf_core)

add_library(qf_rates
    src/rates/yield_curve.cpp
    src/rates/swap.cpp
    src/rates/options.cpp
    src/rates/g2pp.cpp
    src/rates/monte_carlo.cpp
    src/rates/calibration.cpp
    src/rates/lsm.cpp
    src/rates/risk.cpp
    src/rates/xva.cpp
)
add_library(qf::rates ALIAS qf_rates)

target_compile_features(qf_core PUBLIC cxx_std_20)
target_compile_features(qf_rates PUBLIC cxx_std_20)
target_include_directories(qf_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(qf_rates PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(qf_rates PUBLIC qf_core)

foreach(target qf_core qf_rates)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
        if(QF_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        if(QF_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
        if(QF_ENABLE_SANITIZERS)
            # Static libraries do not perform a final link, so sanitizer runtime
            # options must propagate to executables that consume the libraries.
            target_compile_options(${target} PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(${target} PUBLIC -fsanitize=address,undefined)
        endif()
    endif()
endforeach()

# Eigen is integrated without leaking it into the public ABI or install export.
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(TARGET Eigen3::Eigen)
    target_link_libraries(qf_core PRIVATE $<BUILD_INTERFACE:Eigen3::Eigen>)
else()
    FetchContent_Declare(
        eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(eigen3)
    if(NOT eigen3_POPULATED)
        if(POLICY CMP0169)
            cmake_policy(SET CMP0169 OLD)
        endif()
        FetchContent_Populate(eigen3)
    endif()
    target_include_directories(qf_core SYSTEM PRIVATE "${eigen3_SOURCE_DIR}")
endif()
target_compile_definitions(qf_core PRIVATE QF_HAS_EIGEN=1)

if(QF_BUILD_TESTS)
    include(CTest)
    find_package(Catch2 3 QUIET)
    if(NOT Catch2_FOUND)
        FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG v3.7.1
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(Catch2)
    endif()
    add_executable(qf_rates_tests
        tests/test_curves.cpp
        tests/test_swap.cpp
        tests/test_options.cpp
        tests/test_numerics.cpp
        tests/test_g2pp.cpp
        tests/test_monte_carlo.cpp
        tests/test_calibration_lsm_risk.cpp
        tests/test_xva.cpp
    )
    target_link_libraries(qf_rates_tests PRIVATE qf::rates Catch2::Catch2WithMain)
    target_compile_features(qf_rates_tests PRIVATE cxx_std_20)
    include(Catch)
    catch_discover_tests(qf_rates_tests)
endif()

if(QF_BUILD_EXAMPLES)
    add_executable(qf_rates_demo examples/end_to_end.cpp)
    target_link_libraries(qf_rates_demo PRIVATE qf::rates)
endif()

if(QF_BUILD_BENCHMARKS)
    add_executable(qf_rates_bench benchmarks/pricing_benchmark.cpp)
    target_link_libraries(qf_rates_bench PRIVATE qf::rates)
endif()

if(QF_BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(qf_rates_python python/bindings.cpp)
    target_link_libraries(qf_rates_python PRIVATE qf::rates)
endif()

install(TARGETS qf_core qf_rates
    EXPORT qf_ratesTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT qf_ratesTargets
    FILE qf_ratesTargets.cmake
    NAMESPACE qf::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/qf_rates
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/qf_ratesConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/qf_ratesConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/qf_ratesConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/qf_rates
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/qf_ratesConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/qf_ratesConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/qf_rates
)
