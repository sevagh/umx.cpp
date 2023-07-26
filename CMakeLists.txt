# cmake file to compile src/
# link against included submodules libnyquist

cmake_minimum_required(VERSION 3.0)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# improve CMAKE_CXX_FLAGS_RELEASE to every possible optimization considering Eigen3, target arch (my CPU is a Ryzen 5950X), and compiler
# also linux kernel features and whatnot
# aside from "-O3"
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mfma -ffast-math -fno-signed-zeros -fno-trapping-math -fassociative-math -freciprocal-math -fno-math-errno -fno-rounding-math -fno-signaling-nans -fno-unsafe-math-optimizations -fno-trapping-math -fno-math-errno")

# define a macro NDEBUG for Eigen3 release builds
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")

# set EIGEN_USE_BLAS to 1 and link to OpenBLAS
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE")

project(umx.cpp)
enable_testing()

# set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# add openmp support
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    include_directories(${OpenMP_CXX_INCLUDE_DIRS})
endif()

# compile vendored submodule libnyquist
set(LIBNYQUIST_BUILD_EXAMPLE OFF CACHE BOOL "Disable libnyquist example")
add_subdirectory(vendor/libnyquist)

# add library Eigen3
include_directories(vendor/eigen)

# add OpenBLAS for blas + lapack
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# include vendor submodules libnyquist
include_directories(vendor/libnyquist/include)

# include src/ as include directory
include_directories(src)

# include src/*.cpp and src/*.c as source files
file(GLOB SOURCES "src/*.cpp")

# compile library, link against libnyquist
add_library(umx.cpp.lib SHARED ${SOURCES})
target_link_libraries(umx.cpp.lib libnyquist ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} lapacke)
if(OPENMP_FOUND)
    target_link_libraries(umx.cpp.lib ${OpenMP_CXX_LIBRARIES})
endif()

file(GLOB SOURCES_TO_LINT "src/*.cpp" "src/*.hpp" "umx.cpp" "test/*.cpp")

# add target to run standard lints and formatters
add_custom_target(lint
    COMMAND clang-format -i ${SOURCES_TO_LINT} --style=file
    # add clang-tidy command
    # add include dirs to clang-tidy
    COMMAND cppcheck --enable=all --suppress=missingIncludeSystem ${SOURCES_TO_LINT} --std=c++17
    COMMAND scan-build -o ${CMAKE_BINARY_DIR}/scan-build-report make -C ${CMAKE_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# add target to compile umx.cpp, the main driver program for umx.cpp
add_executable(umx.cpp.main umx.cpp)
# link it against umx.cpp.lib
target_link_libraries(umx.cpp.main umx.cpp.lib)

# add target to run cpp tests in test/ directory with gtest

# include test/*.cpp as test files
file(GLOB TEST_SOURCES "test/*.cpp")

add_executable(umx.cpp.test ${TEST_SOURCES})
target_link_libraries(umx.cpp.test umx.cpp.lib gtest gtest_main libnyquist)
add_test(NAME tests COMMAND umx.cpp.test)
