# TBB build scripts.
#
# - STATIC_TBB_INCLUDE_DIR
# - STATIC_TBB_LIB_DIR
# - STATIC_TBB_LIBRARIES
#
# Notes:
# The name "STATIC" is used to avoid naming collisions for other 3rdparty CMake
# files (e.g. PyTorch) that also depends on MKL.

include(ExternalProject)

# Where MKL and TBB headers and libs will be installed.
# This needs to be consistent with mkl.cmake.
set(MKL_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/mkl_install)
set(STATIC_MKL_INCLUDE_DIR "${MKL_INSTALL_PREFIX}/include/")
set(STATIC_MKL_LIB_DIR "${MKL_INSTALL_PREFIX}/${NNRT_INSTALL_LIB_DIR}")

# TBB variables exported for PyTorch Ops and TensorFlow Ops
set(STATIC_TBB_INCLUDE_DIR "${STATIC_MKL_INCLUDE_DIR}")
set(STATIC_TBB_LIB_DIR "${STATIC_MKL_LIB_DIR}")
set(STATIC_TBB_LIBRARIES tbb_static tbbmalloc_static)

# To generate a new patch, go inside TBB source folder:
# 1. Checkout to a new branch
# 2. Make changes, commit
# 3. git format-patch master
# 4. Copy the new patch file back to Open3d.
ExternalProject_Add(
    ext_tbb
    PREFIX tbb
    GIT_REPOSITORY https://github.com/wjakob/tbb.git
    GIT_TAG 141b0e310e1fb552bdca887542c9c1a8544d6503 # Sept 2020
    UPDATE_COMMAND ""
    PATCH_COMMAND git checkout -f 141b0e310e1fb552bdca887542c9c1a8544d6503
    COMMAND git apply ${NNRT_3RDPARTY_DIR}/tbb/0001-Allow-selecttion-of-static-dynamic-MSVC-runtime.patch
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${MKL_INSTALL_PREFIX}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DSTATIC_WINDOWS_RUNTIME=${STATIC_WINDOWS_RUNTIME}
        -DTBB_BUILD_TBBMALLOC=ON
        -DTBB_BUILD_SHARED=OFF
        -DTBB_BUILD_STATIC=ON
        -DTBB_BUILD_TESTS=OFF
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
        -DTBB_INSTALL_ARCHIVE_DIR=${NNRT_INSTALL_LIB_DIR}
        -DTBB_CMAKE_PACKAGE_INSTALL_DIR=${NNRT_INSTALL_LIB_DIR}/cmake/tbb
    BUILD_BYPRODUCTS
        ${STATIC_TBB_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}tbb_static${CMAKE_STATIC_LIBRARY_SUFFIX}
        ${STATIC_TBB_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}tbbmalloc_static${CMAKE_STATIC_LIBRARY_SUFFIX}
)
