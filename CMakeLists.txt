#
# GPU word feature extraction code for NLP
#

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(wordfeat C CXX)

set(Wordfeat_VERSION_MAJOR 0)
set(Wordfeat_VERSION_MAJOR 1)

# if(UNIX)
#     SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
# endif()

# Include project source directory and project binary directory
include_directories(${Caffe_INCLUDE_DIR} ${PROJECT_BINARY_DIR})

# Set directory for project headers
set(WordFeat_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
# Include directory for project headers
include_directories(${WordFeat_INCLUDE_DIR})

# Set library to link
set(WordFeat_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

include_directories(${WordFeat_LIB_DIR})

add_subdirectory(src)

# Include cmake module
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# Include cmake helpers 
include(cmake/Targets.cmake)

#
# Setting for Nvidia CUDA GPU 
#
find_package(CUDA REQUIRED)

if (CUDA_FOUND)
  # For some MPs it may be necessary to specify the compute capability of your
  # NVidia GPU. In that case, simply uncomment the following two lines that start
  # with 'set' and change the value of COMPUTE_CAPABILITY to one of the following
  # hardware architectures: Tesla-class '11', '12', or '13'; Fermi-class '20'
  # or '21'; Kepler-class '30', '32', '35' or '37'; or Maxwell-class '50' or '52'
  set(COMPUTE_CAPABILITY 20)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--gpu-architecture;sm_${COMPUTE_CAPABILITY})

  # Set additional linker options on UNIX platforms
  if (UNIX)
      set(wordfeat_LINK_LIBRARIES -lm)
      if (NOT APPLE)
          include(CheckLibraryExists)
          check_library_exists(rt clock_gettime "time.h" HAVE_CLOCK_GETTIME)
          if (HAVE_CLOCK_GETTIME)
              list(APPEND wordfeat_LINK_LIBRARIES -lrt)
          endif ()
      endif ()
  endif ()

  # 
  # Find packages:
  #   1. GLOG
  #   2. GFLAGS 

  find_package(Glog REQUIRED)
  include_directories(SYSTEM ${GLOG_INCLUDE_DIRS})
  list(APPEND wordfeat_LINK_LIBRARIES ${GLOG_LIBRARIES})
  
  find_package(GFlags REQUIRED)
  include_directories(SYSTEM ${GFLAGS_INCLUDE_DIRS})
  list(APPEND wordfeat_LINK_LIBRARIES ${GFLAGS_LIBRARIES})

  # 
  # Set wordfeat runtime output directory
  set(wordfeat_RUNTIME_OUTPUT_DIR ${PROJECT_SOURCE_DIR}/build)

  # 
  # Compile executables for GPU program 
  set(EXECUTABLES device_check wordfeat_naive wordfeat)
  foreach (TARGET ${EXECUTABLES})
    CUDA_ADD_EXECUTABLE(
      ${TARGET}
      ${PROJECT_SOURCE_DIR}/tools/${TARGET}.cu
      OPTIONS ${wordfeat_NVCC_OPTIONS}
    )
    if (DEFINED wordfeat_STANDARD_LIBRARY)
      set_target_properties(
          ${TARGET}
          PROPERTIES
          COMPILE_FLAGS ${wordfeat_STANDARD_LIBRARY}
          LINK_FLAGS ${wordfeat_STANDARD_LIBRARY}
        )
    endif ()
    if (UNIX)
      target_link_libraries(
          ${TARGET}
          ${wordfeat_LINK_LIBRARIES}
      )
    endif ()

    # Link source object file 
    target_link_libraries(${TARGET} wordfeat_lib)

    wordfeat_set_runtime_directory(${TARGET} ${wordfeat_RUNTIME_OUTPUT_DIR})
  endforeach ()

# 
# CTEST Part
#

# COMMENT OUT FOR NOW
#  enable_testing()


endif()
