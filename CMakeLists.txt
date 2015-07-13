#
# GPU word feature extraction code for NLP
#

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(wordfeat)

# Include directory for project headers
include_directories(${PROJECT_SOURCE_DIR}/include)

#
# Setting for Nvidia based
#
find_package(CUDA)

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

  set(EXECUTABLES feature_extract feature_extract_naive)
  foreach (TARGET ${EXECUTABLES})
    CUDA_ADD_EXECUTABLE(
      ${TARGET}
      ${PROJECT_SOURCE_DIR}/src/${TARGET}.cu
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
  endforeach ()

# 
# CTEST Part
#

# COMMENT OUT FOR NOW
#  enable_testing()


endif()
