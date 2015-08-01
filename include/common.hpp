#pragma once

//=====================================================================================================================
// Version number to include  
//
#define Wordfeat_VERSION_MAJOR @Wordfeat_VERSION_MAJOR@
#define Wordfeat_VERSION_MINOR @Wordfeat_VERSION_MINOR@

//=====================================================================================================================
// Headers to include 
//

//---------------------------------------------------------------------------------------------------------------------
// C++
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <list>
#include <map>
#include <sstream>
#include <string>
#include <vector>

//---------------------------------------------------------------------------------------------------------------------
// Google Library
#include "gflags/gflags.h"
#include "glog/logging.h"

#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_

//---------------------------------------------------------------------------------------------------------------------
// CUDA
#if defined(__CUDACC__)
    #include <cuda.h>
    #include <cuda_runtime.h>
#endif

//=====================================================================================================================
// Macros
//
#define NDEBUG

#ifdef NDEBUG
  #define wfeatAssert(condition, message)                                                           \
    do                                                                                              \
    {                                                                                               \
      if (!(condition))                                                                             \
      {                                                                                             \
        std::cerr << "Assertion failed: (" #condition "), function " << __FUNCTION__ << ", file "   \
                  << __FILE__  << ", line " << __LINE__ << ": " << message << std::endl;            \
          std::exit(EXIT_FAILURE);                                                                  \
      }                                                                                             \
    } while (0)
#else
    #define wfeatAssert(condition, message)
#endif

#define wfeatCheck(stmt)                                                       \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      LOG(ERROR) << "Failed to run stmt  " << #stmt;                           \
      LOG(ERROR) << "Got CUDA error ...  " << cudaGetErrorString(err);         \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define SSTR( x ) dynamic_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()

#define SEP_OP "--------------------"

#define INFOSEP(A) std::string(SEP_OP + std::string(A) + SEP_OP)

//---------------------------------------------------------------------------------------------------------------------
// Program Specified Macros
// 

#define IN_DIM        2
#define FEAT_DIM      19
#define FEAT_SIZE     3
#define WORD_WINDOW   5
#define WINDOW_RADIUS (WORD_WINDOW/2) 

#define PAD_NUM       (int) 0

//=====================================================================================================================
// Namespace
//

namespace wordfeat{
//
// String Error Output
  std::string wfeatStrerror(int errnum);

// 
// Global Initalization
void GlobalInit(int* pargc, char*** pargv);

} // namespace wordfeat

