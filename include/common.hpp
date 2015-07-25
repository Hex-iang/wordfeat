#pragma once

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
      LOG(ERROR) << "Got CUDA error ...  " << cudaGetErrorString(err));        \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define DEBUG_HEAD "==============DEBUG============"
#define DEBUG_TAIL "==============END=============="
//=====================================================================================================================
// Namespace
//

namespace wfeatInternal 
{
// String Error Output
#if defined(_WIN32)
  std::string wfeatStrerror(int errnum)
  {
    std::string str;
    char buffer[1024];

    if (errnum)
    {
      (void) strerror_s(buffer, sizeof(buffer), errnum);
      str = buffer;
    }

    return str;
  }
#elif (_POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600 || __APPLE__) && ! _GNU_SOURCE
  std::string wfeatStrerror(int errnum)
  {
    std::string str;
    char buffer[1024];

    if (errnum)
    {
      (void) strerror_r(errnum, buffer, sizeof(buffer));
      str = buffer;
    }

    return str;
  }
#elif defined(_GNU_SOURCE)
  std::string wfeatStrerror(int errnum)
  {
    std::string str;
    char buffer[1024];

    if (errnum)
    {
      str = strerror_r(errnum, buffer, sizeof(buffer));
    }

    return str;
  }
#else
  std::string wfeatStrerror(int errnum)
  {
    std::string str;

    if (errnum)
    {
      str = strerror(errnum);
    }

    return str;
  }
#endif
} // namespace wfeatInternal

//=====================================================================================================================
// 
// Global Initalization

namespace wordfeat{

void GlobalInit(int* pargc, char*** pargv) {
  // Google flags.
  ::gflags::ParseCommandLineFlags(pargc, pargv, true);
  // Google logging.
  ::google::InitGoogleLogging(*(pargv)[0]);
  // Provide a backtrace on segfault.
  ::google::InstallFailureSignalHandler();
}

} // namespace wordfeat
