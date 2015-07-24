// Utility functions for cross plantform programming
// 
// Code Copy Right Specification
// 
// * Some part of this code is modified from: 
//      https://github.com/ashwin/coursera-heterogeneous.git 
//   
#pragma once

//===============================================================
// Headers to include 
//

//---------------------------------------------------------------
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
#include <sstream>
#include <string>
#include <vector>

//---------------------------------------------------------------
// Google Library
#include "gflags/gflags.h"
#include "glog/logging.h"

//---------------------------------------------------------------
// CUDA
#if defined(__CUDACC__)
    #include <cuda.h>
    #include <cuda_runtime.h>
#endif

//===============================================================
// Macros
//
#define NDEBUG

#ifdef NDEBUG
    #define wfeatAssert(condition, message)                                                                  \
        do                                                                                                \
        {                                                                                                 \
            if (!(condition))                                                                             \
            {                                                                                             \
                std::cerr << "Assertion failed: (" #condition "), function " << __FUNCTION__ << ", file " \
                          << __FILE__  << ", line " << __LINE__ << ": " << message << std::endl;          \
                std::exit(EXIT_FAILURE);                                                                  \
            }                                                                                             \
        } while (0)
#else
    #define wfeatAssert(condition, message)
#endif


//===============================================================
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


//===============================================================
// Timer
//

#if defined(__CUDACC__)
    #define wfeatTimerDeviceSynchronize() cudaDeviceSynchronize()
#else
    #define wfeatTimerDeviceSynchronize()
#endif

// Namespace because Windows.h causes errors
namespace wfeatInternal
{
#if defined(_WIN32)
    #include <Windows.h>

    class wfeatTimer
    {
    private:
        double        timerResolution;
        LARGE_INTEGER startTime;
        LARGE_INTEGER endTime;

    public:
        wfeatTimer::wfeatTimer()
        {
            LARGE_INTEGER freq;
            QueryPerformanceFrequency(&freq);
            timerResolution = 1.0 / freq.QuadPart;
        }

        void start()
        {
            wfeatTimerDeviceSynchronize();
            QueryPerformanceCounter(&startTime);
        }

        void stop()
        {
            wfeatTimerDeviceSynchronize();
            QueryPerformanceCounter(&endTime);
        }

        double value()
        {
            return (endTime.QuadPart - startTime.QuadPart) * timerResolution;
        }
    };
#elif defined(__APPLE__)
    #include <mach/mach_time.h>

    class wfeatTimer
    {
    private:
        uint64_t startTime;
        uint64_t endTime;

    public:
        void start()
        {
            wfeatTimerDeviceSynchronize();
            startTime = mach_absolute_time();
        }

        void stop()
        {
            wfeatTimerDeviceSynchronize();
            endTime = mach_absolute_time();
        }

        double value()
        {
            static mach_timebase_info_data_t tb;

            if (0 == tb.denom)
                (void) mach_timebase_info(&tb); // Calculate ratio of mach_absolute_time ticks to nanoseconds

            return ((double) endTime - startTime) * (tb.numer / tb.denom) / NSEC_PER_SEC;
        }
    };
#else
    #if defined(_POSIX_TIMERS) && _POSIX_TIMERS > 0
        #include <time.h>
    #else
        #include <sys/time.h>
    #endif

    #if !defined(NSEC_PER_SEC)
        #define NSEC_PER_SEC 1e9L
    #endif
    #if !defined(MSEC_PER_NSEC)
        #define MSEC_PER_NSEC (NSEC_PER_SEC / CLOCKS_PER_SEC)
    #endif

    class wfeatTimer
    {
    private:
        long startTime;
        long endTime;

        long getTime()
        {
            long time;
        #if defined(_POSIX_TIMERS) && _POSIX_TIMERS > 0
            struct timespec ts;

            if (0 == clock_gettime(CLOCK_MONOTONIC, &ts))
            {
                time  = NSEC_PER_SEC;
                time *= ts.tv_sec;
                time += ts.tv_nsec;
            }
        #else
            struct timeval tv;

            if (0 == gettimeofday(&tv, NULL))
            {
                time  = NSEC_PER_SEC;
                time *= tv.tv_sec;
                time += tv.tv_usec * MSEC_PER_NSEC;
            }
        #endif
            return time;
        }

    public:
        void start()
        {
            wfeatTimerDeviceSynchronize();
            startTime = getTime();
        }

        void stop()
        {
            wfeatTimerDeviceSynchronize();
            endTime = getTime();
        }

        double value()
        {
            return ((double) endTime - startTime) / NSEC_PER_SEC;
        }
    };
#endif
} // namespace wfeatInternal

enum wfeatTimeType
{
    GENERIC,
    GPU,
    CPU,
    TRANSFER,
    wfeatTimeTypeINVALID // Keep this at the end
};

namespace wfeatInternal
{
    const char* wfeatTimeTypeStr[] =
    {
        "GENERIC ",
        "GPU ONLY",
        "CPU ONLY",
        "TRANSFER",
        "***InvalidTimeType***", // Keep this at the end
    };

    const char* wfeatTimeTypeToStr(const wfeatTimeType timeType)
    {
        return wfeatTimeTypeStr[timeType];
    }

    struct wfeatTimerInfo
    {
        wfeatTimeType  type;
        std::string message;
        wfeatTimer     timer;

        bool operator==(const wfeatTimerInfo& t2) const
        {
            return (type == t2.type && (0 == message.compare(t2.message)));
        }
    };

    typedef std::vector<wfeatTimerInfo> wfeatTimerInfoList;

    wfeatTimerInfoList timerInfoList;
} // namespace wfeatInternal

void wfeatTime_start(const wfeatTimeType timeType, const std::string timeMessage)
{
    wfeatAssert(timeType >= GENERIC && timeType < wfeatTimeTypeINVALID, "Unrecognized wfeatTimeType value");

    wfeatInternal::wfeatTimer timer;
    timer.start();

    wfeatInternal::wfeatTimerInfo timerInfo = { timeType, timeMessage, timer };

    wfeatInternal::timerInfoList.push_back(timerInfo);

    DLOG(INFO) << "Pushing Start Timer Finished. Vector Size: " << wfeatInternal::timerInfoList.size();

}

void wfeatTime_stop(const wfeatTimeType timeType, const std::string timeMessage)
{
    wfeatAssert(timeType >= GENERIC && timeType < wfeatTimeTypeINVALID, "Unrecognized wfeatTimeType value");

    const wfeatInternal::wfeatTimerInfo searchInfo = { timeType, timeMessage, wfeatInternal::wfeatTimer() };
    const wfeatInternal::wfeatTimerInfoList::iterator iter = std::find(wfeatInternal::timerInfoList.begin(), wfeatInternal::timerInfoList.end(), searchInfo);

    wfeatInternal::wfeatTimerInfo& timerInfo = *iter;

    DLOG(INFO) << "Retrive Start Timer Finished. Vector Size: " << wfeatInternal::timerInfoList.size();

    wfeatAssert(searchInfo == timerInfo, "Could not find a corresponding wfeatTimerInfo struct registered by wfeatTime_start()");

    timerInfo.timer.stop();

    std::cout << "[" << wfeatInternal::wfeatTimeTypeToStr( timerInfo.type ) << "] ";
    std::cout << std::fixed << std::setprecision(9) << timerInfo.timer.value() << " ";
    std::cout << timerInfo.message << std::endl;

    wfeatInternal::timerInfoList.erase(iter);
}
