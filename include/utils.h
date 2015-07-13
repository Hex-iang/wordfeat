// Utility function
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
// CUDA
#if defined(__CUDACC__)
    #include <cuda.h>
    #include <cuda_runtime.h>
#endif

//===============================================================
// Macros
//
#define WFDEBUG

#ifdef WFDEBUG
    #define wfAssert(condition, message)                                                                  \
        do                                                                                                \
        {                                                                                                 \
            if (!(condition))                                                                             \
            {                                                                                             \
                std::cerr << "Assertion failed: (" #condition "), function " << __FUNCTION__ << ", file " \
                          << __FILE__  << ", line " << __LINE__ << ": " << message << std::endl;          \
                std::exit(EXIT_FAILURE);                                                                  \
            }                                                                                             \
        } while (0)

#undef WFDEBUG
#else
    #define wfAssert(condition, message)
#endif


//===============================================================
// Namespace
//

namespace wfInternal 
{
// String Error Output
#if defined(_WIN32)
    std::string wfStrerror(int errnum)
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
    std::string wfStrerror(int errnum)
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
    std::string wfStrerror(int errnum)
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
    std::string wfStrerror(int errnum)
    {
        std::string str;

        if (errnum)
        {
            str = strerror(errnum);
        }

        return str;
    }
#endif
} // namespace wfInternal


//===============================================================
// Timer
//

#if defined(__CUDACC__)
    #define wfTimerDeviceSynchronize() cudaDeviceSynchronize()
#else
    #define wfTimerDeviceSynchronize()
#endif

// Namespace because Windows.h causes errors
namespace wfInternal
{
#if defined(_WIN32)
    #include <Windows.h>

    // wfTimer class adapted from: https://bitbucket.org/ashwin/cudatimer
    class wfTimer
    {
    private:
        double        timerResolution;
        LARGE_INTEGER startTime;
        LARGE_INTEGER endTime;

    public:
        wfTimer::wfTimer()
        {
            LARGE_INTEGER freq;
            QueryPerformanceFrequency(&freq);
            timerResolution = 1.0 / freq.QuadPart;
        }

        void start()
        {
            wfTimerDeviceSynchronize();
            QueryPerformanceCounter(&startTime);
        }

        void stop()
        {
            wfTimerDeviceSynchronize();
            QueryPerformanceCounter(&endTime);
        }

        double value()
        {
            return (endTime.QuadPart - startTime.QuadPart) * timerResolution;
        }
    };
#elif defined(__APPLE__)
    #include <mach/mach_time.h>

    class wfTimer
    {
    private:
        uint64_t startTime;
        uint64_t endTime;

    public:
        void start()
        {
            wfTimerDeviceSynchronize();
            startTime = mach_absolute_time();
        }

        void stop()
        {
            wfTimerDeviceSynchronize();
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

    class wfTimer
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
            wfTimerDeviceSynchronize();
            startTime = getTime();
        }

        void stop()
        {
            wfTimerDeviceSynchronize();
            endTime = getTime();
        }

        double value()
        {
            return ((double) endTime - startTime) / NSEC_PER_SEC;
        }
    };
#endif
} // namespace wfInternal

enum wfTimeType
{
    Generic,
    GPU,
    Compute,
    Copy,
    wfTimeTypeINVALID // Keep this at the end
};

namespace wfInternal
{
    const char* wfTimeTypeStr[] =
    {
        "Generic",
        "GPU    ",
        "Compute",
        "Copy   ",
        "***InvalidTimeType***", // Keep this at the end
    };

    const char* wfTimeTypeToStr(const wfTimeType timeType)
    {
        return wfTimeTypeStr[timeType];
    }

    struct wfTimerInfo
    {
        wfTimeType  type;
        std::string message;
        wfTimer     timer;

        bool operator==(const wfTimerInfo& t2) const
        {
            return (type == t2.type && (0 == message.compare(t2.message)));
        }
    };

    typedef std::list<wfTimerInfo> wfTimerInfoList;

    wfTimerInfoList timerInfoList;
} // namespace wfInternal

void wfTime_start(const wfTimeType timeType, const std::string timeMessage)
{
    wfAssert(timeType >= Generic && timeType < wfTimeTypeINVALID, "Unrecognized wfTimeType value");

    wfInternal::wfTimer timer;
    timer.start();

    wfInternal::wfTimerInfo timerInfo = { timeType, timeMessage, timer };

    wfInternal::timerInfoList.push_front(timerInfo);
}

void wfTime_stop(const wfTimeType timeType, const std::string timeMessage)
{
    wfAssert(timeType >= Generic && timeType < wfTimeTypeINVALID, "Unrecognized wfTimeType value");

    const wfInternal::wfTimerInfo searchInfo = { timeType, timeMessage, wfInternal::wfTimer() };
    const wfInternal::wfTimerInfoList::iterator iter = std::find(wfInternal::timerInfoList.begin(), wfInternal::timerInfoList.end(), searchInfo);

    wfInternal::wfTimerInfo& timerInfo = *iter;

    wfAssert(searchInfo == timerInfo, "Could not find a corresponding wfTimerInfo struct registered by wfTime_start()");

    timerInfo.timer.stop();

    std::cout << "[" << wfInternal::wfTimeTypeToStr( timerInfo.type ) << "] ";
    std::cout << std::fixed << std::setprecision(9) << timerInfo.timer.value() << " ";
    std::cout << timerInfo.message << std::endl;

    wfInternal::timerInfoList.erase(iter);
}
