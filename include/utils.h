// Utility functions for cross plantform programming
// 
// Code Copy Right Specification
// 
// * Some part of this code is modified from: 
//      https://github.com/ashwin/coursera-heterogeneous.git 
//   
#pragma once

//=====================================================================================================================
// Common header
#include <common.h>

//=====================================================================================================================
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
        // Calculate ratio of mach_absolute_time ticks to nanoseconds
        (void) mach_timebase_info(&tb);

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

  DLOG(INFO) << "Pushing Timer Finished. Timer List Size: " << wfeatInternal::timerInfoList.size();

}

void wfeatTime_stop(const wfeatTimeType timeType, const std::string timeMessage)
{
  wfeatAssert(timeType >= GENERIC && timeType < wfeatTimeTypeINVALID, "Unrecognized wfeatTimeType value");

  const wfeatInternal::wfeatTimerInfo searchInfo = { timeType, timeMessage, wfeatInternal::wfeatTimer() };
  const wfeatInternal::wfeatTimerInfoList::iterator iter = std::find(wfeatInternal::timerInfoList.begin(), wfeatInternal::timerInfoList.end(), searchInfo);

  wfeatInternal::wfeatTimerInfo& timerInfo = *iter;

  wfeatAssert(searchInfo == timerInfo, "Could not find a corresponding wfeatTimerInfo struct registered by wfeatTime_start()");

  timerInfo.timer.stop();
  
  LOG(INFO) << "[" << wfeatInternal::wfeatTimeTypeToStr( timerInfo.type ) << "] "
            << std::fixed << std::setprecision(9) << timerInfo.timer.value() << " "
            << timerInfo.message << std::endl;

  DLOG(INFO) << "Retrive Timer Finished. Timer List Size: " << wfeatInternal::timerInfoList.size();

  wfeatInternal::timerInfoList.erase(iter);
}
