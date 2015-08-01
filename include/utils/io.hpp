#pragma once
// Utility functions for cross plantform programming
// 
// Code Copy Right Specification
// 
// * Some part of this code is modified from: 
//      https://github.com/ashwin/coursera-heterogeneous.git 
//   

//=====================================================================================================================
// Common header
#include <common.hpp>

//=====================================================================================================================
// Timer
//

#if defined(__CUDACC__)
  #define wfeatTimerDeviceSynchronize() cudaDeviceSynchronize()
#else
  #define wfeatTimerDeviceSynchronize()
#endif

// Namespace because Windows.h causes errors
namespace wordfeat
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
    wfeatTimer::wfeatTimer();
    void start();
    void stop();
    double value();
  };
#elif defined(__APPLE__)
  #include <mach/mach_time.h>

  class wfeatTimer
  {
  private:
    uint64_t startTime;
    uint64_t endTime;

  public:
    void start();
    void stop();
    double value();
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

    long getTime();
  public:
    void start();
    void stop();
    double value();
  };
#endif
} // namespace wordfeat

enum wfeatTimeType
{
  GENERIC,
  GPU,
  CPU,
  TRANSFER,
  wfeatTimeTypeINVALID // Keep this at the end
};

namespace wordfeat
{

  const char* wfeatTimeTypeToStr(const wfeatTimeType timeType);

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

} // namespace wordfeat

void wfeatTime_start(const wfeatTimeType timeType, const std::string timeMessage);

void wfeatTime_stop(const wfeatTimeType timeType, const std::string timeMessage);

//=======================================================================================
// Unroll function: flatten the input data structure to a flat matrix 
//---------------------------------------------------------------------------------------
void unroll_data_to_mat(int * &hostMat, std::vector<std::vector<std::pair<int, int> > >&inData, 
                        int L, int M, std::map<int, std::string>& wordDict);


//=======================================================================================
// parseInput function: read input file and output a vector for storing input
// 
// => Here notice that token starts from index = WORD_WINDOW - 1 
//    for case word window = 5, we have: 
//      * 0 - <*>
//      * 1 - _B-2
//      * 2 - _B-1
//      * 3 - _B+1
//      * 4 - _B+2 
//---------------------------------------------------------------------------------------
bool parseInput(std::ifstream &infile,  std::vector< std::vector< std::pair<int, int> > >&inData, 
                std::map<int, std::string> &wordDict, int wordWindow, int& maxSentLength);

//=======================================================================================
// outputData function: output data to stdout 
//---------------------------------------------------------------------------------------
void outputData(std::map<int, std::string>& wordDict, int * hostFeat, 
                int N, int L, int D, int S);

//=======================================================================================
// outputData function: output data to file 
//---------------------------------------------------------------------------------------
void outputData(std::map<int, std::string>& wordDict, int * hostFeat, 
                int N, int L, int D, int S, std::string& filename);
