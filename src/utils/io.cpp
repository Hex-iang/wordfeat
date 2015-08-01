#include <utils/io.hpp>

using namespace std;

namespace wordfeat
{
#if defined(_WIN32)
  #include <Windows.h>

  wfeatTimer::wfeatTimer()
  {
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    timerResolution = 1.0 / freq.QuadPart;
  }

  void wfeatTimer::start()
  {
    wfeatTimerDeviceSynchronize();
    QueryPerformanceCounter(&startTime);
  }

  void wfeatTimer::stop()
  {
    wfeatTimerDeviceSynchronize();
    QueryPerformanceCounter(&endTime);
  }

  double wfeatTimer::value()
  {
    return (endTime.QuadPart - startTime.QuadPart) * timerResolution;
  }

#elif defined(__APPLE__)
  #include <mach/mach_time.h>

  void wfeatTimer::start()
  {
    wfeatTimerDeviceSynchronize();
    startTime = mach_absolute_time();
  }

  void wfeatTimer::stop()
  {
    wfeatTimerDeviceSynchronize();
    endTime = mach_absolute_time();
  }

  double wfeatTimer::value()
  {
    static mach_timebase_info_data_t tb;

    if (0 == tb.denom)
      // Calculate ratio of mach_absolute_time ticks to nanoseconds
      (void) mach_timebase_info(&tb);

    return ((double) endTime - startTime) * (tb.numer / tb.denom) / NSEC_PER_SEC;
  }
  
#else
  long wfeatTimer::getTime()
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

  void wfeatTimer::start()
  {
    wfeatTimerDeviceSynchronize();
    startTime = getTime();
  }

  void wfeatTimer::stop()
  {
    wfeatTimerDeviceSynchronize();
    endTime = getTime();
  }

  double wfeatTimer::value()
  {
    return ((double) endTime - startTime) / NSEC_PER_SEC;
  }
#endif
} // namespace wordfeat


namespace wordfeat
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
  
  wfeatTimerInfoList timerInfoList;
} // namespace wordfeat

void wfeatTime_start(const wfeatTimeType timeType, const std::string timeMessage)
{
  wfeatAssert(timeType >= GENERIC && timeType < wfeatTimeTypeINVALID, "Unrecognized wfeatTimeType value");

  wordfeat::wfeatTimer timer;
  timer.start();

  wordfeat::wfeatTimerInfo timerInfo = { timeType, timeMessage, timer };

  wordfeat::timerInfoList.push_back(timerInfo);

  // DLOG(INFO) << "Pushing Timer Finished. Timer List Size: " << wordfeat::timerInfoList.size();

}

void wfeatTime_stop(const wfeatTimeType timeType, const std::string timeMessage)
{
  wfeatAssert(timeType >= GENERIC && timeType < wfeatTimeTypeINVALID, "Unrecognized wfeatTimeType value");

  const wordfeat::wfeatTimerInfo searchInfo = { timeType, timeMessage, wordfeat::wfeatTimer() };
  const wordfeat::wfeatTimerInfoList::iterator iter = std::find(wordfeat::timerInfoList.begin(), wordfeat::timerInfoList.end(), searchInfo);

  wordfeat::wfeatTimerInfo& timerInfo = *iter;

  wfeatAssert(searchInfo == timerInfo, "Could not find a corresponding wfeatTimerInfo struct registered by wfeatTime_start()");

  timerInfo.timer.stop();
  
  LOG(INFO) << "[" << wordfeat::wfeatTimeTypeToStr( timerInfo.type ) << "] "
            << std::fixed << std::setprecision(9) << timerInfo.timer.value() << " "
            << timerInfo.message << std::endl;

  // DLOG(INFO) << "Retrive Timer Finished. Timer List Size: " << wordfeat::timerInfoList.size();

  wordfeat::timerInfoList.erase(iter);
}

//=======================================================================================
// Define Macro for debugging different procedure 
//
// #define TOKENIZE_PROC
// #define UNROLL_DATA_TO_MAT


//=======================================================================================
// Unroll function: flatten the input data structure to a flat matrix 
//---------------------------------------------------------------------------------------
void unroll_data_to_mat(int * &hostMat, vector<vector<pair<int, int> > >&inData, int L, int M, map<int, string>& wordDict)
{
  // Calculate and allocate host memory 
  for( int i = 0 ; i < (int) inData.size(); i++){
    for ( int j =0 ; j < L; j++ )
    {
      if( j >= (int) inData[i].size() )
      {
        // Padding for sentence 
        hostMat[ i * L * M + j * M + 0 ] = PAD_NUM;
        hostMat[ i * L * M + j * M + 1 ] = PAD_NUM;
      }else{
        // Assignment value
        hostMat[ i * L * M + j * M + 0 ] = inData[i][j].first;
        hostMat[ i * L * M + j * M + 1 ] = inData[i][j].second;
      }
    }
  }

#ifdef UNROLL_DATA_TO_MAT
  // Sanity check for unrolling
  for( int i = 0 ; i < (int) inData.size(); i++){
  //   for ( int j =0 ; j < inData[i].size(); j++ )
    for ( int j =0 ; j < L; j++ )
    {
      LOG(INFO) << "(" << wordDict[hostMat[ i * L * M + j * M + 0 ]] << "," << hostMat[ i * L * M + j * M + 0 ] << ")-("
                       << wordDict[hostMat[ i * L * M + j * M + 1 ]] << "," << hostMat[ i * L * M + j * M + 1 ] << ")";
    }
  }
#endif

}

//=======================================================================================
// parseInput function: read input file and output a vector for storing input 
//---------------------------------------------------------------------------------------
bool parseInput(ifstream &infile,  vector<vector<pair<int, int> > >&inData, 
    map<int, string> &wordDict, int wordWindow, int& maxSentLength)
{
  int tokenNum    = wordWindow;
  int wordRadius  = wordWindow / 2;
  int sentNum     = 0;
  
  map<string, int> tokenDict;
  string word, tag; 
  
  // Init by pushing in a vector
  inData.push_back( vector<pair<int, int> >() );
  while( infile >> word >> tag )
  { 
    // Tokenize word or tag if there isn't any previously
    if ( tokenDict.find(word) == tokenDict.end() ){
      tokenDict[word] = tokenNum++;
      
#ifdef TOKENIZE_PROC
      DLOG(INFO) << "Tokenize word \"" << word << "\" as " << tokenDict[word];
#endif

    }

    if ( tokenDict.find(tag) == tokenDict.end() ){
      tokenDict[tag]  = tokenNum++;

#ifdef TOKENIZE_PROC
      DLOG(INFO) << "Tokenize tag \"" << tag << "\" as " << tokenDict[tag];
#endif

    }
    
    // Put in the word and its tag
    inData[sentNum].push_back( make_pair(tokenDict[word], tokenDict[tag]) );
    
    //------------------------------------------------------------------------------------
    // At the end of the a sentence, insert place holder tokens
    if( word == "." && tag == "."){
      // Append place holder token
      for (int i = 1; i <= wordRadius; i++){
        inData[sentNum].insert( inData[sentNum].begin(), make_pair(i, i) );
        inData[sentNum].push_back( make_pair(wordWindow-i, wordWindow-i) );
      } 
      // update maximum sentence length
      maxSentLength = (maxSentLength > (int) inData[sentNum].size())? maxSentLength: inData[sentNum].size();

      inData.push_back( vector<pair<int, int> >() );
      sentNum ++;
    }

  }
  inData.pop_back();

  //--------------------------------------------------------------------------------------------------
  // Swap key value for map
  for (  map<string, int>::iterator it = tokenDict.begin(); it != tokenDict.end(); it++ ){
    wordDict[it->second] = it->first;
  }
  // Free the space of token dictionary
  tokenDict.clear();
  
  //--------------------------------------------------------------------------------------------------
  // Put in <ss> <s> </s> <ss> to token dictionary 
  //    as well as sentence padding "<*>"
  wordDict[0] = string("<*>");
  for (int i = 1; i <= wordRadius; i++){
    wordDict[i]                 = "_B-" + SSTR(i);
    wordDict[wordWindow - i]    = "_B+" + SSTR(i);
  }

#ifdef TOKENIZE_PROC
  DLOG(INFO) << DEBUG_HEAD;
  DLOG(INFO) << "Total number of sentences: " << inData.size();
  DLOG(INFO) << DEBUG_TAIL;

  // Print the recovered sentences
  for( vector<vector<pair<int, int> > >::iterator it=inData.begin(); it != inData.end(); it++){
    for ( vector<pair<int, int> >::iterator itt = it->begin(); itt != it->end(); itt++ )
    {
      cout << wordDict[itt->first] << " " << wordDict[itt->second] << endl;
    }
  }
#endif  

  return true;

}


//=======================================================================================
// outputData function: output data to stdout 
//---------------------------------------------------------------------------------------
void outputData(map<int, string>& wordDict, int * hostFeat, 
                int N, int L, int D, int S)
{
  static int featSet[] = {
    0,  1,  2,  3,  4,  5,  6,
    10, 11, 12, 13, 14, 15, 16,
    17, 18, 20, 21, 22
  };

  {
    // Print output feature into output file 
    for(int i = 0; i < N; i++){
      for(int j = 0; j < L; j++){
        // Base index for feature in current sentence
        int baseIdx = i*L*D*S + j*D*S;

        for(int k = 0; k < D; k++){
          // Feature index for feature dimension
          int featIdx = baseIdx + k*S;
        
          // Print out all dimensions of feature 
          for(int l = 0; l < S; l++){
            // Element index for the elements in a single dimension of feature 
            int elementIdx = featIdx + l;
            // Escape PADDING
            if(hostFeat[elementIdx] == PAD_NUM) break;
          
            if( l == 0 )
            { 
              // For first dimension of feature, output prefix
              cout << "FEAT U" << setfill('0') << setw(2) 
                      << featSet[k] << ":" << wordDict[hostFeat[elementIdx]];
            }else{ 
              // Fot the rest dimension of feature, output separator in features
              cout << "/" << wordDict[hostFeat[elementIdx]];
            }
          } 
          // Escape PADDING
          if(hostFeat[featIdx] != PAD_NUM) cout << endl;
        } // end of D loop

        // Escape PADDING
        if(hostFeat[baseIdx] != PAD_NUM)   cout << "FEAT B"<< endl;
      } // end of L loop
      // Output a separator at the end of sentence
      cout << endl;
    }
  }
}

//=======================================================================================
// outputData function: output data to file 
//---------------------------------------------------------------------------------------
void outputData(map<int, string>& wordDict, int * hostFeat, 
                int N, int L, int D, int S, string& filename)
{
  static int featSet[] = {
    0,  1,  2,  3,  4,  5,  6,
    10, 11, 12, 13, 14, 15, 16,
    17, 18, 20, 21, 22
  };

  ofstream outFile; 
  outFile.open(filename.c_str());

  // Write to outFile if it is open
  if( outFile.is_open() )
  {
    // Print output feature into output file 
    for(int i = 0; i < N; i++){
      for(int j = 0; j < L; j++){
        // Base index for feature in current sentence
        int baseIdx = i*L*D*S + j*D*S;

        for(int k = 0; k < D; k++){
          // Feature index for feature dimension
          int featIdx = baseIdx + k*S;
        
          // Print out all dimensions of feature 
          for(int l = 0; l < S; l++){
            // Element index for the elements in a single dimension of feature 
            int elementIdx = featIdx + l;
            // Escape PADDING
            if(hostFeat[elementIdx] == PAD_NUM) break;
          
            if( l == 0 )
            { 
              // For first dimension of feature, output prefix
              outFile << "FEAT U" << setfill('0') << setw(2) 
                      << featSet[k] << ":" << wordDict[hostFeat[elementIdx]];
            }else{ 
              // Fot the rest dimension of feature, output separator in features
              outFile << "/" << wordDict[hostFeat[elementIdx]];
            }
          } 
          // Escape PADDING
          if(hostFeat[featIdx] != PAD_NUM) outFile << endl;
        } // end of D loop

        // Escape PADDING
        if(hostFeat[baseIdx] != PAD_NUM)   outFile << "FEAT B"<< endl;
      } // end of L loop
      // Output a separator at the end of sentence
      outFile << endl;
    }
  }
}
