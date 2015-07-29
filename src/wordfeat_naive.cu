#include <common.hpp>
#include <utils.hpp>

DEFINE_string(outfile, "",
              "output file path   [REQUIRED]  (with extracted word feature)" );

DEFINE_string(infile, "",
              "input file path    [REQUIRED]  (input file with POS tagged)" );

// @@ Currently, word window size is fixed as 2
// DEFINE_int32(window, 2, 
//               "word window size   [DEFAULT:2] (word window size for feature)" );

//=======================================================================================
// MACRO for debugging different procedure 

#define INITIALZATION
// #define TOKENIZE_PROC
// #define PRINT_CACHE
// #define UNROLL_DATA_TO_MAT
// #define OUTPUT_FEATS_TO_CONSOLE 

//=======================================================================================
#define IN_DIM        2
#define FEAT_DIM      19
#define FEAT_SIZE     3
#define WORD_WINDOW   5
#define WINDOW_RADIUS (WORD_WINDOW/2) 

#define PAD_NUM       (int) 0
//=======================================================================================
// Assume that tile width is static for now
//  There will be 32 x 32 threads on each block and only 32 x THREAD_WIDTH number of threads 
//  will be actively working. We will use the first THREAD_WIDTH threads in each row to 
//  keep computation locality
//  
#define TILE_WIDTH    32
#define THREAD_WIDTH  (TILE_WIDTH - 2*WINDOW_RADIUS)


//=======================================================================================
// Name space 
using namespace wordfeat;
using namespace std;

//=======================================================================================
// Kernal function: Extract feature for a single sentence
//---------------------------------------------------------------------------------------
// Assume that input sentence is a matrix of N x L x M
// where  
//        N: the number of sentences
//        L: the length in words of a specified sentence
//        M: the size of raw feature of each word (IN_DIM in this case)
// The output is a feature matrix of N x L x D x S
// where  
//        L: the length in words of a specified sentence
//        D: the number of dimensions of feature
//        S: the size of each feature dimension to pre-allocated 
__global__ void extract_feat(int * inMat,   int N, int L, int M,
                             int * outFeat, int D, int S)
{
  // Shared memory for a block of threads
  //    There will be 32 x 32 x 2 cache for each block
  __shared__ int cache[TILE_WIDTH][TILE_WIDTH][IN_DIM];
  
  // Loading shared memory
  int tx = threadIdx.x; int ty = threadIdx.y; 
  int bx = blockIdx.x;  int by = blockIdx.y;

  int row         = by * blockDim.y + ty;
  int col_out     = bx * THREAD_WIDTH + tx; 
  int col_in      = col_out - WINDOW_RADIUS;
  
  if( row < N && col_in < L && col_in >= 0 )
  {
    int unroll_in   = row * L * M + col_in * M;
    #pragma unroll
    for( int i = 0; i < IN_DIM; i++) cache[ty][tx][i] = inMat[ unroll_in + i];
  }
  __syncthreads();

#ifdef PRINT_CACHE
  if( tx == 0 && ty == 0 )
  {
    // Specify a row in cache to print
    int c_row = 0; 
    for( int j = 0; j < TILE_WIDTH; j ++)
    {
      for( int k = 0; k < IN_DIM; k ++)        
      {
        printf("bx = %d, by = %d, cache[%d][%d][%d] = %d\n", 
                    bx, by, c_row, j, k, cache[c_row][j][k] );
      }
    }
  }
#endif

  // Computation
  //=============================================================================
  // Feature extraction code
  if( tx < THREAD_WIDTH && col_out < L && row < N ) 
  {
    // Index problem
    int cx = tx + WINDOW_RADIUS;
    if( cache[ty][tx][0] != PAD_NUM                   && cache[ty][tx][1] != PAD_NUM                   && 
        cache[ty][tx + 2*WINDOW_RADIUS][0] != PAD_NUM && cache[ty][tx + 2*WINDOW_RADIUS][1] != PAD_NUM )
    {
      int featIdx = row*L*D*S + col_out*D*S;

      // Feature (U00 - U04) extraction
      #pragma unroll
      for(int i = 0; i < WORD_WINDOW; i++)
      {
        if( i != 0) featIdx = featIdx + S;

        outFeat[featIdx + 0] = cache[ty][tx + i + 0][0];
        outFeat[featIdx + 1] = PAD_NUM;
        outFeat[featIdx + 2] = PAD_NUM;
      }

      // Feature (U05 - U06) extraction
      #pragma unroll
      for(int i = 0; i < 2; i++)
      {
        featIdx = featIdx + S;
        outFeat[featIdx + 0] = cache[ty][cx - 1 + i + 0][0];
        outFeat[featIdx + 1] = cache[ty][cx - 1 + i + 1][0];
        outFeat[featIdx + 2] = PAD_NUM;
      }

      // Feature (U10 - U14) extraction
      #pragma unroll
      for(int i = 0; i < WORD_WINDOW; i++)
      {
        featIdx = featIdx + S;
        outFeat[featIdx + 0] = cache[ty][tx + i + 0][1];
        outFeat[featIdx + 1] = PAD_NUM;
        outFeat[featIdx + 2] = PAD_NUM;
      }

      // Feature (U15 - U18) extraction 
      #pragma unroll
      for(int i = 0; i < 4; i++){
        featIdx = featIdx + S;
        outFeat[featIdx + 0] = cache[ty][tx + i + 0][1];
        outFeat[featIdx + 1] = cache[ty][tx + i + 1][1];
        outFeat[featIdx + 2] = PAD_NUM;
      }
      
      // Feature (U20 - U22) extraction
      #pragma unroll
      for(int i = 0; i < 3; i++){
        featIdx = featIdx + S;
        outFeat[featIdx + 0] = cache[ty][tx + i + 0][1];
        outFeat[featIdx + 1] = cache[ty][tx + i + 1][1];
        outFeat[featIdx + 2] = cache[ty][tx + i + 2][1];
      }
    }
  }
  __syncthreads();
}

//=======================================================================================
// Unroll function: flatten the input data structure to a flat matrix 
//---------------------------------------------------------------------------------------
void unroll_data_to_mat(int * &hostMat, vector<vector<pair<int, int> > >&inData, int L, int M, map<int, string>& wordDict)
{
  // Calculate and allocate host memory 
  for( int i = 0 ; i < inData.size(); i++){
    for ( int j =0 ; j < L; j++ )
    {
      if( j >= inData[i].size() )
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
  for( int i = 0 ; i < inData.size(); i++){
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
  // Here notice that token starts from index = WORD_WINDOW - 1 
  // for case word window = 5, we have: 
  //    * 0 - <*>
  //    * 1 - _B-2
  //    * 2 - _B-1
  //    * 3 - _B+1
  //    * 4 - _B+2
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
      maxSentLength = (maxSentLength > inData[sentNum].size())? maxSentLength: inData[sentNum].size();

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

int main(int argc, char * argv[])
{
  // Setup usage flags
  gflags::SetUsageMessage("command line message\n" "usage: wordfeat <command> <args>\n\n"
    "commands:\n");
  
  // Set default behavior as output to console
  FLAGS_alsologtostderr = 1;
  GlobalInit(&argc, &argv);

#ifdef INITIALIZATION
  DLOG(WARNING) << "DEBUG INFORMATION"; 
  DLOG(WARNING) << "IN_FILE:     " << FLAGS_infile;
  DLOG(WARNING) << "OUT_FILE:    " << FLAGS_outfile;
#endif

  // Input error
  wfeatAssert( (FLAGS_infile != "" && FLAGS_outfile != ""), "There is non-specified input/output file");
  
  // To read from input file 
  ifstream infile( FLAGS_infile.c_str() );
  
  // Variable for parsing the input data structure
  vector<vector< pair<int, int> > > inData;
  map<int, string> wordDict;
  // int wordWindow = FLAGS_window;
  int wordWindow = WORD_WINDOW;
  int L = 0;
  
  // Timer start
  wfeatTime_start(CPU, "Loading input data");
  // Parse input file into specified data structure
  parseInput(infile, inData, wordDict, wordWindow, L);
  // Timer stop
  wfeatTime_stop(CPU, "Loading input data");


  // Compute and allocate host memory
  int N = inData.size(); int M = IN_DIM; 
  int D = FEAT_DIM; int S = FEAT_SIZE;

  // Host memory allocation
  int * hostMat; int * hostFeat;
  int inMatSize   = N*L*M*sizeof(int);
  int outFeatSize = N*L*D*S*sizeof(int); 
  
  // Memory utility information
  LOG(INFO) << "[ Data Structure Dimension and Memory Utility Status ]"; 
  LOG(INFO) << "inMat dims:    " << N << " x " << L << " x " << M;
  LOG(INFO) << "inMat size:    " << inMatSize << " bytes.";
  LOG(INFO) << "outFeat dims:  " << N << " x " << L << " x " << D << " x " << S;
  LOG(INFO) << "outFeat size:  " << outFeatSize << " bytes.";

  wfeatTime_start(CPU, "Allocate host memory...");
  hostMat = (int *) malloc( inMatSize );
  hostFeat = (int *) malloc( outFeatSize );
  wfeatTime_stop(CPU, "Allocate host memory...");

  // Unroll data structure into a N X L X M matrix
  wfeatTime_start(CPU, "Unrolling input data structure");
  unroll_data_to_mat(hostMat, inData, L, M, wordDict);
  wfeatTime_stop(CPU, "Unrolling input data structure");

  // Allocate device memory
  int * deviceInMat = NULL; 
  int * deviceOutFeat = NULL; 

  wfeatTime_start(CPU, "Allocate device memory...");
  // Allocate memory
  wfeatCheck(cudaMalloc( (void **) &deviceInMat,   inMatSize ));
  wfeatCheck(cudaMalloc( (void **) &deviceOutFeat, outFeatSize ));
  // Set up initial value for output 
  wfeatCheck(cudaMemset( deviceOutFeat, 0, N*L*D*S*sizeof(int) ));
  wfeatTime_stop(CPU, "Allocate device memory...");

  // Data transfer
  wfeatTime_start(GENERIC, "Transfer Data from CPU to GPU");
  wfeatCheck(cudaMemcpy( deviceInMat, hostMat, N*L*M*sizeof(int), cudaMemcpyHostToDevice));
  wfeatTime_stop(GENERIC, "Transfer Data from CPU to GPU");
  
  // Determining Grid and Block size for running kernel
  int gridX = (N - 1) / TILE_WIDTH + 1;
  int gridY = (L - 1) / TILE_WIDTH + 1;
  
  dim3 dimGrids( gridY, gridX, 1);
  dim3 dimBlocks( TILE_WIDTH, TILE_WIDTH, 1);

  DLOG(INFO) << "[ Grid and Block status ]";
  DLOG(INFO) << "dimGrids.x  = " << dimGrids.x  << ", dimGrids.y  = " << dimGrids.y;
  DLOG(INFO) << "dimBlocks.x = " << dimBlocks.x << ", dimBlocks.y = " << dimBlocks.y;

  // Run Kernel function 
  LOG(INFO) << "[ Start GPU computation ]";
  wfeatTime_start(GPU, "Kernel Function: extract_feat");
  
  extract_feat<<<dimGrids, dimBlocks>>>(deviceInMat,    N,  L,  M,
                                        deviceOutFeat,  D,  S);

  wfeatTime_stop(GPU, "Kernel Function: extract_feat");
  LOG(INFO) << "[ Finished. ]";

  // Data transfer back
  wfeatTime_start(GENERIC, "Transfer Data from GPU to CPU");
  wfeatCheck(cudaMemcpy( hostFeat, deviceOutFeat, N*L*D*S*sizeof(int), cudaMemcpyDeviceToHost));
  wfeatTime_stop(GENERIC, "Transfer Data from GPU to CPU");

  // Nasty print for output structure from device
  wfeatTime_start(CPU, "Output Data.");

#ifdef OUTPUT_FEATS_TO_CONSOLE 
  outputData(wordDict, hostFeat, N, L, D, S);
#else
  outputData(wordDict, hostFeat, N, L, D, S, FLAGS_outfile);
#endif

  wfeatTime_stop(CPU, "Output Data."); 

  
  // Free host memory
  free(hostMat);
  free(hostFeat);

  // Free device memory
  wfeatCheck(cudaFree(deviceInMat));
  wfeatCheck(cudaFree(deviceOutFeat));

  return 0;
}
