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
// #define UNROLL_DATA_TO_MAT
//=======================================================================================
#define IN_DIM        2
#define FEAT_DIM      19
#define FEAT_SIZE     3
#define WORD_WINDOW   5
#define WINDOW_RADIUS WORD_WINDOW/2 

#define PAD_NUM       (int) 0
//=======================================================================================
// Assume that tile width is static for now
//  There will be 32 x 32 threads on each block and only 32 x THREAD_WIDTH number of threads 
//  will be actively working. We will use the first THREAD_WIDTH threads in each row to 
//  keep computation locality
//  
#define TILE_WIDTH    32
#define THREAD_WIDTH  (TILE_WIDTH - 2*WORD_WINDOW)


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
__global__ void extract_feat(int * inMat,  int N, int L, 
                             int * outFeat, int D, int S)
{
  // Shared memory for a block of threads
  //    There will be 32 x 32 x 2 cache for each block
  __shared__ int cache[TILE_WIDTH][TILE_WIDTH][IN_DIM];
  
  // Loading shared memory
  int tx = threadIdx.x; int ty = threadIdx.y; 
  int bx = blockIdx.x; int by = blockIdx.y;

  int row         = by * blockDim.y + ty;
  int col_out     = bx  * THREAD_WIDTH + tx; 
  int col_in      = col_out - WINDOW_RADIUS;
  
  if( row < N && col_in < L && col_in >= 0 )
  {
    int unroll_in   = row * L + col_in; 
    #pragma unroll
    for( int i = 0; i < IN_DIM; i++) cache[ty][tx][i] = inMat[ unroll_in + i];
  }
  __syncthreads();

  // Computation
  //=============================================================================
  // Feature extraction code
  if( tx < THREAD_WIDTH && col_out < L && row < N ) 
  {
    int cx = tx + WINDOW_RADIUS;
    if( cache[ty][cx - WINDOW_RADIUS][0] != PAD_NUM && cache[ty][cx - WINDOW_RADIUS][1] != PAD_NUM && 
        cache[ty][cx + WINDOW_RADIUS][0] != PAD_NUM && cache[ty][cx + WINDOW_RADIUS][1] != PAD_NUM )
    {
      int featIdx = row*L*D*S + col_out*D*S - S;

      // Feature (U00 - U04) extraction
      #pragma unroll
      for(int i = 0; i < WORD_WINDOW; i++)
      {
        featIdx = featIdx + S;
        outFeat[featIdx + 0] = cache[ty][cx - 1 + i + 0][0];
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
        outFeat[featIdx + 0] = cache[ty][cx - 1 + i + 0][1];
        outFeat[featIdx + 1] = PAD_NUM;
        outFeat[featIdx + 2] = PAD_NUM;
      }

      // Feature (U15 - U18) extraction 
      #pragma unroll
      for(int i = 0; i < 4; i++){
        featIdx = featIdx + S;
        outFeat[featIdx + 0] = cache[ty][cx - WINDOW_RADIUS + i + 0][1];
        outFeat[featIdx + 1] = cache[ty][cx - WINDOW_RADIUS + i + 1][1];
        outFeat[featIdx + 2] = PAD_NUM;
      }
      
      // Feature (U20 - U22) extraction
      #pragma unroll
      for(int i = 0; i < 3; i++){
        featIdx = featIdx + S;
        outFeat[featIdx + 0] = cache[ty][cx - WINDOW_RADIUS + i + 0][1];
        outFeat[featIdx + 1] = cache[ty][cx - WINDOW_RADIUS + i + 1][1];
        outFeat[featIdx + 2] = cache[ty][cx - WINDOW_RADIUS + i + 2][1];
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
      LOG(INFO) << wordDict[hostMat[ i * L * M + j * M + 0 ]] << " "
                << wordDict[hostMat[ i * L * M + j * M + 1 ]];
    }
  }
#endif

}

bool parseInput(ifstream &infile,  vector<vector<pair<int, int> > >&inData, 
    map<int, string> &wordDict, int wordWindow, int& maxSentLength)
{
  // Here notice that token starts from index = WORD_WINDOW - 1 
  // for case word window = 5, we have: 
  //    * 0 - <*>
  //    * 1 - <ss>
  //    * 2 - <s>
  //    * 3 - </s>
  //    * 4 - </ss>
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

    inData[sentNum].push_back( make_pair(tokenDict[word], tokenDict[tag]) );
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
    string pad;
    for (int j = 0; j < i; j++) pad += "s";

    wordDict[i]                 = "<" + pad + ">";
    wordDict[wordWindow - i]    = "</" + pad + ">";
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

int main(int argc, char * argv[])
{
  // Setup usage flags
  gflags::SetUsageMessage("command line message\n" "usage: wordfeat <command> <args>\n\n"
    "commands:\n");
  
  // Set default behavior as output to console
  FLAGS_alsologtostderr = 1;
  GlobalInit(&argc, &argv);

#ifdef INITIALIZATION
  DLOG(INFO) << DEBUG_HEAD; 
  DLOG(INFO) << "IN_FILE:     " << FLAGS_infile;
  DLOG(INFO) << "OUT_FILE:    " << FLAGS_outfile;
  // DLOG(INFO) << "WORD_WINDOW: " << FLAGS_window;
  DLOG(INFO) << DEBUG_TAIL;
#endif

  // Input error
  if( FLAGS_infile == "" || FLAGS_outfile == "")
    return 1;
  
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
  DLOG(INFO) << "N: " << N << ",M: " << M << ",D: " << D << ",S: " << S;

  // Host memory allocation
  int * hostMat; int * hostFeat;
  hostMat = (int *) malloc( N*L*M*sizeof(int) );
  hostFeat = (int *) malloc( N*L*D*S*sizeof(int) );

  LOG(INFO) << "Maximum sentence length: " << L;

  // Unroll data structure into a N X L X M matrix
  wfeatTime_start(CPU, "Unrolling input data structure");
  unroll_data_to_mat(hostMat, inData, L, M, wordDict);
  wfeatTime_stop(CPU, "Unrolling input data structure");

  // Allocate device memory
  int * deviceInMat = NULL; 
  int * deviceOutFeat = NULL; 
  wfeatCheck(cudaMalloc( (void **) &deviceInMat,   N*L*M*sizeof(int) ));
  wfeatCheck(cudaMalloc( (void **) &deviceOutFeat, N*L*D*S*sizeof(int) ));

  // Data transfer
  wfeatTime_start(GENERIC, "Transfer Data from CPU to GPU");
  wfeatCheck(cudaMemcpy( deviceInMat, hostMat, N*L*M*sizeof(int), cudaMemcpyHostToDevice));
  wfeatTime_stop(GENERIC, "Transfer Data from CPU to GPU");
  
  // Determining Grid and Block size for running kernel
  dim3 dimGrids( (N - 1) / TILE_WIDTH + 1, (L - 1) / TILE_WIDTH + 1, 1);
  dim3 dimBlocks( TILE_WIDTH, TILE_WIDTH, 1 );

  LOG(INFO) << "Start GPU computation.";
  wfeatTime_start(GPU, "Kernel Function: extract_feat");
  
  // Run Kernel function 
  extract_feat<<<dimGrids, dimBlocks>>>(deviceInMat,    L,  M,
                                        deviceOutFeat,  D,  S);

  wfeatTime_stop(GPU, "Kernel Function: extract_feat");
  LOG(INFO) << "Finished.";

  // Data transfer back
  wfeatTime_start(GENERIC, "Transfer Data from GPU to CPU");
  wfeatCheck(cudaMemcpy( hostFeat, deviceOutFeat, N*L*D*S*sizeof(int), cudaMemcpyDeviceToHost));
  wfeatTime_stop(GENERIC, "Transfer Data from GPU to CPU");

#define OUTPUT
#ifdef OUTPUT
  // Print output feature 
  for(int i = 0; i < N; i++){
    for(int j = 0; j < L; j++){
      int baseIdx = i*L*D*S + j*D*S;
      for(int k = 0; k < D; k++){
        int featIdx = baseIdx + k*S;

        for(int l = 0; l < S; l++){
          int elementIdx = featIdx + l;
          // if(hostFeat[elementIdx] == PAD_NUM)
          //   break;
          
          if( l == 0) cout << "(" << hostFeat[elementIdx] << "-"<< wordDict[hostFeat[elementIdx]] << ")";
          else cout << "/" << "(" << hostFeat[elementIdx] << "-"<< wordDict[hostFeat[elementIdx]] << ")";
        }

        // if(hostFeat[featIdx] != PAD_NUM)
          cout << endl;
      }
      // extra blank line
      // if(hostFeat[baseIdx] != PAD_NUM)
        cout << endl;
    } // end of L 
    cout << "============================================================" << endl;
  }
#endif

  // Free host memory
  free(hostMat);
  free(hostFeat);

  // Free device memory
  wfeatCheck(cudaFree(deviceInMat));
  wfeatCheck(cudaFree(deviceOutFeat));


  return 0;
}
