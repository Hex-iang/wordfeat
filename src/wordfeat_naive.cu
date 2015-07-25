#include <common.hpp>
#include <utils.hpp>

DEFINE_string(outfile, "",
              "output file path   [REQUIRED]  (with extracted word feature)" );

DEFINE_string(infile, "",
              "input file path    [REQUIRED]  (input file with POS tagged)" );

DEFINE_int32(window, 2, 
              "word window size   [DEFAULT:2] (word window size for feature)" );

//=======================================================================================
// MACRO for debugging different procedure 

#define INITIALZATION
// #define TOKENIZE_PROC

//=======================================================================================
#define IN_DIMENTION 2

//=======================================================================================
// Name space 
using namespace wordfeat;
using namespace std;

//=======================================================================================
// Kernal function: Extract feature for a single sentence
//---------------------------------------------------------------------------------------
// Assume that input sentence is a matrix of N x L x M
// where  
//        L: the length in words of a specified sentence
//        M: the size of raw feature of each word
// The output is a feature matrix of N x L x D x S
// where  
//        L: the length in words of a specified sentence
//        D: the number of dimensions of feature
//        S: the size of each feature dimension to pre-allocated 
__global__ void extract_feat_single(unsigned int * inMat,  int inL,  int inM, 
                                    unsigned int * outMat, int outL, int outD, int outS)
{

}

int * unroll_data_to_mat(vector<vector<pair<int, int> > >&inData, int L, int M)
{
  // Calculate and allocate host memory 
  int N = inData.size();
  int * hostMat = (int *) malloc( N * L * M * sizeof(int) );

  for( int i = 0 ; i < inData.size(); i++){
    for ( int j =0 ; j < inData[i].size(); j++ )
    {
      hostMat[ i * L * M + j * M + 0 ] = inData[i][j].first;
      hostMat[ i * L * M + j * M + 1 ] = inData[i][j].second;
    }
  }

  return hostMat;
}

bool parseInput(ifstream &infile,  vector<vector<pair<int, int> > >&inData, 
    map<int, string> &wordDict, int wordWindow, int& maxSentLength)
{
  // Here notice that token starts from 2 * WORD_WINDOW
  // for case word window = 2, we have: 
  //    * 0 - <ss>
  //    * 1 - <s>
  //    * 2 - </s>
  //    * 3 - </ss>
  int tokenNum = wordWindow * 2;
  int sentNum = 0;
  
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
      for (int i = 0; i < wordWindow; i++){
        inData[sentNum].insert( inData[sentNum].begin(), make_pair(i, i) );
        inData[sentNum].push_back( make_pair(wordWindow*2-1-i, wordWindow*2-1-i) );
      }
      // update maximum sentence length
      maxSentLength = (maxSentLength > inData[sentNum].size())? maxSentLength: inData[sentNum].size();

      inData.push_back( vector<pair<int, int> >() );
      sentNum ++;
    }

    inData[sentNum].push_back( make_pair(tokenDict[word], tokenDict[tag]) );
  }
  inData.pop_back();


  for (int i = 0; i < wordWindow; i++){
    string pad;
    for (int j = 0; j <= i; j++) pad += "s";
    wordDict[i]                   = "<" + pad + ">";
    wordDict[wordWindow*2 - 1 - i]  = "</" + pad + ">";
  }
  // Swap key value for map
  for (  map<string, int>::iterator it = tokenDict.begin(); it != tokenDict.end(); it++ ){
    wordDict[it->second] = it->first;
  }
  // Free the space of token dictionary
  tokenDict.clear();

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
  gflags::SetUsageMessage("command line message\n"
    "usage: wordfeat <command> <args>\n\n"
    "commands:\n");
  
  // Set default behavior as output to console
  FLAGS_alsologtostderr = 1;
  GlobalInit(&argc, &argv);

#ifdef INITIALIZATION
  DLOG(INFO) << DEBUG_HEAD; 
  DLOG(INFO) << "IN_FILE:     " << FLAGS_infile;
  DLOG(INFO) << "OUT_FILE:    " << FLAGS_outfile;
  DLOG(INFO) << "WORD_WINDOW: " << FLAGS_window;
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
  int wordWindow = FLAGS_window;
  int maxSentLength = 0;
  
  // Parse input file into specified data structure
  parseInput(infile, inData, wordDict, wordWindow, maxSentLength);

  LOG(INFO) << "Maximum sentence length: " << maxSentLength;

  int * hostMat = unroll_data_to_mat(inData, maxSentLength, IN_DIMENTION );
   
  LOG(INFO) << "Finished.";

  free(hostMat);

  return 0;
}