#include <utils.h>

void Initialization(int &argc, char * argv[])
{
  // Google logging
  ::google::InitGoogleLogging(argv[0]);
  // Provide a backtrace on failure
  ::google::InstallFailureSignalHandler();

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif 

  gflags::ParseCommandLineFlags(&argc, &argv, true);

}

int main(int argc, char* argv[])
{
  Initialization(argc, argv);

  int deviceCount = 0;
  cudaGetDeviceCount(& deviceCount);
  LOG(INFO) << "CUDA Device count: " << deviceCount << ".";

  wfeatTime_start(GPU, "Getting GPU properties");

  for( int devNum = 0; devNum < deviceCount; devNum++ ) 
  {
    cudaDeviceProp deviceProp; 

    cudaGetDeviceProperties(&deviceProp, devNum);
       
    if (devNum == 0) {
      if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
        LOG(INFO) << "No CUDA GPU has been detected.";
      } else if (deviceCount == 1) {
        LOG(INFO) << "There is 1 device supporting CUDA";
      } else {
        LOG(INFO) << "There are " << deviceCount << " devices supporting CUDA";
      }   
    }
    
    // output corresponding information about devise
    LOG(INFO) << "Device " <<  devNum <<  " name: " <<  deviceProp.name;
    LOG(INFO) << " Computational Capabilities: " <<  deviceProp.major << "." << deviceProp.minor;
    LOG(INFO) << " Maximum global memory size: " <<  deviceProp.totalGlobalMem;
    LOG(INFO) << " Maximum constant memory size: ", deviceProp.totalConstMem;
    LOG(INFO) << " Maximum shared memory size per block: " << deviceProp.sharedMemPerBlock;
    LOG(INFO) << " Maximum threads per block: " << deviceProp.maxThreadsPerBlock;
    LOG(INFO) << " Maximum block dimensions: " << deviceProp.maxThreadsDim[0] << " x " 
                                               << deviceProp.maxThreadsDim[1] << " x " 
                                               << deviceProp.maxThreadsDim[2]; 
    LOG(INFO) << " Maximum grid dimensions: "  << deviceProp.maxGridSize[0] << " x "
                                               << deviceProp.maxGridSize[1] << " x "
                                               << deviceProp.maxGridSize[2];
    LOG(INFO) << " Warp size: " << deviceProp.warpSize;
  }
  
  wfeatTime_stop(GPU, "Getting GPU properties");
}
