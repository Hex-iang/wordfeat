#include <common.h>
#include <utils.h>

__global__ void extract_feat(int * inMat, int * FeatMat, int numRow, int numCol )
{

}


int main(int argc, char * argv[])
{
  gflags::SetUsageMessage("command line message\n"
    "usage: wordfeat <command> <args>\n\n"
    "commands:\n"
    "  train           train or finetune a model\n"
    "  test            score a model\n"
    "  device_query    show GPU diagnostic information\n"
    "  time            benchmark model execution time");
  GlobalInit(&argc, &argv);

}
