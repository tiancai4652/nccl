/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"

void dumpLine(int* values, int nranks, const char* prefix) {
  constexpr int line_length = 128;
  char line[line_length];
  int num_width = snprintf(nullptr, 0, "%d", nranks-1);  // safe as per "man snprintf"
  int n = snprintf(line, line_length, "%s", prefix);
  for (int i = 0; i < nranks && n < line_length-1; i++) {
    n += snprintf(line + n, line_length - n, " %*d", num_width, values[i]);
    // At this point n may be more than line_length-1, so don't use it
    // for indexing into "line".
  }
  if (n >= line_length) {
    // Sprintf wanted to write more than would fit in the buffer. Assume
    // line_length is at least 4 and replace the end with "..." to
    // indicate that it was truncated.
    snprintf(line+line_length-4, 4, "...");
  }
  INFO(NCCL_INIT, "%s", line);
}

ncclResult_t ncclBuildRings(int nrings, int* rings, int rank, int nranks, int* prev, int* next) {
  for (int r=0; r<nrings; r++) {
    char prefix[40];
    /*sprintf(prefix, "[%d] Channel %d Prev : ", rank, r);
    dumpLine(prev+r*nranks, nranks, prefix);
    sprintf(prefix, "[%d] Channel %d Next : ", rank, r);
    dumpLine(next+r*nranks, nranks, prefix);*/

    int current = rank;
    for (int i=0; i<nranks; i++) {
      rings[r*nranks+i] = current;
      current = next[r*nranks+current];
    }
    snprintf(prefix, sizeof(prefix), "Channel %02d/%02d :", r, nrings);
    if (rank == 0) dumpLine(rings+r*nranks, nranks, prefix);
    if (current != rank) {
      WARN("Error : ring %d does not loop back to start (%d != %d)", r, current, rank);
      // return ncclInternalError;// for 2d topology
    }
    // Check that all ranks are there
    for (int i=0; i<nranks; i++) {
      int found = 0;
      for (int j=0; j<nranks; j++) {
        if (rings[r*nranks+j] == i) {
          found = 1;
          break;
        }
      }
      if (found == 0) {
        WARN("Error : ring %d does not contain rank %d", r, i);
        // return ncclInternalError;// for 2d topology
      }
    }
  }
  return ncclSuccess;
}

 ncclResult_t ncclBuild2dRings(int nChannels, int* rings, int rank, int nranks, int* ringPrev, int* ringNext) {  
  // 对于2D拓扑，我们需要验证每个维度的小环而不是单一大环  
  for (int c = 0; c < nChannels; c++) {  
    int* prev = ringPrev + c * nranks;  
    int* next = ringNext + c * nranks;  
      
    // 识别当前channel的维度类型  
    bool isXDimension = (c % 2 == 0);  
      
    // 验证小环的完整性  
    for (int r = 0; r < nranks; r++) {  
      if (prev[r] != -1 && next[r] != -1) {  
        // 验证环形连接：prev[next[r]] 应该等于 r  
        if (prev[next[r]] != r) {  
          WARN("2D Ring validation failed: inconsistent connection for rank %d in channel %d", r, c);  
          return ncclInternalError;  
        }  
      }  
    }  
      
    // 构建ring数组 - 为每个小环分别构建  
    int ringIndex = 0;  
    for (int r = 0; r < nranks; r++) {  
      if (prev[r] == -1) { // 找到环的起始点  
        int current = r;  
        while (current != -1) {  
          rings[c * nranks + ringIndex++] = current;  
          current = next[current];  
          if (current == r) break; // 回到起始点，环完成  
        }  
      }  
    }  
  }  
    
  return ncclSuccess;  
}