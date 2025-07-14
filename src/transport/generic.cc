#include "comm.h"
#include "transport.h"
#include "bootstrap.h"

ncclResult_t ncclTransportRingConnect(struct ncclComm* comm) {
  struct ringConnInfo {
    bool useNetPXN;
    bool useGdr;
  };
  struct ringConnInfo* ringInfo = NULL;
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    comm->useGdr = true;
    comm->useNetPXN = false;
    
    // 检查是否启用2D ring
    const int xDim = 2;  // 写死X维度
    const int yDim = comm->nRanks / xDim;
    const bool use2D = true;
    
    if (use2D) {
      printf("[2D_RING_DEBUG] Setting up 2D ring connections: %dx%d. \n", xDim, yDim);
      
      // 建立1D ring连接（原有逻辑）
      for (int c = 0; c < comm->nChannels; c++) {
        struct ncclChannel* channel = comm->channels + c;
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, 0), ret, fail);
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_RING], 0), ret, fail);
      
      // 建立2D ring的额外连接
      const int rank = comm->rank;
      const int xRank = rank % xDim;
      const int yRank = rank / xDim;
      
      // 为每个channel建立2D连接
      for (int c = 0; c < comm->nChannels; c++) {
        struct ncclChannel* channel = comm->channels + c;
        
        // X维连接：同一行的rank连接
        int xPrev = (xRank - 1 + xDim) % xDim + yRank * xDim;
        int xNext = (xRank + 1) % xDim + yRank * xDim;
        
        // Y维连接：同一列的rank连接
        int yPrev = xRank + ((yRank - 1 + yDim) % yDim) * xDim;
        int yNext = xRank + ((yRank + 1) % yDim) * xDim;
        
        // 建立X维连接
        if (xPrev != rank) {
          NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &xPrev, 1, &xNext, 1), ret, fail);
        }
        
        // 建立Y维连接
        if (yPrev != rank) {
          NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &yPrev, 1, &yNext, 2), ret, fail);
        }

        // INFO(NCCL_INIT, "2D ring connections established for rank %d: X(%d,%d) Y(%d,%d)", 
        //    rank, xRank, yRank, xPrev, yPrev);
        printf("[2D_RING_DEBUG] 2D ring connections %d: %d %d %d %d. \n", rank, xPrev,xNext,yPrev,yNext);
      }
      
      // 设置2D ring的graph
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_RING], 1), ret, fail);
      
      
    } else {
      // 原有的1D ring连接逻辑
      for (int c = 0; c < comm->nChannels; c++) {
        struct ncclChannel* channel = comm->channels + c;
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, 0), ret, fail);
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_RING], 0), ret, fail);
    }
    
    if (ncclParamLocalRegister() || ncclParamGraphRegister()) {
      NCCLCHECK(ncclCalloc(&ringInfo, comm->nRanks));
      ringInfo[comm->rank].useGdr = comm->useGdr;
      ringInfo[comm->rank].useNetPXN = comm->useNetPXN;
      NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, ringInfo, sizeof(struct ringConnInfo)), ret, fail);
      for (int i = 0; i < comm->nRanks; ++i) {
        if (!ringInfo[i].useGdr) comm->useGdr = false;
        if (ringInfo[i].useNetPXN) comm->useNetPXN = true;
        if (comm->useGdr == false && comm->useNetPXN == true) break;
      }
    }
    INFO(NCCL_INIT, "Connected all rings, use ring PXN %d GDR %d", comm->useNetPXN, comm->useGdr);
  }
exit:
  free(ringInfo);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTransportTreeConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    // Connect Trees
    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_TREE_ARITY, channel->tree.down, 1, &channel->tree.up, 0), ret, fail);
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->tree.up, NCCL_MAX_TREE_ARITY, channel->tree.down, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
    INFO(NCCL_INIT, "Connected all trees");
  }
exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTransportPatConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    for (int mask=1; mask<comm->nRanks; mask<<=1) {
      int prevPeer = (comm->rank + mask) % comm->nRanks;
      int nextPeer = (comm->rank + comm->nRanks - mask) % comm->nRanks;
      for (int c = 0; c < comm->nChannels; c++) {
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &prevPeer, 1, &nextPeer, 0), ret, fail); // ReduceScatter
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
      for (int c = 0; c < comm->nChannels; c++) {
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &nextPeer, 1, &prevPeer, 0), ret, fail); // AllGather
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
    }
    INFO(NCCL_INIT, "Connected binomial trees");
  }
exit:
  return ret;
fail:
  goto exit;
}
