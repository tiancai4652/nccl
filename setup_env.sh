#!/bin/bash
# NCCL CPU化改造环境变量配置脚本

# 获取当前目录的绝对路径
NCCL_ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export NCCL_ROOT_DIR="$NCCL_ROOT_DIR"
export CUDA_HOME="$NCCL_ROOT_DIR/fake_cuda"  
export CUDA_LIB="$CUDA_HOME/lib"
export CUDA_INC="$CUDA_HOME/include"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_LIB:$NCCL_ROOT_DIR/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 拓扑和运行配置
export NCCL_TOPO_FILE="$NCCL_ROOT_DIR/topo/default_topology.xml"
export GPU_DEV_NUM=4

# 调试配置
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=ALL

echo "NCCL CPU化环境变量已设置:"
echo "  NCCL_ROOT_DIR = $NCCL_ROOT_DIR"
echo "  CUDA_HOME     = $CUDA_HOME"
echo "  GPU_DEV_NUM   = $GPU_DEV_NUM"
echo "  NCCL_TOPO_FILE = $NCCL_TOPO_FILE"
