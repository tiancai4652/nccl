# NCCL CPU化改造通用指南

本指南基于NCCL_GP项目的思路，提供一个系统化的步骤，将任何版本的NCCL改造为可在CPU上运行的版本，用于学习、调试和流信息提取。

## 改造原理

NCCL CPU化的核心思想是：
1. **替换CUDA依赖**：使用虚拟的CUDA接口替代真实GPU调用
2. **屏蔽硬件操作**：让所有GPU相关操作变为空操作或返回预设值
3. **保留通信逻辑**：完整保留算法选择、拓扑分析、流生成等核心逻辑
4. **XML拓扑驱动**：通过XML文件定义虚拟的硬件拓扑

## 步骤1：准备工作

### 1.1 获取目标NCCL版本
```bash
# 克隆或下载你需要的NCCL版本
git clone https://github.com/NVIDIA/nccl.git
cd nccl
git checkout v2.19.1  # 替换为你需要的版本
```

### 1.2 创建工作分支
```bash
git checkout -b cpu-adaptation
```

## 步骤2：创建fake_cuda替代库

### 2.1 创建fake_cuda目录结构
```bash
mkdir -p fake_cuda/{include,lib,src}
```

### 2.2 复制CUDA头文件
```bash
# 从系统CUDA安装或NCCL_GP项目中复制关键头文件到fake_cuda/include/
# 主要包括：
# - cuda_runtime.h
# - cuda.h  
# - cudnn.h (如果需要)
# - cublas.h (如果需要)
```

### 2.3 创建虚拟CUDA实现
在`fake_cuda/src/fake_cuda.cc`中实现所有CUDA API的虚拟版本：

```cpp
// fake_cuda/src/fake_cuda.cc
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>

// GPU设备数量（可通过环境变量GPU_DEV_NUM控制）
static int getGpuDeviceCount() {
    const char* env = getenv("GPU_DEV_NUM");
    return env ? atoi(env) : 4;  // 默认4个虚拟GPU
}

// 关键CUDA API实现
cudaError_t CUDARTAPI cudaGetDeviceCount(int *count) {
    *count = getGpuDeviceCount();
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaSetDevice(int device) {
    // 虚拟设置，实际不做任何操作
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
    // 根据kind决定是否真的拷贝数据
    if (kind == cudaMemcpyHostToHost || kind == cudaMemcpyHostToDevice) {
        memcpy(dst, src, count);
    }
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
    return cudaMemcpy(dst, src, count, kind);  // 同步执行
}

// 更多CUDA API的虚拟实现...
// 参考NCCL_GP/src/graph/fake_cuda.cc完整实现
```

### 2.4 创建虚拟库文件
```bash
# 编译fake_cuda实现为静态库
cd fake_cuda/src
g++ -fPIC -c fake_cuda.cc -I../include
ar rcs ../lib/libcudart_static.a fake_cuda.o
ar rcs ../lib/libcudart.a fake_cuda.o
```

## 步骤3：修改编译系统

### 3.1 修改环境变量配置
创建`setup_env.sh`脚本：
```bash
#!/bin/bash
export NCCL_ROOT_DIR=$(pwd)
export CUDA_HOME=$NCCL_ROOT_DIR/fake_cuda  
export CUDA_LIB=$CUDA_HOME/lib
export CUDA_INC=$CUDA_HOME/include
export LD_LIBRARY_PATH=$CUDA_LIB:$NCCL_ROOT_DIR/build/lib
export NCCL_TOPO_FILE=$NCCL_ROOT_DIR/topo/default_topology.xml
export GPU_DEV_NUM=4
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=ALL
```

### 3.2 修改Makefile
在`src/Makefile`中添加fake_cuda源文件：

```makefile
# 在LIBSRCFILES中添加fake_cuda实现
LIBSRCFILES := init.cc ... graph/fake_cuda.cc

# 如果需要Flow Extractor功能，还要添加：
LIBSRCFILES := ... flow_extractor.cc
```

## 步骤4：屏蔽硬件相关代码

### 4.1 识别需要屏蔽的函数
搜索NCCL源码中所有硬件相关调用：
```bash
# 搜索GPU相关调用
grep -r "cudaGetDevice\|cudaSetDevice\|cudaMalloc\|cudaFree" src/
grep -r "nvidia-ml\|nvmlDevice" src/
grep -r "cuCtx\|cuDevice" src/
```

### 4.2 关键文件修改点

#### 4.2.1 `src/init.cc`
```cpp
// 在设备初始化相关函数中，替换真实GPU查询为虚拟实现
static ncclResult_t initTransports(struct ncclComm* comm) {
  // 原版会查询真实GPU，修改为使用虚拟GPU信息
  int nDev = getGpuDeviceCount();  // 使用虚拟函数
  // ... 其余逻辑保持不变
}
```

#### 4.2.2 `src/transport/*.cc`
屏蔽网络传输相关的硬件操作，让其返回成功但不实际传输。

#### 4.2.3 `src/proxy.cc`  
在代理操作中屏蔽实际的网络I/O，保留逻辑流程。

### 4.3 保留关键逻辑
**重要**：只屏蔽硬件I/O操作，完整保留以下逻辑：
- 算法选择逻辑
- 拓扑分析逻辑  
- 通信模式计算
- ProxyOp生成
- 流步骤计算

## 步骤5：创建XML拓扑系统

### 5.1 创建拓扑目录
```bash
mkdir -p topo
```

### 5.2 创建默认拓扑文件
`topo/default_topology.xml`：
```xml
<?xml version="1.0" ?>
<system version="1">
  <!-- 定义虚拟GPU节点 -->
  <gpu dev="0" sm="80" rank="0" gdr="1">
    <nvlink target="1" count="12" bw="25"/>
    <nvlink target="2" count="12" bw="25"/>
  </gpu>
  <gpu dev="1" sm="80" rank="1" gdr="1">
    <nvlink target="0" count="12" bw="25"/>  
    <nvlink target="3" count="12" bw="25"/>
  </gpu>
  <!-- 更多GPU定义... -->
  
  <!-- 定义网络连接 -->
  <net>
    <nic dev="0" speed="100" port="1" guid="0x001" />
    <nic dev="1" speed="100" port="1" guid="0x002" />
  </net>
</system>
```

### 5.3 修改XML读取逻辑
在`src/graph/xml.cc`中确保正确读取自定义XML文件而不是硬件探测结果。

## 步骤6：添加流提取功能（可选）

如果需要流信息提取功能，可以参考NCCL_GP添加：

### 6.1 创建流提取器
`src/flow_extractor.h`和`src/flow_extractor.cc`

### 6.2 在关键点插桩
在`src/enqueue.cc`和`src/proxy.cc`中添加流记录调用。

## 步骤7：编译和测试

### 7.1 编译
```bash
source setup_env.sh
make -j$(nproc) DEBUG=1 TRACE=1
```

### 7.2 创建测试程序
```cpp
// test/test_cpu_nccl.cpp
#include <nccl.h>
#include <stdio.h>

int main() {
    int nDev = 4;
    ncclComm_t* comms = (ncclComm_t*)malloc(nDev * sizeof(ncclComm_t));
    int* devs = (int*)malloc(nDev * sizeof(int));
    
    for (int i = 0; i < nDev; i++) devs[i] = i;
    
    // 初始化NCCL通信器
    ncclResult_t result = ncclCommInitAll(comms, nDev, devs);
    if (result != ncclSuccess) {
        printf("NCCL初始化失败: %s\n", ncclGetErrorString(result));
        return -1;
    }
    
    // 测试集合通信（不会真实传输数据）
    const size_t count = 1024;
    ncclGroupStart();
    for (int i = 0; i < nDev; ++i) {
        ncclAllReduce(NULL, NULL, count, ncclFloat, ncclSum, comms[i], 0);
    }
    ncclGroupEnd();
    
    printf("NCCL CPU版本测试成功！\n");
    
    // 清理
    for (int i = 0; i < nDev; i++) ncclCommDestroy(comms[i]);
    free(comms);
    free(devs);
    return 0;
}
```

## 步骤8：验证功能

### 8.1 基本功能测试
```bash
./test/test_cpu_nccl
```

### 8.2 流提取测试（如果实现了）
检查是否生成了正确的流信息文件。

### 8.3 算法一致性验证
对比CPU版本和GPU版本在相同参数下的算法选择结果。

## 常见问题和解决方案

### Q1: 编译时找不到CUDA头文件
**A**: 确保fake_cuda/include目录包含了所需的头文件，并且CUDA_INC环境变量正确设置。

### Q2: 运行时段错误
**A**: 检查fake_cuda实现中是否遗漏了某些CUDA API，添加相应的虚拟实现。

### Q3: 算法选择结果异常
**A**: 确认拓扑XML文件正确描述了预期的硬件布局，并且算法选择逻辑没有被意外修改。

### Q4: 不同NCCL版本适配差异
**A**: 主要关注以下可能的变化：
- 新增的CUDA API调用
- 算法选择逻辑的改变
- 数据结构的变更
- 编译系统的更新

## 版本适配注意事项

1. **API兼容性**：新版本可能引入新的CUDA API，需要在fake_cuda中添加对应实现
2. **数据结构变化**：注意comm、channel等关键数据结构的字段变更  
3. **算法更新**：新版本可能添加新算法，确保相关逻辑被正确保留
4. **编译依赖**：检查新版本是否引入新的编译依赖

## 总结

通过以上步骤，可以将任何版本的NCCL改造为CPU运行版本，保留完整的通信逻辑用于研究和仿真。关键是正确实现fake_cuda替代层，并谨慎屏蔽硬件操作而不破坏核心算法逻辑。

改造完成后，你将获得：
- ✅ 无需GPU的NCCL运行环境
- ✅ 完整的算法选择逻辑
- ✅ 准确的通信流信息
- ✅ 可定制的虚拟拓扑
- ✅ 与原版逻辑一致的结果
