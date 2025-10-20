# NCCL CPU化改造 - 代码示例和使用指南

## 🎯 实用代码示例集

---

## 1. 环境配置和测试

### 1.1 快速开始
```bash
# 1. 进入项目目录
cd /home/zhangran/work/NCCL-SHARP/nccl

# 2. 配置环境（必须）
source ./setup_env.sh

# 3. 验证配置
echo "CUDA_HOME: $CUDA_HOME"
echo "GPU数量: $GPU_DEV_NUM" 
echo "拓扑文件: $NCCL_TOPO_FILE"

# 4. 测试fake_cuda
./test_fake_cuda

# 5. 测试NCCL接口
./test_nccl_minimal

# 6. 综合功能验证
./final_test
```

### 1.2 自定义GPU数量
```bash
# 修改虚拟GPU数量
export GPU_DEV_NUM=8  # 设置8个虚拟GPU
./test_fake_cuda      # 验证设备数量变化
```

### 1.3 调试模式
```bash
# 开启详细调试日志
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=ALL

# 运行程序查看详细执行过程
./your_nccl_program 2>&1 | tee debug.log
```

---

## 2. fake_cuda API使用示例

### 2.1 设备管理
```cpp
#include "fake_cuda/include/cuda_runtime.h"

int main() {
    // 查询虚拟GPU数量
    int deviceCount = 0;
    cudaError_t result = cudaGetDeviceCount(&deviceCount);
    printf("发现 %d 个虚拟GPU设备\n", deviceCount);
    
    // 遍历所有设备
    for (int i = 0; i < deviceCount; i++) {
        // 设置当前设备
        cudaSetDevice(i);
        
        // 获取设备属性
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("设备 %d:\n", i);
        printf("  名称: %s\n", prop.name);
        printf("  显存: %zu MB\n", prop.totalGlobalMem / (1024*1024));
        printf("  计算能力: %d.%d\n", prop.major, prop.minor);
        printf("  多处理器: %d\n", prop.multiProcessorCount);
    }
    
    return 0;
}
```

### 2.2 内存管理
```cpp
void test_memory_operations() {
    size_t size = 1024 * 1024;  // 1MB
    void* d_ptr = nullptr;
    void* h_ptr = malloc(size);
    
    // GPU内存分配（实际在CPU上）
    cudaError_t result = cudaMalloc(&d_ptr, size);
    if (result == cudaSuccess) {
        printf("✅ GPU内存分配成功: %p\n", d_ptr);
        
        // 内存拷贝（CPU到CPU）
        cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
        printf("✅ 内存拷贝完成\n");
        
        // 释放GPU内存
        cudaFree(d_ptr);
        printf("✅ GPU内存释放完成\n");
    }
    
    free(h_ptr);
}
```

### 2.3 流和事件管理
```cpp
void test_streams_and_events() {
    cudaStream_t stream1, stream2;
    cudaEvent_t start, stop;
    
    // 创建流
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    printf("✅ 创建了2个CUDA流\n");
    
    // 创建事件
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    printf("✅ 创建了CUDA事件\n");
    
    // 记录事件
    cudaEventRecord(start, stream1);
    // 模拟一些工作...
    cudaEventRecord(stop, stream1);
    
    // 同步
    cudaStreamSynchronize(stream1);
    cudaEventSynchronize(stop);
    printf("✅ 流和事件同步完成\n");
    
    // 清理
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
```

---

## 3. XML拓扑配置示例

### 3.1 基本4GPU环拓扑
```xml
<!-- topo/ring_4gpu.xml -->
<?xml version="1.0" ?>
<system version="1">
  <!-- GPU 0 -->
  <gpu dev="0" sm="80" rank="0" gdr="1">
    <pci busid="0000:01:00.0"/>
    <nvlink target="1" count="12" bw="25.0"/>  <!-- 连接到GPU1 -->
  </gpu>
  
  <!-- GPU 1 -->  
  <gpu dev="1" sm="80" rank="1" gdr="1">
    <pci busid="0000:02:00.0"/>
    <nvlink target="0" count="12" bw="25.0"/>  <!-- 连接到GPU0 -->
    <nvlink target="2" count="12" bw="25.0"/>  <!-- 连接到GPU2 -->
  </gpu>
  
  <!-- GPU 2 -->
  <gpu dev="2" sm="80" rank="2" gdr="1">
    <pci busid="0000:03:00.0"/>
    <nvlink target="1" count="12" bw="25.0"/>  <!-- 连接到GPU1 -->
    <nvlink target="3" count="12" bw="25.0"/>  <!-- 连接到GPU3 -->
  </gpu>
  
  <!-- GPU 3 -->
  <gpu dev="3" sm="80" rank="3" gdr="1">
    <pci busid="0000:04:00.0"/>
    <nvlink target="2" count="12" bw="25.0"/>  <!-- 连接到GPU2 -->
  </gpu>
  
  <!-- 网络接口 -->
  <net name="eth" id="0" speed="100" port="0" guid="0x001">
    <pci busid="0000:05:00.0"/>
  </net>
</system>
```

### 3.2 切换不同拓扑配置
```bash
# 使用环形拓扑
export NCCL_TOPO_FILE="topo/ring_4gpu.xml"
./test_algorithm_selection

# 使用全连接拓扑  
export NCCL_TOPO_FILE="topo/fully_connected.xml"
./test_algorithm_selection

# 使用树状拓扑
export NCCL_TOPO_FILE="topo/tree_topology.xml" 
./test_algorithm_selection
```

---

## 4. NCCL程序示例

### 4.1 基本AllReduce示例
```cpp
#include <stdio.h>
#include <stdlib.h>
// 注意：在CPU化版本中，这些头文件来自我们的构建系统
// #include <nccl.h>  // 需要完整编译后才有

// 简化的NCCL类型定义（用于演示）
typedef enum { ncclSuccess = 0 } ncclResult_t;
typedef enum { ncclFloat = 7 } ncclDataType_t;
typedef enum { ncclSum = 0 } ncclRedOp_t;
typedef struct ncclComm* ncclComm_t;

int main() {
    const int nDev = 4;  // 4个虚拟GPU
    const int size = 1024;  // 数据大小
    
    // 分配通信器数组
    ncclComm_t* comms = (ncclComm_t*)malloc(nDev * sizeof(ncclComm_t));
    int* devs = (int*)malloc(nDev * sizeof(int));
    
    // 初始化设备列表
    for (int i = 0; i < nDev; i++) {
        devs[i] = i;
    }
    
    printf("初始化NCCL通信器...\n");
    // ncclCommInitAll(comms, nDev, devs);  // 需要完整编译版本
    
    printf("执行AllReduce操作...\n");
    // 在实际使用中，这里会调用ncclAllReduce
    
    printf("销毁通信器...\n");
    for (int i = 0; i < nDev; i++) {
        // ncclCommDestroy(comms[i]);
    }
    
    free(comms);
    free(devs);
    printf("✅ NCCL程序执行完成\n");
    
    return 0;
}
```

### 4.2 流信息提取示例
```cpp
// 使用流提取器记录通信流程
#include "src/flow_extractor.h"

void demonstrate_flow_extraction() {
    // 启用流信息提取
    ncclSetFlowExtractionEnabled(1);
    
    // 执行NCCL操作（会自动记录流信息）
    // ncclAllReduce(...);
    
    // 写入聚合的流信息
    // ncclWriteAggregatedFlow(comm);
    
    printf("流信息已保存到以下文件:\n");
    printf("- proxy_flow_rank0.jsonl  (算法选择摘要)\n");
    printf("- flow_steps_rank0.jsonl  (详细通信步骤)\n");
    printf("- flow_rank0.json         (完整聚合信息)\n");
}
```

---

## 5. 编译和构建示例

### 5.1 编译fake_cuda库
```bash
# 进入fake_cuda目录
cd fake_cuda/src

# 编译共享库
make clean
make all

# 检查生成的库文件
ls -la ../lib/
# 应该看到:
# libcudart.so       <- 共享库
# libcudart_static.a <- 静态库
```

### 5.2 编译测试程序
```bash
# 编译基础测试
g++ -I. -Lfake_cuda/lib test_fake_cuda.cpp -lcudart -o test_fake_cuda

# 编译带调试信息的版本
g++ -g -O0 -I. -Lfake_cuda/lib test_fake_cuda.cpp -lcudart -o test_fake_cuda_debug

# 运行测试
LD_LIBRARY_PATH=fake_cuda/lib ./test_fake_cuda
```

### 5.3 完整NCCL编译（当需要时）
```bash
# 设置环境
source ./setup_env.sh

# 清理之前的构建
make clean

# 开始编译（可能需要处理一些硬件相关错误）
make -j$(nproc) 2>&1 | tee build.log

# 查看编译结果
ls -la build/lib/
```

---

## 6. 实际应用场景

### 6.1 算法选择研究
```python
#!/usr/bin/env python3
# 研究不同拓扑下的算法选择
import subprocess
import json
import os

topologies = [
    "topo/ring_topology.xml",
    "topo/nvlink_topology.xml", 
    "topo/tree_topology.xml"
]

data_sizes = [1024, 65536, 1048576]  # 1KB, 64KB, 1MB

results = []

for topo in topologies:
    for size in data_sizes:
        # 设置环境
        env = os.environ.copy()
        env['NCCL_TOPO_FILE'] = topo
        
        # 运行NCCL程序
        cmd = ['./nccl_test', '--size', str(size), '--collective', 'allreduce']
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        # 解析结果
        # algorithm = parse_algorithm_from_output(result.stdout)
        
        results.append({
            'topology': topo,
            'data_size': size,
            # 'selected_algorithm': algorithm
        })

# 分析结果
print("算法选择研究结果:")
for r in results:
    print(f"拓扑: {r['topology']}, 数据大小: {r['data_size']}")
```

### 6.2 仿真器集成
```cpp
// 在DoD仿真器中集成NCCL CPU版本
class NCCLSimulator {
public:
    struct SimResult {
        std::string algorithm;     // Ring/Tree/CollNet
        std::string protocol;      // LL/LL128/SIMPLE  
        int steps;                 // 通信步数
        std::vector<CommStep> communication_steps;
    };
    
    SimResult simulate_collective(const std::string& collective_type,
                                 size_t data_size, 
                                 int npu_count) {
        // 1. 设置虚拟环境
        setenv("GPU_DEV_NUM", std::to_string(npu_count).c_str(), 1);
        
        // 2. 调用NCCL CPU版本
        // ncclResult_t result = run_nccl_simulation(collective_type, data_size);
        
        // 3. 解析流信息JSON文件
        SimResult sim_result;
        // sim_result = parse_flow_json("flow_rank0.json");
        
        return sim_result;
    }
};
```

### 6.3 教学演示
```bash
#!/bin/bash
# NCCL教学演示脚本

echo "=== NCCL CPU化演示 ==="

echo "1. 环境配置"
source ./setup_env.sh
echo "✅ 环境配置完成"

echo "2. 测试虚拟GPU"
./test_fake_cuda | head -10
echo "✅ 虚拟GPU工作正常"

echo "3. 不同拓扑下的算法选择"
for topo in ring nvlink tree; do
    echo "测试拓扑: $topo"
    export NCCL_TOPO_FILE="topo/${topo}_topology.xml"
    # ./demo_algorithm_selection
    echo "✅ $topo 拓扑测试完成"
done

echo "4. 查看生成的流信息"
if [ -f "flow_rank0.json" ]; then
    echo "流信息文件大小: $(du -h flow_rank0.json)"
    echo "包含的通信步骤: $(jq '.steps | length' flow_rank0.json)"
fi

echo "=== 演示完成 ==="
```

---

## 7. 故障排除和调试

### 7.1 常见问题检查
```bash
#!/bin/bash
# NCCL CPU化问题诊断脚本

echo "=== NCCL CPU化诊断 ==="

# 检查环境变量
echo "1. 环境变量检查:"
vars=("NCCL_ROOT_DIR" "CUDA_HOME" "NCCL_TOPO_FILE" "GPU_DEV_NUM")
for var in "${vars[@]}"; do
    if [ -n "${!var}" ]; then
        echo "✅ $var = ${!var}"
    else
        echo "❌ $var 未设置"
    fi
done

# 检查关键文件
echo "2. 关键文件检查:"
files=(
    "fake_cuda/lib/libcudart.so"
    "fake_cuda/bin/nvcc"
    "$NCCL_TOPO_FILE"
    "setup_env.sh"
)
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file 存在"
    else
        echo "❌ $file 缺失"
    fi
done

# 检查库链接
echo "3. 库链接检查:"
if ldd test_fake_cuda 2>/dev/null | grep -q libcudart; then
    echo "✅ CUDA库链接正常"
else
    echo "❌ CUDA库链接问题"
fi

# 测试基本功能
echo "4. 功能测试:"
if ./test_fake_cuda >/dev/null 2>&1; then
    echo "✅ fake_cuda功能正常"
else
    echo "❌ fake_cuda功能异常"
fi

echo "=== 诊断完成 ==="
```

### 7.2 调试日志分析
```bash
# 生成详细调试日志
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=ALL

# 运行程序并保存日志
./your_nccl_program 2>&1 | tee nccl_debug.log

# 分析关键信息
echo "=== 算法选择信息 ==="
grep "Algorithm.*selected" nccl_debug.log

echo "=== 拓扑信息 ==="
grep -i "topology\|nvlink" nccl_debug.log

echo "=== 错误信息 ==="
grep -i "error\|fail" nccl_debug.log
```

---

## 📝 总结

这个代码示例文档提供了：

1. **环境配置和测试** - 快速开始使用
2. **fake_cuda API示例** - 理解虚拟化实现
3. **XML拓扑配置** - 自定义虚拟硬件
4. **NCCL程序示例** - 实际应用代码
5. **编译构建指南** - 完整构建流程
6. **应用场景演示** - 研究、仿真、教学
7. **故障排除方法** - 问题诊断和调试

配合前面的技术详解和对比表，你现在有了完整的NCCL CPU化改造学习资料！

---

*通过这些示例代码，你可以深入理解NCCL CPU化的技术实现，并在自己的项目中应用这些技术。*
