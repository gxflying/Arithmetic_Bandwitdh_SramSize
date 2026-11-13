# 推理资源模型的公开测量与框架依据

核对日期为 2026-11-13。下表的时间和吞吐来自作者公开实验，本文所列带宽与运算速率的换算均使用十进制单位。每项实验对应的硬件、执行路径和计时范围随表给出。

## 1. 从框架取出模型所需的形状

**vLLM 的调度单位。** vLLM V1 v0.10.1.1 的 chunked prefill 可将新 prompt 分块，按文档描述优先排 decode，然后用剩余 token 预算处理 prefill。同一轮线性投影可以共享不同请求的权重，矩阵的行数等于该轮计算的新 token 总数。注意力的长度应逐请求计算。如某个请求已有 `p` 个有效前缀 token、本轮追加 `q` 个新 token，因果注意力的有效 query-key 对数为 `qp+q(q+1)/2`。前缀命中可以跳过已计算部分的投影，新增 query 仍须访问其可见 KV。PagedAttention 按 block 管理 KV，并能在适用场景下共享物理块，地址映射、末块取整与实际重复读取次数影响流量和容量。[vLLM 调度与并行说明](https://docs.vllm.ai/en/v0.10.1.1/configuration/optimization.html#chunked-prefill)、[PagedAttention 原论文](https://arxiv.org/abs/2309.06180)

**vLLM 的 Qwen2 计算图。** 每层包含 RMSNorm、QKV 联合投影、Q/K 旋转位置编码、分组注意力和输出投影。FFN 的 gate/up 使用合并列并行投影，随后执行 SiLU 与乘法，并通过行并行 down 投影返回隐藏维度。源码对 KV 头数大于等于 TP 数时采用均分；小于 TP 数时在兼容的 TP 组内复制 KV 头。估算每设备 KV 时须检查头数整除条件。PP 将不同层分配到不同设备，单请求依次经过各段。[Qwen2 算子及分片源码 v0.10.1.1](https://github.com/vllm-project/vllm/blob/v0.10.1.1/vllm/model_executor/models/qwen2.py#L59-L186)、[vLLM 并行策略](https://docs.vllm.ai/en/v0.10.1.1/configuration/optimization.html#parallelism-strategies)

**DeepInfer 的生产例子。** DeepSeek 在 V4.1 Flash 部署说明中给出的策略为 attention 按请求做 DP，MoE 分布专家做 EP，视觉编码、prefill、decode 分池。该部署 prefill 用 2–8 路 EP/DP，decode 用 64–128 路 EP/DP。较长 prompt 能给每个活跃专家提供较多 token；decode 的宽 EP 则有利于摊薄每步专家权重搬运。部署说明同时记录了 EP 扩展后的通信与负载均衡代价。框架提供 balanced CP，该文所述生产配置未启用 TP/CP。公开 release 缺完整的分布式 KV 缓存系统、分离部署集成及跨 DP 副本负载均衡。版本固定至 2026-11-03 提交 `a679bb86`。[DeepInfer 生产部署 Q&A](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/doc/inference_deployment_qa.md#inference-system-overview)、[DeepInfer 公开范围](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/README.md)

**从 dense 外推到这类部署。** 对 routed expert，若此轮本地某专家实际收到 `m_e` 个 token、其 FFN 矩阵共有 `W_e` 个参数，理想线性计算量为 `2 Σ_e m_e W_e` FLOP。权重完整读一次、且各专家权重未跨步驻留的假设下，字节量为 `Σ_{e:m_e>0} b_e W_e`。共享专家、gate、量化 scale、dispatch/combine 通信须另计。源码定义了 shared expert、top-k gate 及各 rank 的 expert 参数。DeepSeek 还使用独立的压缩与索引状态，执行可变长 speculative verification。要估每个最终输出 token，应从“本轮验证了多少位置、被接受并提交了多少输出”换算。计算 dense GQA 的完整 KV 读量无法给出 V4.1 的实际字节数。[DeepInfer MoE 源码](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/moe.py)、[KV 布局、低精度与推测解码](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/doc/inference_deployment_qa.md#kv-cache-storage)

## 2. Qwen2.5-7B-Instruct 整模型

作者的实验使用单张 A100 80 GB、BF16、batch 1，生成 2,048 个 token。表中这些配置用 vLLM 0.6.3、CUDA 12.1、PyTorch 2.4.0 与 Flash Attention 2.6.3。官方脚本计时围绕整次 `llm.generate`，吞吐按配置输出长度除以耗时，因此包含 prefill。[Qwen 官方性能表](https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html)、[Qwen 计时及计算代码](https://github.com/QwenLM/Qwen3/blob/8205ba2a17690375467df31481412d65a1751271/examples/speed-benchmark/speed_benchmark_vllm.py#L91-L167)

| 输入 token | 实测输出 token/s | 换算整次生成时间（s） |
|---:|---:|---:|
| 1 | 84.28 | 24.30 |
| 6,144 | 80.70 | 25.38 |
| 14,336 | 77.69 | 26.36 |
| 30,720 | 70.33 | 29.12 |

以首行为例，层数 28，hidden 3,584，FFN 18,944，Q/KV 头数 28/4，头维 128，词表 152,064。忽略 norm、bias 与逐行读取的输入 embedding，逐步读取的 BF16 大矩阵共 14.1406 GB（含完整输出投影），每增加一个历史 token，全层 KV 增加 57,344 B。单 token 输入、2,048 token 输出条件下，若每步读一次所有大矩阵和实际历史 KV，以中间位置近似平均上下文，首行折合约 **1.20 TB/s 的逻辑数据服务率**。这项换算同时摊入内核开销和少量 prefill，用于检查数量级。显存控制器读写量及 SRAM 容量无法从表中唯一反推。A100 80 GB 的 PCIe/SXM 峰值分别为 1.935/2.039 TB/s，Qwen 表仅提供型号和容量。[模型配置](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/config.json)、[A100 规格](https://www.nvidia.com/en-us/data-center/a100/)

## 3. 单层 attention 的两种形状

**Decode。** Flash-Decoding 作者在 A100 上测量 FP16 单 token GQA，设备本地 16 个 Q 头、2 个 KV 头、头维 128。只取原文表格中 batch=1 的两行。逻辑 KV 读取量为 `2 × B × K × 2 × 128 × 2 B`，包含 K 和 V。[Flash-Decoding 原始实验](https://pytorch.org/blog/flash-decoding/)

| KV 长度 | 逻辑 KV 字节 | Flash-Decoding 实测（μs） | 逻辑 KV 率（TB/s） | FA 2.0.9 实测（μs） |
|---:|---:|---:|---:|---:|
| 65,536 | 67,108,864 | 64.4 | 1.042 | 2,300.6 |
| 131,072 | 134,217,728 | 106.6 | 1.259 | 4,592.2 |

用首行标定常带宽模型后单独预测第二行，预测 128.8 μs，较实测高 **20.8%**。从同一组数据拟合“固定开销 + 线性字节成本”能消除两个点的残差，但两点不足以同时拟合并验证两个参数。在同样的 65,536 长度下，Flash-Decoding 与所列旧版 FA 的实测延迟相差约 35.7 倍。前者可沿 KV 长度分段并行，再通过 log-sum-exp 合并各段输出；对于 batch 1，占用率、切分数和归并代价都会影响逻辑带宽。上述数值可检验模型对形状和调度的敏感性。[算法及实验条件](https://pytorch.org/blog/flash-decoding/)

**多 query 前向。** FlashAttention 论文附录 E.6 表 18 测了 A100 40 GB 上的一层 attention，FP16，batch 16、8 个头、头维 64，不含 QKV 投影，关闭 mask 与 dropout，平均 100 次计时。按两个稠密矩阵积的 `4BHdN²` FLOP 计算有效矩阵吞吐；分母包括 attention 内全部工作。用于 causal prefill 时要重新代入因果对数，且选择与 mask 匹配的 kernel 利用率。[FlashAttention 原论文附录 E.6 与表 18](https://arxiv.org/pdf/2205.14135#page=34)

| 序列长度 | FlashAttention 前向实测（ms） | 矩阵部分等效吞吐（TFLOP/s） |
|---:|---:|---:|
| 1,024 | 0.68 | 50.53 |
| 2,048 | 2.40 | 57.27 |
| 4,096 | 8.42 | 65.29 |

## 4. 区分 vector 吞吐、访存量与 SRAM 容量

Triton 的融合 softmax 在行可放入片上的前提下，对 FP32 矩阵的每个元素执行一次外存读和一次外存写。朴素多核实现的源级账目则有 `5MN+2M` 个元素读和 `3MN+2M` 个元素写。原始 benchmark 固定 4,096 行，当列数为 4,096 时，融合核与朴素核实测分别为 1,404.67 和 338.35 GB/s。报告统一用“输入加输出的最少字节数 / 时间”，折合 175.58 和 42.29 G 输出元素/s。每个元素经过 exp 与两次行归约，整核速率同时受访存、ALU、SFU、归约路径和并行度影响。原页未标出表格所用 GPU 型号，适合验证同页融合收益及计量方法。[Triton Fused Softmax 的实现、指标和数据](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)

FlashAttention 论文图 2 还给出 A100、batch 64、16 头、头维 64、序列 1,024 的测量。标准 attention 和 FlashAttention 的 HBM 读写分别为 40.3 与 4.4 GB，图中该列计入前向和反向，不能用来给纯推理校时。它能用来验证算子融合、分块与重算确实会改变物理 HBM 访问量。同图的前向实验还显示增大 block 时性能改善会趋于饱和。对芯片做容量设计，应记录每个并发 tile 真正可分配的片上空间，以及 register、SRAM、片外 DRAM 三者之间的流量，再用独立硬件计数器检验重复搬运。[FlashAttention 原论文图 2](https://arxiv.org/pdf/2205.14135#page=6)
