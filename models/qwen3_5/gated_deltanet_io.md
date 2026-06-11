# Qwen3.5 Gated DeltaNet 输入输出维度拆解

本文基于 `models/qwen3_5/README.md` 中 Qwen3.5 397B-A17B 架构图，对绿色框里的 Gated DeltaNet 模块做输入输出维度拆解。

记号说明：

- `S`: sequence length
- `H`: hidden size，Qwen3.5 397B-A17B 中为 4096
- `Nqk`: Q/K 线性注意力头数，图中为 16
- `Nv`: V 线性注意力头数，图中为 64
- `D`: 每个线性注意力头维度，图中为 128
- batch 维度在表中省略；如果带 batch，则在最前面加 `[B, ...]`

## 主流程输入输出

| 步骤 | 操作 | 输入 | 输出 | 说明 |
|---|---:|---:|---:|---|
| 1 | 输入 hidden states | `[S, 4096]` | `[S, 4096]` | 来自上一层或 embedding，进入 Gated DeltaNet |
| 2 | Q 线性投影 | `[S, 4096]` | `[S, 16, 128]` | 生成 query，用于从 Delta 状态中读取信息 |
| 3 | K 线性投影 | `[S, 4096]` | `[S, 16, 128]` | 生成 key，用于定位和写入 Delta 状态 |
| 4 | V 线性投影 | `[S, 4096]` | `[S, 64, 128]` | 生成 value，作为写入状态的信息 |
| 5 | beta 线性投影 | 图示为 `[S, 4096] @ [4096, 1]` | 图示为 `[S, 1]` | 图中画成 scalar gate；若按公开实现则为 `[S, 64]`，即 per-value-head gate |
| 6 | alpha/g 线性投影 | 图示为 `[S, 4096] @ [4096, 1]` | 图示为 `[S, 1]` | 图中画成 scalar gate；若按公开实现则为 `[S, 64]`，即 per-value-head decay gate |
| 7 | z 输出门线性投影 | `[S, 4096]` | `[S, 64, 128]` | 生成输出门控，用于控制 DeltaNet 输出通道强度 |
| 8 | Q causal Conv + SiLU | `[S, 16, 128]` | `[S, 16, 128]` | 给 Q 混入短程局部上下文 |
| 9 | K causal Conv + SiLU | `[S, 16, 128]` | `[S, 16, 128]` | 给 K 混入短程局部上下文 |
| 10 | V causal Conv + SiLU | `[S, 64, 128]` | `[S, 64, 128]` | 给 V 混入短程局部上下文 |
| 11 | Q L2Norm | `[S, 16, 128]` | `[S, 16, 128]` | 稳定 query 向量长度 |
| 12 | K L2Norm | `[S, 16, 128]` | `[S, 16, 128]` | 稳定 key 向量长度 |
| 13 | Q/K head repeat | `[S, 16, 128]` | `[S, 64, 128]` | 将 16 个 Q/K head 扩展到 64 个 value head 对齐 |
| 14 | beta sigmoid | 图示 `[S, 1]`；实现 `[S, 64]` | 图示 `[S, 1]`；实现 `[S, 64]` | 得到写入强度，范围在 0 到 1 |
| 15 | alpha/g 衰减门计算 | 图示 `[S, 1]`；实现 `[S, 64]` | 图示 `[S, 1]`；实现 `[S, 64]` | 得到状态衰减或遗忘系数 |
| 16 | Gated Delta Rule | `Q,K,V: [S,64,128]`，`alpha/g,beta: [S,1]` 或 `[S,64]` | `[S,64,128]` | 核心线性递推记忆：衰减旧状态、写入新差值、读取输出 |
| 17 | RMSNorm | `[S,64,128]` | `[S,64,128]` | 对 DeltaNet 输出做归一化 |
| 18 | z 输出门控 | `Y: [S,64,128]`，`z: [S,64,128]` | `[S,64,128]` | 用 `SiLU(z)` 控制每个通道的输出强弱 |
| 19 | reshape / flatten | `[S,64,128]` | `[S,8192]` | 把多头输出拼回单个向量 |
| 20 | 输出线性投影 | `[S,8192]` | `[S,4096]` | 投影回模型 hidden size |
| 21 | 外层残差连接 | `[S,4096] + [S,4096]` | `[S,4096]` | 回到 decoder block 主干 |

## Gated Delta Rule 子步骤

对第 `t` 个 token，每个 value head 维护一个状态矩阵：

```text
S_t: [128, 128]
```

如果把 64 个 value head 一起看，状态为：

```text
State_t: [64, 128, 128]
```

| 子步骤 | 公式 | 输入 | 输出 | 说明 |
|---|---|---:|---:|---|
| 16.1 | `S'_t = alpha_t * S_{t-1}` | `alpha_t: [64]` 或 broadcast 后 `[64]`，旧状态 `[64,128,128]` | `[64,128,128]` | 先对旧状态做衰减 |
| 16.2 | `v_mem = k_t^T S'_t` | `k_t: [64,128]`，`S'_t: [64,128,128]` | `[64,128]` | 用当前 key 读取状态中已有的 value 预测 |
| 16.3 | `delta = beta_t * (v_t - v_mem)` | `beta_t: [64]` 或 broadcast 后 `[64]`，`v_t/v_mem: [64,128]` | `[64,128]` | 计算需要新写入的差值 |
| 16.4 | `S_t = S'_t + k_t \otimes delta` | `k_t: [64,128]`，`delta: [64,128]` | `[64,128,128]` | 用外积把差值写回状态 |
| 16.5 | `y_t = q_t^T S_t` | `q_t: [64,128]`，`S_t: [64,128,128]` | `[64,128]` | 用 query 从新状态中读出当前 token 输出 |

## 维度差异说明

架构图中 `alpha/g` 和 `beta` 的线性层标为 `W [4096, 1]`，因此严格按图读，它们的输出是 `[S, 1]`，之后可广播到 64 个 value heads。

公开 Transformers 实现中，`alpha/g` 和 `beta` 是 per-value-head gate：

```python
in_proj_b: Linear(hidden_size, num_v_heads)
in_proj_a: Linear(hidden_size, num_v_heads)
```

由于 Qwen3.5 397B-A17B 的 `num_v_heads = 64`，所以实现中的输出是：

```text
beta:    [S, 64]
alpha/g: [S, 64]
```

因此可以这样理解：图中更像是简化画法或 scalar gate 表达；实现中则为每个 value head 单独生成门控。