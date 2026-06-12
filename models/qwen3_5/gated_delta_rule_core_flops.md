# Gated Delta Rule Core FLOPs 估算

本文只统计 **Gated Delta Rule core** 的计算量，不包含：

- `in_proj_qkv`
- `in_proj_z`
- `in_proj_a / in_proj_b`
- causal conv
- gated RMSNorm
- `out_proj`

也就是说，这里只估算 Gated Delta Rule 本身在 `prefill` 和 `decode` 场景下的 FLOPs，以及考虑 Tensor Parallelism 后单芯片上的 FLOPs。

## 1. 符号

```text
B: batch size
S: prefill sequence length
H: value heads 总数
K: key/query head dim
V: value head dim
C: prefill chunk_size，通常为 64
N: chunk 数，N = ceil(S / C)
P: TP size
```

Gated Delta Rule core 的主要张量形状：

```text
q:     [B, S, H, K]
k:     [B, S, H, K]
v:     [B, S, H, V]
beta:  [B, S, H]
g:     [B, S, H]
state: [B, H, K, V]
```

在 TP 按 head 切分时：

```text
H_local = H / P
state_local = [B, H/P, K, V]
```

因此 core FLOPs 通常可以按 head 维近似均分到每张芯片。

执行单元标注口径：

- `Cube`: 主要是矩阵乘、batched matmul、带 reduction 维度的 contraction。
- `Vector`: 主要是逐元素、broadcast、exp、cumsum、mask、加减、reshape/padding 等。
- `Cube + Vector`: 同一步里既有 matmul，也有逐元素门控、mask、衰减等操作。

实际 kernel 可能会做融合，表中的标注按主要算子形态归类。

## 2. Decode Core 输入输出与 FLOPs

decode 阶段通常每次处理一个新 token。单 token recurrent 公式是：

```text
state = exp(g_t) * state
v_mem = k_t^T state
delta = beta_t * (v_t - v_mem)
state = state + k_t delta^T
y_t = q_t^T state
```

对应 shape：

```text
q_t:   [B, H, K]
k_t:   [B, H, K]
v_t:   [B, H, V]
state: [B, H, K, V]
y_t:   [B, H, V]
```

逐项输入输出、公式和 FLOPs：

| 步骤 | 公式 | 输入 shape | 输出 shape | 功能 | 执行单元 | FLOPs |
|---|---|---:|---:|---|---|---:|
| 1 | `gamma_t = exp(g_t)` | `g_t: [B,H]` | `gamma_t: [B,H]` | 把 log decay 转成保留率 | Vector | `B * H` |
| 2 | `S'_t = gamma_t[...,None,None] * S_{t-1}` | `gamma_t: [B,H]`, `S_{t-1}: [B,H,K,V]` | `S'_t: [B,H,K,V]` | 对旧 state 做衰减 | Vector | `B * H * K * V` |
| 3 | `v_mem = k_t^T S'_t` | `k_t: [B,H,K]`, `S'_t: [B,H,K,V]` | `v_mem: [B,H,V]` | 用当前 key 读取 state 已经能预测出的 value | Cube | `2 * B * H * K * V` |
| 4 | `delta_t = beta_t[...,None] * (v_t - v_mem)` | `beta_t: [B,H]`, `v_t/v_mem: [B,H,V]` | `delta_t: [B,H,V]` | 只保留需要新写入的 residual | Vector | `2 * B * H * V` |
| 5 | `S_t = S'_t + k_t[...,None] * delta_t[...,None,:]` | `S'_t: [B,H,K,V]`, `k_t: [B,H,K]`, `delta_t: [B,H,V]` | `S_t: [B,H,K,V]` | rank-1/broadcast 外积写入 state | Vector | `2 * B * H * K * V` |
| 6 | `y_t = q_t^T S_t` | `q_t: [B,H,K]`, `S_t: [B,H,K,V]` | `y_t: [B,H,V]` | 用 query 从更新后的 state 读取输出 | Cube | `2 * B * H * K * V` |

所以 decode core 全量 FLOPs 为：

```text
FLOPs_decode_core
≈ B * H * (7 * K * V + 2 * V)
```

由于通常 `K * V >> V`，可以近似为：

```text
FLOPs_decode_core
≈ 7 * B * H * K * V
```

考虑 TP 后，单芯片 FLOPs：

```text
FLOPs_decode_core_per_chip
≈ B * (H / P) * (7 * K * V + 2 * V)
```

近似：

```text
FLOPs_decode_core_per_chip
≈ 7 * B * (H / P) * K * V
```

TP 按 head 切分时，decode core 在单芯片上的输入输出形状变为：

```text
q_t/k_t: [B, H/P, K]
v_t:     [B, H/P, V]
state:   [B, H/P, K, V]
y_t:     [B, H/P, V]
```

因此上表中所有包含 `H` 的 FLOPs 项都替换成 `H/P`。

## 3. Prefill Core 输入输出与 FLOPs

prefill 使用 chunk 算法。它不是逐 token recurrent 循环，而是：

```text
chunk 内:
  用三角矩阵并行展开 token 之间的 delta-rule 依赖

chunk 间:
  传递固定大小的 recurrent state
```

prefill core 在进入 chunk 计算前，会先把 sequence-first 张量转成 head-first，并按 chunk reshape：

```text
q/k:  [B,S,H,K] -> [B,H,S,K] -> [B,H,N,C,K]
v:    [B,S,H,V] -> [B,H,S,V] -> [B,H,N,C,V]
beta: [B,S,H]   -> [B,H,S]   -> [B,H,N,C]
g:    [B,S,H]   -> [B,H,S]   -> [B,H,N,C]
```

其中最后一个 chunk 如果不足 `C`，会 padding 到 `N * C`，最终输出再裁回 `S`。

### 3.1 Chunk 内预处理

chunk 内预处理一次性处理所有 `B * H * N` 个 chunk。

| 步骤 | 公式 | 输入 shape | 输出 shape | 功能 | 执行单元 | FLOPs |
|---|---|---:|---:|---|---|---:|
| 1 | `k_beta = k * beta[...,None]` | `k: [B,H,N,C,K]`, `beta: [B,H,N,C]` | `k_beta: [B,H,N,C,K]` | 把写入门乘到 key 上 | Vector | `B * H * N * C * K` |
| 2 | `v_beta = v * beta[...,None]` | `v: [B,H,N,C,V]`, `beta: [B,H,N,C]` | `v_beta: [B,H,N,C,V]` | 把写入门乘到 value 上 | Vector | `B * H * N * C * V` |
| 3 | `G_i = sum_{r=0}^{i} g_r` | `g: [B,H,N,C]` | `G: [B,H,N,C]` | 计算 chunk 内累计 log decay | Vector | `~ B * H * N * C` |
| 4 | `D_{i,j} = exp(G_i - G_j), i >= j` | `G: [B,H,N,C]` | `D: [B,H,N,C,C]` | 得到 chunk 内任意写入从 `j` 传播到 `i` 的保留率 | Vector | `~ B * H * N * C^2` |
| 5 | `A_{i,j} = - beta_i <k_i,k_j> D_{i,j}, i > j` | `k_beta: [B,H,N,C,K]`, `k: [B,H,N,C,K]`, `D: [B,H,N,C,C]` | `A: [B,H,N,C,C]` | 构造 delta-rule 的直接依赖矩阵 | Cube + Vector | `2 * B * H * N * C^2 * K` |
| 6 | `A_{i,:i} = A_{i,:i} + A_{i,:i} A_{:i,:i}` | `A: [B,H,N,C,C]` | `A: [B,H,N,C,C]` | 把直接依赖展开成总依赖 | Vector | `~ (2/3) * B * H * N * C^3` |
| 7 | `T = I + A` | `A: [B,H,N,C,C]` | `T: [B,H,N,C,C]` | 得到 chunk 内三角解算矩阵 | Vector | `B * H * N * C` |
| 8 | `u = T @ v_beta` | `T: [B,H,N,C,C]`, `v_beta: [B,H,N,C,V]` | `u: [B,H,N,C,V]` | 计算零初始 state 下的 chunk 内 residual value | Cube | `2 * B * H * N * C^2 * V` |
| 9 | `r = T @ (k_beta * exp(G)[...,None])` | `T: [B,H,N,C,C]`, `k_beta: [B,H,N,C,K]`, `G: [B,H,N,C]` | `r: [B,H,N,C,K]` | 计算旧 state 查询系数，即 `k_cumdecay` | Cube + Vector | `2 * B * H * N * C^2 * K` |

上表中 `u` 对应代码里被重写后的 `value`，`r` 对应 `k_cumdecay`。

### 3.2 Chunk 间循环

接下来对每个 chunk `n` 做循环。进入第 `n` 个 chunk 时：

```text
q_n/k_n: [B,H,C,K]
u_n:     [B,H,C,V]
r_n:     [B,H,C,K]
G_n:     [B,H,C]
state:   [B,H,K,V]
```

逐项输入输出、公式和 FLOPs：

| 步骤 | 公式 | 输入 shape | 输出 shape | 功能 | 执行单元 | FLOPs |
|---|---|---:|---:|---|---|---:|
| 10 | `R_{p,j} = <q_p,k_j> D_{p,j}` | `q_n/k_n: [B,H,C,K]`, `D_n: [B,H,C,C]` | `R: [B,H,C,C]` | 当前 chunk 内每个输出位置从 chunk 内写入读取多少 | Cube + Vector | `2 * B * H * C^2 * K` |
| 11 | `v_prime = r_n @ state` | `r_n: [B,H,C,K]`, `state: [B,H,K,V]` | `v_prime: [B,H,C,V]` | 旧 state 已经能解释出的 value | Cube | `2 * B * H * C * K * V` |
| 12 | `v_new = u_n - v_prime` | `u_n/v_prime: [B,H,C,V]` | `v_new: [B,H,C,V]` | 得到当前 chunk 真正需要写入的 residual | Vector | `B * H * C * V` |
| 13 | `y_old = (q_n * exp(G_n)[...,None]) @ state` | `q_n: [B,H,C,K]`, `G_n: [B,H,C]`, `state: [B,H,K,V]` | `y_old: [B,H,C,V]` | 从进入 chunk 前的历史 state 读取输出 | Cube + Vector | `2 * B * H * C * K * V` |
| 14 | `y_new = R @ v_new` | `R: [B,H,C,C]`, `v_new: [B,H,C,V]` | `y_new: [B,H,C,V]` | 从当前 chunk 新写入中读取输出 | Cube | `2 * B * H * C^2 * V` |
| 15 | `y_n = y_old + y_new` | `y_old/y_new: [B,H,C,V]` | `y_n: [B,H,C,V]` | 合成当前 chunk 输出 | Vector | `B * H * C * V` |
| 16 | `state_old = exp(G_last) * state` | `G_last: [B,H]`, `state: [B,H,K,V]` | `state_old: [B,H,K,V]` | 旧 state 衰减到 chunk 末尾 | Vector | `B * H * K * V` |
| 17 | `state_write = (k_n * exp(G_last - G_n)[...,None])^T @ v_new` | `k_n: [B,H,C,K]`, `G_n: [B,H,C]`, `v_new: [B,H,C,V]` | `state_write: [B,H,K,V]` | 当前 chunk 的 residual 写入累加到 chunk 末尾 | Cube + Vector | `2 * B * H * C * K * V` |
| 18 | `state_next = state_old + state_write` | `state_old/state_write: [B,H,K,V]` | `state_next: [B,H,K,V]` | 更新 state，传给下一个 chunk | Vector | `B * H * K * V` |

因为 chunk 间循环执行 `N` 次，所以步骤 10 到 18 的主要 matmul FLOPs 需要再乘以 `N`。

最终 prefill core 输出：

```text
y:           [B,H,N,C,V] -> [B,H,S,V] -> [B,S,H,V]
final_state: [B,H,K,V]
```

如果启用 cache，`final_state` 会作为 decode 的初始 recurrent state。

TP 按 head 切分时，prefill core 在单芯片上的主要形状为：

```text
q/k:         [B, H/P, N, C, K]
v/u/y:       [B, H/P, N, C, V]
state:       [B, H/P, K, V]
final_state: [B, H/P, K, V]
```

因此 prefill 表中所有包含 `H` 的 FLOPs 项都替换成 `H/P`。

### 3.3 Prefill FLOPs 合并

合并后，prefill core 全量 FLOPs 近似为：

```text
FLOPs_prefill_core
≈ B * H * N * [
    6 * C^2 * K
  + 4 * C^2 * V
  + 6 * C * K * V
  + (2/3) * C^3
]
```

如果 `K = V = D`，可写成：

```text
FLOPs_prefill_core
≈ B * H * N * [
    10 * C^2 * D
  + 6 * C * D^2
  + (2/3) * C^3
]
```

如果 `S` 正好是 `C` 的倍数，则 `N * C = S`，也可以写成：

```text
FLOPs_prefill_core
≈ B * H * S * [
    10 * C * D
  + 6 * D^2
  + (2/3) * C^2
]
```

考虑 TP 后，单芯片 FLOPs：

```text
FLOPs_prefill_core_per_chip
≈ B * (H / P) * N * [
    6 * C^2 * K
  + 4 * C^2 * V
  + 6 * C * K * V
  + (2/3) * C^3
]
```

也就是：

```text
FLOPs_prefill_core_per_chip
≈ FLOPs_prefill_core / P
```

## 4. 系数说明

普通矩阵乘：

```text
[A, B] @ [B, C] -> [A, C]
```

每个输出元素大约需要：

```text
B 次乘法 + B 次加法
```

所以 FLOPs 估算为：

```text
2 * A * B * C
```

这就是大部分 matmul 项前面系数为 `2` 的原因。如果按 MACs 统计，则这些项前面的 `2` 需要去掉。

三角递推展开来自代码：

```python
for i in range(1, C):
    row = attn[..., i, :i]
    sub = attn[..., :i, :i]
    attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
```

第 `i` 轮大约有：

```text
i^2 次乘法 + i^2 次加法 = 2 * i^2 FLOPs
```

因此总 FLOPs：

```text
sum_{i=1}^{C-1} 2 * i^2
= 2 * (C-1) * C * (2C-1) / 6
≈ (2/3) * C^3
```

所以三角递推展开项为：

```text
~ (2/3) * B * H * N * C^3
```

## 5. 汇总表

| 场景 | 全量 Gated Delta Rule core FLOPs | TP 后单芯片 FLOPs |
|---|---:|---:|
| decode 单 token | `B * H * (7KV + 2V)` | `B * (H/P) * (7KV + 2V)` |
| decode 近似 | `7 * B * H * K * V` | `7 * B * (H/P) * K * V` |
| prefill | `BHN[6C^2K + 4C^2V + 6CKV + (2/3)C^3]` | `B(H/P)N[6C^2K + 4C^2V + 6CKV + (2/3)C^3]` |
| prefill, `K=V=D` | `BHN[10C^2D + 6CD^2 + (2/3)C^3]` | `B(H/P)N[10C^2D + 6CD^2 + (2/3)C^3]` |

## 6. 示例

取一组常见配置：

```text
B = 1
S = 4096
H = 64
K = 128
V = 128
C = 64
N = 64
P = 8
```

decode core 全量：

```text
FLOPs_decode_core
≈ 7 * 1 * 64 * 128 * 128
≈ 7.34M FLOPs
```

decode core 单芯片：

```text
FLOPs_decode_core_per_chip
≈ 7.34M / 8
≈ 0.92M FLOPs
```

prefill core 全量：

```text
FLOPs_prefill_core
≈ 1 * 64 * 64 * [
    10 * 64^2 * 128
  + 6 * 64 * 128^2
  + (2/3) * 64^3
]
≈ 47.96B FLOPs
```

prefill core 单芯片：

```text
FLOPs_prefill_core_per_chip
≈ 47.96B / 8
≈ 6.0B FLOPs
```

## 7. 结论

decode core 是固定状态单步更新：

```text
O(B * H * K * V)
```

prefill core 是 chunk 内并行扫描：

```text
O(B * H * N * (C^2K + C^2V + CKV + C^3))
```

当 TP 按 head 维切分时，Gated Delta Rule core 的 `state` 和计算都按 `H` 维切开，所以单芯片 FLOPs 通常近似等于全量 FLOPs 除以 `P`。
