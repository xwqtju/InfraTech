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

## 2. Decode Core FLOPs

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

逐项 FLOPs：

| 步骤 | 公式 | FLOPs |
|---|---|---:|
| 衰减 state | `state = exp(g_t) * state` | `B * H * K * V` |
| 从 state 读已有 value | `v_mem = k_t^T state` | `2 * B * H * K * V` |
| 计算 residual | `delta = beta_t * (v_t - v_mem)` | `2 * B * H * V` |
| 外积写入 state | `state = state + k_t delta^T` | `2 * B * H * K * V` |
| query 读取输出 | `y_t = q_t^T state` | `2 * B * H * K * V` |

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

## 3. Prefill Core FLOPs

prefill 使用 chunk 算法。它不是逐 token recurrent 循环，而是：

```text
chunk 内:
  用三角矩阵并行展开 token 之间的 delta-rule 依赖

chunk 间:
  传递固定大小的 recurrent state
```

主要大项如下：

| 步骤 | 公式/操作 | FLOPs |
|---|---|---:|
| chunk 内 key 相关性 | `k_beta @ k^T` | `2 * B * H * N * C^2 * K` |
| 三角递推展开 | lower-triangular recurrence | `~ (2/3) * B * H * N * C^3` |
| 计算 chunk 内 residual value | `attn @ v_beta` | `2 * B * H * N * C^2 * V` |
| 计算旧 state 查询系数 | `attn @ (k_beta * exp(g))` | `2 * B * H * N * C^2 * K` |
| chunk 内读取矩阵 | `q @ k^T` | `2 * B * H * N * C^2 * K` |
| 旧 state 对写入的预测 | `k_cumdecay @ state` | `2 * B * H * N * C * K * V` |
| 从旧 state 读取 | `q @ state` | `2 * B * H * N * C * K * V` |
| 从当前 chunk 新写入读取 | `attn @ v_new` | `2 * B * H * N * C^2 * V` |
| 更新 state 的写入项 | `k^T @ v_new` | `2 * B * H * N * C * K * V` |

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

