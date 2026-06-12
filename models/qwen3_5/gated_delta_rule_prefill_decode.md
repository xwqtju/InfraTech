# Gated Delta Rule 在 prefill 和 decode 中的执行过程

本文以知乎帖子 [Gated Delta Rule 讲解](https://zhuanlan.zhihu.com/p/2012544428099794206) 为入口，结合 Gated DeltaNet 论文和本仓库里的 Qwen3.5 实现，解释 Gated Delta Rule 在推理时的两条路径：

- `prefill`: 一次处理整段 prompt，尽量用 chunk/parallel 形式计算。
- `decode`: 已经有缓存后，每次只处理一个新 token，用 recurrent state 做常数缓存更新。

对应代码主要在：

- `models/qwen3_5/modeling_qwen3_5.py`
- `models/qwen3_5/modeling_qwen3_5_moe.py`
- `models/qwen3_5/gated_deltanet_io.md`

## 1. 先建立直觉

标准 causal attention 的记忆是不断增长的 KV cache：

```text
K_cache: [T, num_heads, head_dim]
V_cache: [T, num_heads, head_dim]
```

decode 第 `t` 步要把当前 query 和历史所有 key 做匹配，因此缓存大小随 `T` 增长，读历史的计算也和上下文长度相关。

Gated Delta Rule 换了一种记忆方式：每个 value head 不缓存所有历史 token，而是维护一个固定大小的矩阵状态：

```text
S_t: [head_k_dim, head_v_dim]
```

在 Qwen3.5 的 Gated DeltaNet 中，常见维度是：

```text
head_k_dim = 128
head_v_dim = 128
num_v_heads = 32 / 48 / 64，取决于具体模型配置
```

如果以 Qwen3.5 397B-A17B 架构图里的 `num_v_heads = 64` 为例，每层 recurrent state 是：

```text
[B, 64, 128, 128]
```

这就是 Gated DeltaNet decode 能做到固定大小缓存的关键。

可以把 `S_t` 理解成一张不断被编辑的“key 到 value 的线性映射表”：

- `k_t` 决定这次要改哪个方向的记忆。
- `v_t` 是希望写进去的新内容。
- `q_t` 决定从这张表里读出什么。
- `beta_t` 控制这次 delta 写入的强度。
- `g_t` 控制旧状态保留多少，也就是遗忘门。

## 2. 单 token 的核心公式

对第 `t` 个 token，令：

```text
q_t: [Dk]
k_t: [Dk]
v_t: [Dv]
S_{t-1}: [Dk, Dv]
beta_t: scalar 或 per-head scalar，范围约为 (0, 1)
gamma_t = exp(g_t)，范围约为 (0, 1)
```

本仓库 fallback recurrent 实现对应以下步骤：

```text
1. 先遗忘旧状态
   S'_t = gamma_t * S_{t-1}

2. 用当前 key 从旧状态中读出“已经记住的 value”
   v_mem = k_t^T S'_t

3. 计算差值，只写入还没记住或需要修正的部分
   delta_t = beta_t * (v_t - v_mem)

4. 用外积把差值写回状态
   S_t = S'_t + k_t delta_t^T

5. 用 query 从更新后的状态中读输出
   y_t = q_t^T S_t
```

这里的重点是第 3 步。普通线性 attention 常见写法更像直接累加 `k_t v_t^T`，而 Delta Rule 写的是：

```text
k_t (v_t - k_t^T S'_t)^T
```

也就是“先看当前状态沿 `k_t` 方向已经能预测出什么，再只写误差”。这会减少重复覆盖，使记忆编辑更像在线回归中的一次残差更新。

Gated DeltaNet 论文的核心观察是：遗忘门和 delta 更新是互补的。遗忘门让模型能快速擦掉不再需要的历史，delta 更新让写入更有针对性。

## 3. Gated DeltaNet 模块里的前处理

进入 Gated Delta Rule 前，Qwen3.5 的 Gated DeltaNet 会先从 `hidden_states` 生成几组量。

以本仓库实现为准，主干流程是：

```text
hidden_states: [B, S, hidden_size]

in_proj_qkv(hidden_states)
  -> mixed_qkv: [B, S, key_dim + key_dim + value_dim]

causal depthwise conv + SiLU
  -> mixed_qkv: [B, S, key_dim + key_dim + value_dim]

split
  -> query: [B, S, num_k_heads, head_k_dim]
  -> key:   [B, S, num_k_heads, head_k_dim]
  -> value: [B, S, num_v_heads, head_v_dim]

in_proj_b(hidden_states).sigmoid()
  -> beta: [B, S, num_v_heads]

in_proj_a(hidden_states), A_log, dt_bias
  -> g: [B, S, num_v_heads]

if num_v_heads > num_k_heads:
  repeat_interleave query/key 到 num_v_heads
```

这里有两个容易混淆的点：

1. `g` 在代码里不是直接的保留率，而是 log-space 的负数门控；真正参与 recurrent 更新的是 `exp(g)`。
2. `beta` 和 `g` 在实现中是 per value head 的标量，不是整层共享的单个标量。

代码里对应：

```python
beta = b.sigmoid()
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
```

因为 `A_log.exp()` 和 `softplus(...)` 都是非负，所以 `g <= 0`，进而：

```text
0 < exp(g) <= 1
```

这保证了它天然是一个衰减因子。

## 4. Prefill：整段 prompt 的 chunk 计算

prefill 发生在模型第一次看到 prompt 时。此时通常有一段长度为 `S` 的输入，还没有该层的历史 recurrent state，或者只有上一次请求遗留下来的 chunk continuation state。

本仓库里的分支条件是：

```text
如果没有历史 state，或者 seq_len > 1：
    走 chunk_gated_delta_rule
```

也就是：

```python
core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(...)
```

### 4.1 prefill 的输入准备

prefill 先一次性投影整段 prompt：

```text
hidden_states: [B, S, H]
q, k, v:       [B, S, Hv, D]
beta, g:       [B, S, Hv]
```

其中：

- `H` 是 hidden size。
- `Hv` 是 value heads 数。
- `D` 在当前实现里通常为 128。

然后 Q/K 会做 L2Norm，query 会乘上 scale：

```text
q = l2norm(q) / sqrt(D)
k = l2norm(k)
```

如果启用了 cache，prefill 结束时会保存两类状态：

```text
conv_state:      [B, conv_dim, conv_kernel_size - 1]
recurrent_state: [B, Hv, Dk, Dv]
```

`conv_state` 用来让下一步 decode 的 causal conv 还能看到左侧短程上下文；`recurrent_state` 用来承接 Gated Delta Rule 的长期压缩记忆。

### 4.2 为什么 prefill 不直接逐 token 循环

单 token 公式天然是递推的：

```text
S_0 -> S_1 -> S_2 -> ... -> S_T
```

如果 prefill 按 token 循环，长 prompt 会很慢。Gated DeltaNet 的工程重点之一，就是把这个递推改写成 chunk 内并行、chunk 间递推的形式。

本仓库 fallback 版本默认：

```text
chunk_size = 64
```

流程大致是：

```text
1. 把序列 pad 到 64 的倍数。
2. reshape 成多个 chunk：
   [B, Hv, num_chunks, 64, D]
3. 在每个 chunk 内构造 lower-triangular 的因果关系。
4. 预计算 chunk 内的衰减和 delta 修正。
5. chunk 与 chunk 之间只传递一个 recurrent_state。
```

也就是说，prefill 不是完全没有递推，而是把递推粒度从 token 降到 chunk：

```text
token 级递推:
  token1 -> token2 -> token3 -> ... -> tokenT

chunk 级递推:
  chunk1 -> chunk2 -> chunk3 -> ...
```

chunk 内尽量用矩阵乘法并行处理，这更适合 GPU。

### 4.3 chunk 内发生了什么

fallback 代码会先转成：

```text
q, k, v: [B, Hv, S, D]
beta,g:  [B, Hv, S]
```

再构造：

```text
v_beta = v * beta
k_beta = k * beta
```

这对应单步公式里的：

```text
delta_t = beta_t * (...)
```

随后每个 chunk 内会计算 cumulative decay：

```text
g_cumsum[i] = g[0] + g[1] + ... + g[i]
decay(i, j) = exp(g_cumsum[i] - g_cumsum[j]), j <= i
```

直觉上，`j` 时刻写入的内容传到 `i` 时刻时，中间每一步都要乘遗忘门，所以需要累积衰减。

chunk 内还会构造一个因果 lower-triangular 的 `attn` 矩阵。它不是 softmax attention，而是把 Delta Rule 中“后面的写入要扣掉前面状态已预测内容”的递推关系，改写成三角矩阵求解/扫描形式。

可以把它理解成：在同一个 chunk 里，第 `i` 个 token 的 delta 会受 `0..i-1` token 已经写入内容影响。这个影响如果逐 token 算就是 recurrent；改成三角矩阵后，就能用批量矩阵乘法一次性算出 chunk 内所有 token 的等价修正。

### 4.4 chunk 与 chunk 之间如何传 state

chunk 内并行算完后，每个 chunk 仍然需要知道进入该 chunk 之前的状态：

```text
last_recurrent_state: [B, Hv, Dk, Dv]
```

对第 `i` 个 chunk，代码做三件事：

1. 计算来自历史 state 的读出贡献：

```text
attn_inter = (q_i * exp(cumulative_g_i)) @ last_recurrent_state
```

这表示 chunk 内每个位置从 chunk 之前的旧状态读到的内容，同时考虑从 chunk 开始到当前位置的遗忘衰减。

2. 计算 chunk 内新 token 之间的贡献：

```text
attn @ v_new
```

其中 `v_new` 是扣除了旧 state 已能预测部分之后的 value 修正。

3. 更新 chunk 结束后的 recurrent state：

```text
last_recurrent_state =
    old_state_after_decay
    + chunk_writes_after_decay_alignment
```

这样下一个 chunk 只需要接收一个固定大小的 `last_recurrent_state`，不需要接收前面所有 token 的 K/V。

### 4.5 prefill 的输出

prefill 输出两类结果：

```text
core_attn_out:       [B, S, Hv, Dv]
last_recurrent_state:[B, Hv, Dk, Dv]
```

随后模块会做：

```text
core_attn_out reshape 到二维
RMSNormGated(core_attn_out, z)
reshape 回 [B, S, Hv * Dv]
out_proj -> [B, S, hidden_size]
```

如果 `use_cache=True`，`last_recurrent_state` 会写入 cache，成为 decode 的初始状态。

## 5. Decode：单 token recurrent 更新

decode 发生在 prompt 已经 prefill 完、模型开始逐 token 生成时。此时每层 cache 里已经有：

```text
conv_state
recurrent_state
```

本仓库里的单 token decode 分支条件是：

```text
use_precomputed_states and seq_len == 1
```

此时会走：

```python
mixed_qkv = self.causal_conv1d_update(...)
core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(...)
```

### 5.1 decode 的 causal conv

Gated DeltaNet 在 Q/K/V 进入 Delta Rule 前有一层 depthwise causal conv。prefill 时可以对整段序列做 conv；decode 时只有一个新 token，所以必须从 cache 里取上一轮留下的短窗口：

```text
conv_state: [B, conv_dim, conv_kernel_size - 1]
```

`causal_conv1d_update` 会：

1. 把新 token 的 projected qkv 拼到 conv window 后面。
2. 做一次 causal conv + activation。
3. 原地更新 `conv_state`，保留最新的 `kernel_size - 1` 个位置。

这一步解决的是短程局部混合，不是长期记忆。长期记忆在 `recurrent_state` 里。

### 5.2 decode 的 Gated Delta Rule

单 token decode 就是第 2 节公式的直接实现。以代码中的变量命名：

```text
last_recurrent_state = recurrent_state

g_t    = exp(g[:, :, i])
beta_t = beta[:, :, i]
q_t    = query[:, :, i]
k_t    = key[:, :, i]
v_t    = value[:, :, i]
```

执行过程：

```text
1. 衰减旧状态
   last_recurrent_state *= g_t

2. 用当前 key 从状态读出已有 value
   kv_mem = sum(last_recurrent_state * k_t[..., None], dim=-2)

3. 算残差写入
   delta = (v_t - kv_mem) * beta_t

4. 外积写状态
   last_recurrent_state += k_t[..., None] * delta[..., None, :]

5. 当前 query 从新状态读输出
   y_t = sum(last_recurrent_state * q_t[..., None], dim=-2)
```

输出 `y_t` 的形状是：

```text
[B, Hv, Dv]
```

再经过 gated RMSNorm、flatten、`out_proj`，回到：

```text
[B, 1, hidden_size]
```

最后新的 `last_recurrent_state` 会写回 cache，供下一个 decode token 使用。

### 5.3 decode 为什么是常数大小缓存

decode 每一步只保留：

```text
conv_state:      [B, conv_dim, kernel_size - 1]
recurrent_state: [B, Hv, Dk, Dv]
```

它不会保存：

```text
K_cache: [B, T, heads, D]
V_cache: [B, T, heads, D]
```

因此 Gated DeltaNet 的线性注意力层在 decode 阶段的长期状态大小不随上下文长度 `T` 增长。

需要注意：Qwen3.5 是 hybrid 架构，部分层仍是 Gated Attention。那些 full attention 层仍然会有自己的 KV cache。Gated DeltaNet 层本身的长期记忆是固定大小的。

## 6. Prefill 和 decode 的对照

| 项目 | prefill | decode |
|---|---|---|
| 输入长度 | 一段 prompt，`S >= 1` | 通常单个新 token，`S = 1` |
| 是否已有 state | 通常没有；chunk continuation 时可能有 | 有 |
| 主要 kernel | `chunk_gated_delta_rule` | `recurrent_gated_delta_rule` |
| conv 处理 | 整段 causal conv；有旧 state 时 prepend conv_state | `causal_conv1d_update` 单步更新 |
| Delta Rule 形态 | chunk 内并行，chunk 间递推 | 直接单步递推 |
| 长期缓存 | 输出最终 `recurrent_state` | 读取并更新 `recurrent_state` |
| 复杂度直觉 | 对序列长度线性，GPU 友好的 chunk 计算 | 每 token 固定大小状态更新 |
| 是否保存所有历史 KV | 否 | 否 |

## 7. 用伪代码串起来

### 7.1 prefill 伪代码

```python
def prefill(hidden_states, cache=None):
    qkv = in_proj_qkv(hidden_states)
    z = in_proj_z(hidden_states)
    beta = sigmoid(in_proj_b(hidden_states))
    g = -exp(A_log) * softplus(in_proj_a(hidden_states) + dt_bias)

    qkv = causal_conv_full_sequence(qkv, cache.conv_state if cache else None)
    q, k, v = split_and_reshape(qkv)
    q, k = repeat_qk_to_value_heads(q, k)

    y, recurrent_state = chunk_gated_delta_rule(
        q, k, v,
        g=g,
        beta=beta,
        initial_state=cache.recurrent_state if cache_has_state else None,
        output_final_state=cache is not None,
    )

    if cache is not None:
        cache.conv_state = last_conv_window(qkv)
        cache.recurrent_state = recurrent_state

    y = gated_rmsnorm(y, z)
    return out_proj(flatten_heads(y))
```

### 7.2 decode 伪代码

```python
def decode_one_token(hidden_state_t, cache):
    qkv_t = in_proj_qkv(hidden_state_t)
    z_t = in_proj_z(hidden_state_t)
    beta_t = sigmoid(in_proj_b(hidden_state_t))
    g_t = -exp(A_log) * softplus(in_proj_a(hidden_state_t) + dt_bias)

    qkv_t = causal_conv1d_update(qkv_t, cache.conv_state)
    q_t, k_t, v_t = split_and_reshape(qkv_t)
    q_t, k_t = repeat_qk_to_value_heads(q_t, k_t)

    y_t, recurrent_state = recurrent_gated_delta_rule(
        q_t, k_t, v_t,
        g=g_t,
        beta=beta_t,
        initial_state=cache.recurrent_state,
        output_final_state=True,
    )

    cache.recurrent_state = recurrent_state

    y_t = gated_rmsnorm(y_t, z_t)
    return out_proj(flatten_heads(y_t))
```

## 8. 一句话总结

Gated Delta Rule 的本质是：用一个固定大小的矩阵状态 `S` 表示历史，把每个新 token 变成一次“先遗忘、再读取已有记忆、再写入残差、最后读取输出”的在线更新。

prefill 阶段为了吞吐，把这个在线更新改写成 chunk 内并行、chunk 间传 state；decode 阶段则回到最直接的单步 recurrent 更新，因此每个 Gated DeltaNet 层只需要固定大小的 recurrent state，而不需要随上下文增长的 KV cache。

## 参考资料

- 知乎帖子入口：[https://zhuanlan.zhihu.com/p/2012544428099794206](https://zhuanlan.zhihu.com/p/2012544428099794206)
- Gated Delta Networks: Improving Mamba2 with Delta Rule, arXiv:2412.06464: [https://arxiv.org/abs/2412.06464](https://arxiv.org/abs/2412.06464)
- Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention, arXiv:2605.22791: [https://arxiv.org/abs/2605.22791](https://arxiv.org/abs/2605.22791)
- 本仓库 Qwen3.5 Gated DeltaNet 维度拆解：[gated_deltanet_io.md](./gated_deltanet_io.md)
