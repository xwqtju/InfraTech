# Gated Delta Rule Prefill 过程详解

本文专门解释 `torch_chunk_gated_delta_rule` 在 prefill 阶段的 chunk 计算过程，重点对应两部分：

- chunk 内预处理：把 token 级递推关系改写成三角矩阵计算。
- chunk 间循环：每个 chunk 与进入 chunk 前的 recurrent state 合并，生成输出并更新 state。

对应代码在：

- `models/qwen3_5/modeling_qwen3_5.py`
- `models/qwen3_5/modeling_qwen3_5_moe.py`

## 1. 符号和形状

下文使用以下符号：

```text
B: batch size
H: value heads 数
N: chunk 数
C: chunk_size，代码里默认 64
K: key/query head dim
V: value head dim

q, k: [B, H, N, C, K]
v:    [B, H, N, C, V]
g:    [B, H, N, C]
beta: [B, H, N, C]
state:[B, H, K, V]
```

其中 `state` 是 Gated Delta Rule 的压缩历史记忆。对每个 head，它是一个矩阵：

```text
S: [K, V]
```

可以理解成一张从 key 空间到 value 空间的线性映射表。

单 token recurrent 公式是：

```text
S'_t     = exp(g_t) * S_{t-1}
v_mem_t  = k_t^T S'_t
delta_t  = beta_t * (v_t - v_mem_t)
S_t      = S'_t + k_t delta_t^T
y_t      = q_t^T S_t
```

prefill 的目标不是改变这个公式，而是把 token 级循环：

```text
token 0 -> token 1 -> token 2 -> ... -> token T
```

改写成：

```text
chunk 0 -> chunk 1 -> chunk 2 -> ...
```

chunk 内尽量用矩阵乘法并行，chunk 间只传一个固定大小的 `state`。

## 2. Chunk 内预处理

### 2.1 `g.cumsum(-1)`

代码：

```python
g = g.cumsum(dim=-1)
```

形状：

```text
[B, H, N, C] -> [B, H, N, C]
```

原始 `g` 是 log-space decay。代码中通常有：

```text
g <= 0
0 < exp(g) <= 1
```

`cumsum` 后，第 `i` 个位置得到：

```text
G_i = g_0 + g_1 + ... + g_i
```

它表示从 chunk 开头到当前位置的累计 log 衰减。

### 2.2 `g.unsqueeze(-1) - g.unsqueeze(-2)`

代码逻辑：

```python
g.unsqueeze(-1) - g.unsqueeze(-2)
```

形状：

```text
[B,H,N,C,1] - [B,H,N,1,C] -> [B,H,N,C,C]
```

矩阵中第 `(i, j)` 个元素是：

```text
G_i - G_j
```

含义是：第 `j` 个 token 写入的信息传到第 `i` 个 token 时，需要经过多少累计衰减。

### 2.3 `.tril().exp().tril() -> decay_mask`

代码：

```python
decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
```

形状：

```text
[B,H,N,C,C] -> [B,H,N,C,C]
```

结果可以理解为：

```text
decay_mask[i,j] = exp(G_i - G_j), 仅保留 i >= j
```

也就是说：

```text
j 位置的信息传播到 i 位置时的累计保留率
```

只保留下三角，是因为 causal 结构中后面的 token 不能影响前面的 token。

### 2.4 `k_beta @ key^T`

代码：

```python
v_beta = value * beta.unsqueeze(-1)
k_beta = key * beta.unsqueeze(-1)
attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
```

形状：

```text
k_beta: [B,H,N,C,K]
key^T:  [B,H,N,K,C]

k_beta @ key^T -> [B,H,N,C,C]
```

矩阵元素可以粗略写成：

```text
beta_i * <k_i, k_j>
```

再乘上 `decay_mask[i,j]` 后：

```text
beta_i * <k_i, k_j> * decay(i,j)
```

前面的负号来自 delta rule 的残差写入：

```text
delta_i = beta_i * (v_i - k_i^T S_before_i)
```

`S_before_i` 中包含前面 token 已经写入的内容，所以当前 token 需要减掉前面 token 已经能预测出的部分。这个“减掉旧预测”的影响被编码进 `attn`。

`mask` 会把对角线和上三角清零，所以这一步之后：

```text
attn[i,j] != 0 only if i > j
```

第 `i` 个 token 只依赖它之前的 token。

### 2.5 三角递推展开

代码：

```python
for i in range(1, chunk_size):
    row = attn[..., i, :i].clone()
    sub = attn[..., :i, :i].clone()
    attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
```

这是最关键、也最容易绕的一步。

循环前：

```text
attn[i,j] 表示 i 直接受 j 的影响
```

但是 delta rule 里：

```text
delta_i 依赖 delta_0 ... delta_{i-1}
delta_{i-1} 又依赖更早的 delta
```

所以除了直接影响，还有间接影响。例如：

```text
0 -> 1 -> 3
0 -> 2 -> 3
0 -> 1 -> 2 -> 3
```

这段循环把这些间接路径也折叠到 `attn[i, :i]` 里。

循环后：

```text
attn[i,j] 表示 j 对 i 的总影响，包括中间 token 传递过来的间接影响
```

可以把它理解成 lower-triangular recurrence 的前向求解。它不是 softmax attention。

### 2.6 `attn = attn + eye(C)`

代码：

```python
attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
```

形状不变：

```text
[B,H,N,C,C]
```

给对角线加 1 后，`attn` 成为 chunk 内的 delta 解算矩阵：

```text
当前位置自己的原始写入 + 前面 token 对当前位置的总修正
```

### 2.7 `attn @ v_beta -> value`

代码：

```python
value = attn @ v_beta
```

形状：

```text
[B,H,N,C,C] @ [B,H,N,C,V] -> [B,H,N,C,V]
```

这一步得到的是：

```text
如果进入这个 chunk 前的 state 为 0，chunk 内每个位置应该写入的 residual value
```

注意，这里的 `value` 已经不是原始 `v`，而是经过 chunk 内 delta-rule 修正后的写入量。

### 2.8 `attn @ (k_beta * exp(g)) -> k_cumdecay`

代码：

```python
k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
```

形状：

```text
[B,H,N,C,C] @ [B,H,N,C,K] -> [B,H,N,C,K]
```

它不是最终输出，而是后面扣除历史 `state` 影响时用的辅助量。

直觉上：

```text
k_cumdecay 表示当前 chunk 内每个位置，应该如何用 key 去查询进入 chunk 前的旧 state
```

## 3. Chunk 间循环

chunk 内预处理结束后，代码开始按 chunk 循环：

```python
for i in range(0, total_sequence_length // chunk_size):
    q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
```

形状：

```text
q_i: [B,H,C,K]
k_i: [B,H,C,K]
v_i: [B,H,C,V]
```

此时 `v_i` 已经是上一节中 `attn @ v_beta` 后的 residual value。

每个 chunk 进入时都有：

```text
last_recurrent_state: [B,H,K,V]
```

它代表这个 chunk 之前所有历史 token 压缩后的状态。

### 3.1 `q_i @ k_i^T * decay_mask`

代码：

```python
attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
```

形状：

```text
[B,H,C,K] @ [B,H,K,C] -> [B,H,C,C]
```

含义：

```text
chunk 内第 p 个输出，从 chunk 内第 j 个写入读取多少
```

这里乘 `decay_mask`，是因为 chunk 内较早写入的信息传播到较晚位置时仍然要被遗忘门衰减。

### 3.2 `k_cumdecay_i @ state`

代码：

```python
v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
```

形状：

```text
[B,H,C,K] @ [B,H,K,V] -> [B,H,C,V]
```

这一步计算：

```text
进入 chunk 前的旧 state，已经能为当前 chunk 的每个位置预测出多少 value
```

因为 delta rule 写的是残差：

```text
v_t - k_t^T state
```

所以旧 state 已经预测出的部分要从当前 chunk 的写入里扣掉。

### 3.3 `v_new = v_i - v_prime`

代码：

```python
v_new = v_i - v_prime
```

形状：

```text
[B,H,C,V]
```

`v_new` 是当前 chunk 最终真正要写入 state 的 residual。它同时考虑了：

- chunk 内前面 token 的影响。
- 进入 chunk 前历史 state 的影响。
- `beta` 写入门。
- `g` 衰减门。

### 3.4 `(q_i * exp(g)) @ state`

代码：

```python
attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
```

形状：

```text
[B,H,C,K] @ [B,H,K,V] -> [B,H,C,V]
```

这是输出中来自旧 state 的部分。

为什么 `q_i` 要乘 `exp(g)`？因为从 chunk 开始到当前位置，旧 state 也经历了累计衰减。第 `p` 个位置读到的不是原封不动的旧 state，而是已经衰减后的旧 state。

### 3.5 `attn @ v_new`

代码：

```python
attn @ v_new
```

形状：

```text
[B,H,C,C] @ [B,H,C,V] -> [B,H,C,V]
```

这是输出中来自当前 chunk 内新写入内容的部分。

### 3.6 合成当前 chunk 输出

代码：

```python
core_attn_out[:, :, i] = attn_inter + attn @ v_new
```

形状：

```text
[B,H,C,V]
```

输出由两部分组成：

```text
attn_inter:
  从进入 chunk 前的历史 state 读出的内容

attn @ v_new:
  从当前 chunk 内新写入内容读出的内容
```

合起来就是这个 chunk 所有 token 的 Gated Delta Rule 输出。

### 3.7 更新 state

代码：

```python
last_recurrent_state = (
    last_recurrent_state * g[:, :, i, -1, None, None].exp()
    + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
)
```

形状：

```text
旧 state 衰减:
[B,H,K,V]

chunk 内新写入:
[B,H,K,C] @ [B,H,C,V] -> [B,H,K,V]

更新后 state:
[B,H,K,V]
```

含义：

```text
新的 state =
    旧 state 经过整个 chunk 的衰减
    + chunk 内每个 token 的写入，按它到 chunk 末尾的距离做衰减后累加
```

其中：

```text
g[:, :, i, -1]
```

表示这个 chunk 末尾位置的累计 log 衰减。

而：

```text
g_last - g_position
```

表示某个位置写入后，继续传播到 chunk 末尾还要经历的剩余衰减。

## 4. 两张表对应的整体流程

可以把 prefill 的 chunk 算法浓缩成：

```text
1. g.cumsum(-1)
   得到 chunk 内累计 log 衰减。

2. 构造 decay_mask
   表示 j 位置写入传到 i 位置的保留率。

3. k_beta @ key^T
   计算 token 之间在 key 空间的相互影响。

4. 通过 lower-triangular 展开修正 attn
   把直接依赖和间接依赖都合并进去。

5. attn @ v_beta
   得到 chunk 内在零初始 state 下的 residual 写入。

6. k_cumdecay @ state
   计算历史 state 已经能解释的部分。

7. v_new = value - v_prime
   扣掉历史 state 影响，得到真正要写入的 residual。

8. 当前 chunk 输出 =
      从历史 state 读出的内容
    + 从当前 chunk 新写入读出的内容。

9. 更新 state
   传给下一个 chunk。
```

## 5. 最关键的理解

`prefill` 并没有改变 Gated Delta Rule 的数学含义。它只是把逐 token 的递推：

```text
S_t = exp(g_t) * S_{t-1} + k_t delta_t^T
```

改写成：

```text
chunk 内:
  用 lower-triangular 矩阵并行解出所有 token 的 residual 写入和输出

chunk 间:
  只传递一个固定大小的 recurrent state
```

因此，图里的 `attn` 不是 softmax attention，而是 delta-rule recurrence 的三角解算矩阵。它的作用是让原本 token-by-token 的状态更新，在 prefill 阶段尽可能变成 GPU 友好的批量矩阵乘法。

