# DeepSeek V4模型简介

## 整体架构

<p style="text-align: center;">
  <img src="deepseek_v4_architecture.jpg" alt="DeepSeek V4架构图" />
</p>

## CSA模块计算图

CSA（Compressed Sparse Attention）由压缩器、Indexer和稀疏MQA Attention组成：
- Compressor(C4A)：维护压缩KV缓存和kv_state/score_state。
- Indexer：根据当前Q对压缩KV打分并选出Top-k。
- Attention：拼接window KV、compress Top-k和sink token后执行MQA，并经过O投影回到hidden维度。

可编辑PPT版本：[CSA计算流图.pptx](CSA计算流图.pptx)

<p style="text-align: center;">
  <img src="CSA计算流图.svg" alt="CSA计算流图" />
</p>

DeepSeek V4是新一代MoE大语言模型系列，核心包含DeepSeek-V4-Pro与DeepSeek-V4-Flash两款模型。
两者均支持最长100万（1M）tokens上下文，并围绕长上下文效率、推理稳定性和工程可部署性进行了系统优化。

效果：V4系列在知识问答、代码生成、复杂推理和Agent任务等公开评测中表现突出；其中Pro版本在高难任务上更稳健，Flash版本在较低激活成本下保持了较强综合能力。

性能：官方模型卡显示，V4在长上下文场景采用混合注意力与稀疏专家计算路径，重点降低推理FLOPs与KV缓存占用，更适合高并发在线服务与超长文档处理。

架构特点：
- 采用MoE架构并区分总参数与激活参数，按需路由提升计算利用率
- 引入Hybrid Attention（CSA + HCA），面向100万上下文优化推理效率
- 使用mHC（Manifold-Constrained Hyper-Connections）增强层间信号传递稳定性
- 训练阶段使用Muon优化器，并在超过32T高质量tokens上进行预训练
- 后训练采用两阶段流程（领域专家强化 + 蒸馏整合）提升通用与专项能力

## Pro与Flash模型特点

### DeepSeek-V4-Pro

- 参数规模：总参数约1.6T，推理时激活约49B参数
- 适用场景：更偏向高难度推理、复杂Agent流程、代码与多步骤任务
- 能力表现（模型卡公开结果）：MMLU-Pro 87.5、GPQA Diamond 90.1、SWE-bench Verified 80.6、Terminal-Bench 2.0 67.9、HLE 37.7
- 定位总结：在V4系列中提供更高上限与更强稳定性，适合对准确率和任务完成度要求更高的场景

### DeepSeek-V4-Flash

- 参数规模：总参数约284B，推理时激活约13B参数
- 适用场景：更偏向高吞吐、低时延与成本敏感型部署
- 能力表现（模型卡公开结果）：MMLU-Pro 86.4、GPQA Diamond 88.1、SWE-bench Verified 79.0、Terminal-Bench 2.0 56.9、HLE 34.8
- 定位总结：以更小激活开销换取更优性价比，在多数通用任务中保持强竞争力

## 相关资料：
- [DeepSeek V4论文（技术报告）](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
- [Pro模型卡片（Hugging Face）](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [Flash模型卡片（Hugging Face）](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [整体介绍](https://mp.weixin.qq.com/s/8bxXqS2R8Fx5-1TLDBiEDg)
- [Pro模型定义](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py)
- [Flash模型定义](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py)
