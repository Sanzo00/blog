---
title: LLM推理服务论文
katex: true
sticky: 0
toc: true
typora-copy-images-to: ../../img/Blog/llm-inference
date: 2024-09-19 16:38:06
updated: 2025-06-15 16:38:06
tags: LLM, Inference
categories: Blog
---


记录一些LLM推理优化相关的论文

<!-- more -->



## Parrot [OSDI '24]

Parrot: Efficient Serving of LLM-based Applications with Semantic Variable. [pdf](https://www.usenix.org/system/files/osdi24-lin-chaofan.pdf) [code](https://github.com/microsoft/ParrotServe) [author](https://chaofanlin.com/)

Parrot这篇论文的主要贡献是提出了**一个全新的推理workload：LLM Applications**。



LLM Application是使用LLM来完成特定的任务（摘要，搜索，代码助手等），在一个应用中通常包含多个LLM请求。

<img src="../../img/Blog/llm-inference/image-20240919165055080.png" alt="LLM Application的工作流程" style="zoom:50%;" />







以往推理优化系统是**request-centric**，即对用户的应用是透明的，“一视同仁”的处理用户的请求，缺少**application-level**的信息。

在LLM Application中，请求具有以下特点：

1. 多个连续的LLM请求可能存在**依赖关系**。
2. 即使在单个应用中，LLM请求可能具有不同的**调度偏好**。
3. LLM的请求之前存在大量的**相似性**。



<img src="../../img/Blog/llm-inference/image-20240919170523984.png" alt="多智能体应用中LLM请求的通信流程" style="zoom:50%;" />





由于缺少application-level的信息，现有的推理优化主要有两个问题：

1. 网络通信开销。
2. 任务调度等待开销。



![现有推理服务 vs. Parrot推理服务](/img/Blog/llm-inference/image-20240919165122842.png)







<img src="../../img/Blog/llm-inference/image-20240919165139641.png" alt="Parrot的系统架构图" style="zoom:50%;" />



Parrot设计了一个Semantic Variables的编程抽象，用来将用户的执行逻辑暴露给推理服务端。

基于这个Semantic Variables可以获取到应用内的LLM请求的调用依赖关系，进而做一些系统上的优化，包括DAG-based analysis，Performance Objective Deduction，Application-Centric Scheduling等。



<img src="../../img/Blog/llm-inference/image-20240919171157542.png" alt="使用Parrot写的代码例子" style="zoom:50%;" />







## AquaPipe [SIGMOD '**25**]







## Sparse Attention



### Longformer [Arxiv '20]



### Streaming LLM [ICLR '24]

> 解决的问题

LLM在长上下文的效率和效益的问题：

1. 长上下文的计算和内存开销大。
2. LLM在上下文长度超过预训练长度时，生成质量差。



> 核心idea



![图1:对比4中不同的注意力计算方法](/img/Blog/llm-inference/image-20250615205757413.png)



1. **Dense Attention**：计算复杂度高，当上下文长度超过预训练长度，模型表现差。
2. **Window Attention**：计算复杂度低，当上下文长度超过缓存长度（initial token）被驱逐时，模型表现差。
3. **Sliding Window w/ Re-computation**：计算复杂度中等，通过重新计算，保留了initial token的影响，模型表现良好。
4. **StreamingLLM**：在**Window Attention**的基础上，引入了对initial token的注意力计算，兼顾推理速度和模型生成质量。





Streamllm的方法主要来自于一个观察：

**”Attention sink“**：作者发现LLM对初始token的注意力关注较高。

![图2显示了 Llama-2-7B 在处理多个短句时的平均注意力模式。从可视化结果来看，前两层主要呈现局部注意力，偏向关注临近的 token；而从第2层起，模型普遍对句首 token 表现出异常高的关注度。这种现象说明了高层注意力容易集中在起始位置，可能引发注意力偏置问题（attention sink）。](/img/Blog/llm-inference/image-20250615205218195.png)





作者对**Attention sink**给了一个解释：

LLM的注意力计算，保证所有token的注意力之和为1，即使当前token只需要根据自己就可以推测出下一个token，由于softmax的设计，还是需要将一些注意力分散到其他token上去。

由于LLM的自回归特性，开始的token可以被后面所有tokne注意力到，因此LLM对initial token的关注更高，进而在训练的过程中，赋予inital token特殊的含义。



