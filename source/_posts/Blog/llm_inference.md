---
title: LLM推理服务论文
katex: true
sticky: 0
toc: true
typora-copy-images-to: ../../img/Blog/llm-inference
date: 2024-09-19 16:38:06
updated: 2025-06-28 16:38:06
tags: 
	- LLM
	- Inference
categories: Blog
---


记录一些LLM推理优化相关的论文

<!-- more -->





## FlashAttention



## PageAttention



## RadixAttention



## RingAttention





## RAGCache [Arxiv '24]



## Cache-Craft [SIGMOD '25]



## CacheBlend [EuroSys '25]



## Superposition [ICLR '24]





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

AquaPipe: A Quality-Aware Pipeline for Knowledge Retrieval and Large Language Models [[paper]](https://dl.acm.org/doi/10.1145/3709661) 











## Similarity-based





### PQCache [SIGMOD '25]

PQCache: Product Quantization-based KVCache for Long Context LLM Inference [[paper]](https://arxiv.org/pdf/2407.12820v2) [[code]](https://github.com/HugoZHL/PQCache)



PQCache解决的LLM在long context下的推理效率问题。

长上下文推理对显存需求带来了挑战，如何在有限的显存空间实现高效的推理，同时保存高的输出质量，是一个被广泛关注的问题。

一个符合直觉的方法：selective attention，通过选择部分token进行注意力计算，显著降低了对内存和计算的需求。

现有selective attention方法可以分为：KV Cache dropping (Scissorhands, StreamingLLM [ICLR '24], H2O [NIPS '23])，KV Cache offloading (InfLLM [arxiv '24] SparQAttention) 两类。



如下图所示，PQCache发现selective attention的执行和传统信息检索的Product Quantization的过程很像。

在LLM的注意力计算过程中，向量Q和所有的K进行相似度计算，然后经过softmax并与V加权求和得到输出X。

Q和K的相似度计算和信息检索中根据用户问题检索相似的top-k个向量的过程基本一致。

![Information retrieval vs. LLM inference with seletive attention](/img/Blog/llm-inference/image-20250628201826548.png)





本文选择了开销比较低的Product Quantization来管理KV Cache。

PQ的索引构建和搜索过程如下图所示。

> PQ Construction：

1. 将每个KV Cache向量划分为m个子向量。
2. 对所有KV Cache的每个子向量进行聚类，生成$2^b$个质心。
3. 原来KV Cache向量对应的m个子向量编码为距离最近的质心id。

> PQ Searching:

1. 查询Q向量同样被划分为M个子向量。
2. 每个子向量和对应的空间中$2^b$个向量计算相似度。
3. 根据相似度计算原始向量与Q的相似性得分，选取TopK得分的向量。

![PQ construction and serarching](/img/Blog/llm-inference/image-20250628202224666.png)



PQCache的整体执行流程如下图所示：

1. 在Prefilling阶段，正常计算得到每个输入token的KV Cache，并异步的卸载到CPU。
2. CPU收到KV Cache之后，构建PQ用于后续检索。
3. 在Decoding阶段，加载Centroids和PQ Codes，并计算TopK K向量。
4. 根据计算的TopK向量，加载对应KV向量，并在GPU执行注意力计算。

在实现中，PQCache的KV Cache包含三种：initial tokens，middle tokens，and local tokens.

StreamingLLM中发现attention sink的现象，即initial tokens受到更多的注意力关注，对模型的回答质量有很大的影响。

local tokens表示最近计算的token。middle tokens表示历史KV Cache保存在CPU中。

PQCache将initial和local tokens保存在GPU，并维护一个窗口，超过窗口的local token被卸载到CPU。

![Overview of PQCache](/img/Blog/llm-inference/image-20250628202419202.png)



![PQCache v.s. sequential scheduling.](/img/Blog/llm-inference/image-20250628212757321.png)









## Apt-Serve [SIGMOD '25]

Apt-Serve: Adaptive Request Scheduling on Hybrid Cache for Scalable LLM Inference Serving [[code]](https://github.com/eddiegaoo/Apt-Serve) [[paper]](https://arxiv.org/pdf/2504.07494)







## Training



[[paper]](https://arxiv.org/pdf/2407.12117v3)





Malleus: Straggler-Resilient Hybrid Parallel Training of Large-scale Models via Malleable Data and Model Parallelization 





## Sparse Attention/Long context



### LM-infinite





### Longformer [Arxiv '20]



### StreamingLLM [ICLR '24]

Efficient Streaming Language Models with Attention Sinks [[code]](https://github.com/mit-han-lab/streaming-llm)  [[paper]](https://arxiv.org/abs/2309.17453)



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



> 细节



1. 按照在cache中的位置，重新分配token的位置信息，以保成相对位置的正确性。

   如下图所示，当生成 token 9的时候，每个token的位置为[0, 1, 2, 3, 4, 5, 6, 7]而不是[0, 1, 2, 3, 6, 7, 8, 9]。

   ![图3：StreamingLLM的的KV Cache](/img/Blog/llm-inference/image-20250615213712847.png)

2. key tensor的缓存和使用

   - 对于RoPE，在应用 rotray 变化前缓存 key tensor，在加载的时候对其rotray。

   - 对于ALiBi，在注意力分数上添加一个linear bias。