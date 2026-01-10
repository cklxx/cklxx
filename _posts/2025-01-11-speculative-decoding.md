先猜，再验，把一次前向的成本分摊到多个 token
最近在看大模型推理加速时，刷到了投机解码（Speculative Decoding，也常被叫作 Speculative Sampling）。很有趣，按捺不住不如写一篇文章。
这个方法很像搜索里的结果预测：先用一个高性能的预测器（比如Cache）把可能出现的内容先写出来，再由裁判确认哪些预测确实成立，成立的就直接用掉，省下生成的开销。
在大模型里，这件事发生在 token 粒度，验证会做的更精细。

---
投机解码在做什么，提升什么指标
一句话：
用更小更快的草稿模型（draft）先连续吐出一段候选 token，然后用目标大模型（target）在一次前向里统一判断候选 token ，能接受多少就一次推进多少，从而减少目标模型的前向传播的次数。 
提升的指标分场景：
- 在线单请求（batch 很小，比如 bs=1 或 2）更关心“输出阶段的 token 速度”，也就是常说的 inter token latency 变小，或者 tokens/s 变大。vLLM 的文档就直接把它定位成改善 memory bound 场景下的 inter token latency。 (vLLM)
- 离线压测或高并发服务，更关心“总吞吐”，也就是单位时间系统一共吐出多少 output tokens/s。TensorRT‑LLM 的吞吐表格就是按 output tokens/s 来报的，tok/s 统计包含首 token 时间。 (NVIDIA Developer)
一些场景例子：
1. DeepMind 的 speculative sampling（草稿模型 + 改造的拒绝采样）在 Chinchilla 70B 的分布式设置上，报告了 2 到 2.5 倍的解码加速，同时保持样本质量不变。 (arXiv)
2. NVIDIA TensorRT‑LLM 的内部测试（DGX H200，TensorRT‑LLM 0.15.0.dev 等），在 Llama 3.1 405B 作为目标模型、4 张 H200 上：
  - 不用草稿模型：33.46 tokens/s
  - 用 Llama 3.2 3B 草稿：120.75 tokens/s，对应 3.61 倍
同一篇文章还给了 70B 目标模型在 1 张 H200 上的表格，比如 70B 基线 51.14 tokens/s，用 1B 草稿到 146.05 tokens/s（2.86 倍）。 (NVIDIA Developer)
3. EAGLE‑3 这类“专门训练过的草稿器”在生产框架里的数据也很直观。论文里在单张 H100、SGLang v0.4.4、目标模型 LLaMA‑Instruct 3.1 8B、MT‑Bench 数据集、batch size=1 的设置下：
  - 不用投机：158.34 tokens/s
  - 用 EAGLE‑3：373.25 tokens/s
同一组实验在 batch size=64 下报告吞吐提升 1.38 倍。
总结下，投机解码减少了目标大模型在 decode 阶段的每 token 一次前向频率，让一次前向的产出跨多个 token 摊销。特别是在解码阶段常见的 memory bound 情况下，这种摊销能更充分吃掉硬件的并行算力。 

---
主流程
先按最常见的工程实现来理解，也就是草稿 + 验证：
1. 草稿模型 D：从当前上下文开始，自回归地连续生成 K 个候选 token，得到序列 y1…yK。
2. 目标模型 T：把 原上下文 + y1…yK 整体喂进去跑一次前向，一次性得到接下来 K 个位置的打分，还能额外得到下一个位置的打分（所以最多能推进到 K+1 个 token）。TensorRT‑LLM 的文档也明确写了 target 一次可能返回到 K+1 个 token。 (NVIDIA GitHub)
3. 从左到右检查：能接受多少就接受多少，一旦某个位置不接受，后面的草稿全部作废。
4. 至少会前进 1 个 token：不接受的位置，用目标模型自己的采样规则产出那个 token，保证流程不会卡住。
从工程视角看，这像是把每次只走一步的串行过程，改成先走几步试试，再一次验收。当验收通过的 token 足够多，目标模型的次数明显减少，tokens/s 或总吞吐就上去了。

---
举个工程下的例子对比正常流程和投机流程
batch size=1、贪心模式、lookahead K=3 的最小流程
普通解码：
1. 用目标模型 T 对当前上下文做一次前向，拿到最后一行 logits
2. argmax 选出下一个 token
3. 把 token 追加到上下文，更新 KV cache
4. 重复
投机解码：
1. 草稿模型 D 连续生成 3 个 token：y1,y2,y3，同时把它们追加到草稿侧上下文
2. 目标模型 T 对 当前上下文 + y1,y2,y3 做一次前向，得到 4 行分布（对应 t+1 到 t+4）
3. 从左到右验：
  - 如果第 1 行 argmax 等于 y1，就接受 y1，并保留这一步产生的 KV
  - 如果第 2 行 argmax 等于 y2，就接受 y2，并保留 KV
  - 如果第 3 行 argmax 不等于 y3，停止
4. 在停止的位置，用目标模型自己的分布选出真正 token，并把它作为最后追加的 token
5. 丢弃草稿里未被接受的部分，并把两边的 KV cache 回滚到一致长度
6. 重复

---
两个关键问题
问题一：小模型预测的 token 的方法为什么能用？命中率能到多少？速度能快多少？
- 结果对，许多位置预测下一个 token 其实很容易，小模型和大模型的预测结果会高度一致，比如固定搭配、模板化语言、代码里的常见语法片段。Google Research 在回顾文章里用有些 token 更容易，有些更难的角度解释过这个现象，并把它作为投机解码的核心观察之一。 (Google Research)
- 成本低，草稿模型足够小，单步生成便宜。只要它比大模型快很多，并且命中率不低，整体就赚。TensorRT‑LLM 的说明里把这件事写成了两条假设：并行验证多个草稿 token 的耗时接近验证一个 token，并且在整个生成过程中会有多个 token 能被成功验证。 (NVIDIA GitHub)
命中率 和 加速 则可以用一个非常工程化的估算连起来看：
- 设草稿提前生成的token数量为 $$K=4$$
- 设每个草稿 token 被目标模型接受的概率近似为 $$p$$，并且从左到右需要连续命中才能多接受一个
- 那么一次目标模型验证期望能接受的 token 数大约是$$p + p² + p³ + p⁴$$
- 同时目标模型还能额外产出 1 个自己的 token，于是一次目标模型前向期望推进的 token 数约是 $$1 + p + p² + p³ + p⁴$$
举个数：$$p=0.7$$、$$K=4$$ 时，上式约等于 2.77，也就是目标模型调用次数理论上能缩到原来的约 1/2.77。再算上草稿模型的开销，实际加速会低一些，通常就落在 2 倍上下。 
很容易想到，草稿模型越接近目标模型，命中率更高，但草稿本身也更慢；草稿模型太小会很快，但命中率可能不够。NVIDIA 的吞吐表格里也能看到这种折中，1B 草稿和 3B 草稿都能很好地加速，但 8B 草稿在某些组合下反而没 3B 草稿快。 
值得注意的是：vLLM 的文档中提到过投机解码并不是对所有数据集都能稳定降低 inter token latency，优化还在进行。(vLLM)

---
问题二：怎么验证？为什么说一次前向里统一判断候选 token？
我们都知道，大模型的生成的本质是预测下一个token
关键在于，Transformer架构的输出，本来就不是一个 token，而是一个长度等于词表大小的打分向量。所有的输入token对词表的打分向量都会在一次向前中全部计算出来，所以我们可以直接看某一段token对词表的打分，来判断这一段token里哪些token符合要求可以被采用。
普通解码过程
直接看公式最直观，不想看可以直接看 大模型推理加速之投机解码
假设：
- batch size = 1
- 当前上下文长度是 $$L$$
- 词表大小是 $$V$$
我们只需要看大模型的解码过程的最后一步，这一步计算位于 Transformer 解码 forward 的末端，在所有自注意力层和前馈网络完成之后，用输出投影将 hidden states 映射到词表空间，并显式构造每个位置上的完整 next-token 概率分布，是将 hidden states 显式转换为全词表概率分布的最后一步。
先从 hidden states 开始，设

$$H \in \mathbb{R}^{L \times d}$$，$$W \in \mathbb{R}^{d \times V}$$

他们线性投影得到 logits 矩阵

$$Z = H W \in \mathbb{R}^{L \times V}$$

把它按行展开，形式就是

$$Z =\begin{bmatrix}
h_1^\top W \\
h_2^\top W \\
\vdots \\
h_L^\top W
\end{bmatrix}=
\begin{bmatrix}
z_{1,1} & z_{1,2} & \cdots & z_{1,V} \\
z_{2,1} & z_{2,2} & \cdots & z_{2,V} \\
\vdots  & \vdots  & \ddots & \vdots  \\
z_{L,1} & z_{L,2} & \cdots & z_{L,V}
\end{bmatrix}$$

这里第 $$i$$行

$$z_i = h_i^\top W \in \mathbb{R}^{V}$$

就是在位置 $$i$$，对整个词表的打分向量。
Softmax 是按行独立做的，可以写成

$$P = \mathrm{softmax}(Z) \in \mathbb{R}^{L \times V}$$

更具体地，对任意位置 $$i$$ 和词表索引 $$j$$：

$$P_{i,j} = \frac{\exp(Z_{i,j})}{\sum_{k=1}^{V} \exp(Z_{i,k})}$$

展开成矩阵形式就是

$$P = \begin{bmatrix}
\mathrm{softmax}(z_1) \\
\mathrm{softmax}(z_2) \\
\vdots \\
\mathrm{softmax}(z_L)
\end{bmatrix}=
\begin{bmatrix}
p_{1,1} & p_{1,2} & \cdots & p_{1,V} \\
p_{2,1} & p_{2,2} & \cdots & p_{2,V} \\
\vdots  & \vdots  & \ddots & \vdots  \\
p_{L,1} & p_{L,2} & \cdots & p_{L,V}
\end{bmatrix}$$

并且对每一行都有

$$\sum_{j=1}^{V} P_{i,j} = 1 \quad \forall i \in \{1,\dots,L\}$$

自回归解码时，真正被用来采样的是最后一行

$$P_{L,:} = \mathrm{softmax}(h_L^\top W)$$

根据采样规则我们就可以选出我们要的下一个 token 是什么，但从计算上看，前面所有 $$P_{1,:}$$ 到 $$P_{L-1,:}$$在这次 forward 里已经完整存在，所以如果我们一次输入 $$L + n$$ 个 token， $$P_{L,:}$$ 到 $$P_{L+n-1,:}$$ 也都会被计算出来，拿到这些概率，我们就可以很容易的验证后面的 $$n-1$$ 个token是否符合采样规则。
一个好理解的例子
我们来看看直接使用目标模型预测下一个 token 的过程：
假设词表只有 6 个 token：

$$V = ["A","B","C","D","E",""]$$

你输入 token  $$A$$ ，目标模型前向传播得到 logits 矩阵 $$Z$$，为了直观，我直接写成概率矩阵 $$P_1$$（1×6）

$$[0.05, 0.60, 0.10, 0.20, 0.04, 0.05]$$

很直观，矩阵位置的每个数字对应 token 的解码概率，这里最大的是 $$B$$（0.60），贪心解码下一个token就选 $$B$$。
普通解码，每次只算一行，选一个 token 作为新 token，下一次再算下一行，要预测下一个token则再输入 

$$AB$$

概率矩阵 $$P_2$$ 就变成 2×6：

$$\begin{bmatrix}
0.05, 0.60, 0.10, 0.20, 0.04, 0.05\\
0.05, 0.05, 0.20,0.65, 0.04, 0.05
\end{bmatrix}$$

这里最大的是 $$D$$（0.65），就选 $$D$$，最后生成的序列连起来就是
 $$ABD$$
如果我们先用草稿模型很快啊，自回归生成了两个 token $$BC$$；
再回到目标模型计算，输入 $$ABC$$，目标模型前向传播得概率矩阵 $$P_3$$（3×6

$$\begin{bmatrix}
0.05, 0.60, 0.10, 0.20, 0.04, 0.05\\
0.05, 0.05, 0.20,0.65, 0.04, 0.05\\
0.05, 0.05, 0.63, 0.18, 0.04, 0.05
\end{bmatrix}$$

你看第2、3个token的概率也都一起计算出来了，按照贪心解码，第2个 token 是 $$B$$（0.60），第3个token是  $$D$$ ，（0.65），那么预测结果$$B$$就可以被采用，而 $$C$$ 需要被抛弃，并且我们也能确认第3个 token 是 $$D$$；
细心的你一定发现，过程中发生了两次向前，一次草稿模型一次目标模型，最终生成的结果也是
 $$ABD$$

---
选择标准很关键：正常解码怎么选，投机解码怎么选
常见的采样规则
本质是一个从分布 $$P_t$$选 token 的函数：
- 贪心（greedy，temperature=0）：取 $$argmax(P_t)$$
- Top k：先取概率最高的 $$k$$ 个 token，再在里面采样
- Top p（nucleus）：取概率累加到 $$p$$ 的最小集合，再在里面采样
- Temperature：对 logits 做缩放，再 softmax，再按上面规则选
规则不同，输出自然会不同。
在贪心模式下怎么验
这是最容易工程验证的版本
- 草稿模型只需要给 token 序列 $$y1…yK$$
- 目标模型一次前向得到每个位置的分布 $$P_verify[i, :]$$
- 对每个 $$i$$，从左到右做检查：如果 $$argmax(P_verify[i, :])$$ 等于 $$yi$$，就接受 $$yi$$ 否则停止接受，并用目标模型在该位置按贪心选出真正 token
这个版本几乎能做到输出和纯目标模型贪心一致，vLLM 在文档里也把 Greedy Sampling Equality 作为重要的 lossless 验证测试之一。 
在随机采样模式下怎么验
随机采样下，如果你只用是否在 top k、是否高于阈值来验，输出分布会发生变化，生成质量和分布一致性都会变得不可控。
可以用拒绝采样来保证分布一致。EAGLE‑3 的预备知识部分把这个 acceptance 概率写得很清楚：对草稿给出的 token $$t̂$$，接受概率是 $$min(1, p(t̂)/q(t̂))$$，$$p$$是目标模型概率，$$q$$是草稿模型概率；如果不接受，就从一个修正后的分布里重新采样并丢弃后续草稿。
这里顺手回答你之前提到的困惑：草稿模型表面上确实只给了 token，但要严格做到随机采样下的分布一致，系统通常还需要草稿模型对这些 token 的概率 $$q(t̂)$$。工程里很多实现会直接拿到草稿的 logits 或 logprobs，或者能在草稿生成时把这些值缓存下来。贪心版本就不一定需要 $$q$$，只要 token 本身就够用。
不同采样规则会不会让生成质量变差？
- 严格版（拒绝采样）：不会，目标是“输出分布和纯目标模型一致”，质量维持在同一分布意义下，不靠经验拍脑袋。vLLM 把这类保证称为 theoretical losslessness，并且在实现上也做了 rejection sampler 的一致性测试。 (vLLM)
- 近似版（阈值验收、top k 里就算通过等）：会，忍受一定分布偏移，速度更快，但分布已经改了，质量可能变好也可能变差，更多是经验和业务容忍度问题。

---
计算消耗怎么估算
我们以目标模型前向次数作为基准。
- 普通解码生成 $$N$$ 个 token，需要大约 $$N$$ 次目标模型 decode 前向
- 投机解码每次目标模型前向能推进的 token 数，期望接近 $$1 + p + p² + … + p^K$$ 所以目标模型前向次数大约变成 $$N / (1 + p + p² + … + p^K)$$
再加上草稿模型的代价：
- 每轮你还要做 $$K$$次草稿前向
- 总成本约等于目标前向次数 × (一次目标前向成本 + K × 一次草稿前向成本)
举个 🌰 例子：
- $$K=4$$
- 每个 token 的连续命中概率 $$p=0.7$$
- 草稿模型单步成本约等于目标模型的 1/10
那么一次目标前向平均推进约 2.77 个 token，目标前向次数约缩到 1/2.77。每轮还额外花 0.4 个目标前向的成本在草稿上，总体每个 token 的成本大约变成原来的 0.505，速度约 1.98 倍。
实际的GPU吞吐的计算会更复杂些，有兴趣可以自己尝试思考一下。

---
工程实现
看到这里，想必同学们已经学会投机采样，老板：xx你来，给我们业务上一下这个优化。
代码参考：https://github.com/ai-clarify/mini-llm/blob/main/infer/mlx/bench.py
bench对比 tbd

---

历史和未来

2018 年 Stern、Shazeer、Uszkoreit 提出了 blockwise parallel decoding，核心已经很接近今天大家说的 draft and verify：并行预测多个未来位置，再用一个打分模型验证，最后回退到能确认的最长前缀，在 greedy 条件下可以做到不损失质量的迭代次数减少，甚至在允许轻微质量变化时换来更高倍数的速度收益。 

到了 2022 年，Leviathan、Kalman、Matias 把这个思路系统化，提出 speculative decoding，并把“分布保持不变”放到了算法核心，给出了严谨的采样与纠偏步骤，让它适用于常见的采样策略。 

2023 年 DeepMind 的 speculative sampling 几乎同期独立完成，论文里也明确提到与前者并行独立，重点放在大模型分布式服务的工程现实里，并给出一套可在硬件数值误差范围内保持目标分布的拒绝采样方案。

2024 年又出现了不依赖外部草稿模型的一支路线，代表是 Medusa，它在同一个大模型骨干上加多头解码头来并行预测多个后续 token，再用树状的并行验证机制减少解码步数，论文报告了在多种提示类型上约 2.3 到 2.8 倍的加速且不牺牲生成质量。 

2025 年 EAGLE-3 把另一条关键线索推得更远，聚焦怎样把草稿端训练得更像一个合格的加速器，通过直接 token 预测、多层特征融合等设计提升接受率，从而放大整体速度收益。 这些思路也很快进入了工程生态，Hugging Face 把它整理成 assisted decoding 的接口形态，明确描述了“助手提出候选，主模型一次前向验证”的生成方式。 vLLM 、SGLang 和 TensorRT-LLM 这类推理引擎也把投机解码当成一等特性来做系统级整合，并给出面向吞吐的实测数据与注意事项。

---
进一步的方向：把草稿命中率做上去，EAGLE‑3 做了什么，还有哪些路线

加速取决于草稿快和命中率高，一定需要专门优化草稿器。

EAGLE‑3 的核心思路是把草稿模型做成更擅长预测多步 token 的形态。论文里提到它抛弃了早期版本的特征预测约束，改成直接做 token 预测，并通过 training time test 去模拟多步生成过程，同时融合目标模型多层特征。它报告在多个模型和任务上，温度为 0 的 speedup ratio 最高到 6.5 倍，并且在 SGLang 里 batch size=64 时吞吐还能提升 38%。

除了 EAGLE 系列，还有几条常见路线：
- Medusa：不引入额外草稿模型，直接在同一个模型上加多个 decoding head 并用树状验证。论文里报告 Medusa‑1 能做到超过 2.2 倍加速，Medusa‑2 到 2.3 到 3.6 倍。 (arXiv)
- N gram、Suffix Decoding 这类基于模式匹配的“非神经草稿器”，在重复性很强的任务上很香，比如代码编辑、循环式 agent。vLLM 文档里把这些当成 speculator 的不同实现方式。 (vLLM)
- 级联草稿、多层草稿、早退草稿、同模型自投机等，都是在“更便宜地产生更靠谱的候选”这个方向上做文章。Google 的回顾文章也总结过行业里不少沿着这个范式扩展的工作。 (Google Research)
- DFlash：把投机解码和 Flash-Attention、speculative execution 级别的内核融合放在一起做，核心思路是减少“被拒绝 token”的无效算力，把验证阶段的 attention 和前向尽量压进一次 GPU 执行里。论文和代码里展示的是在高接受率场景下，把 speculative decoding 的收益从算法层面进一步兑现到 kernel 层面，对 inter token latency 更友好。当前主要价值在工程路径上，而不是提出新的解码范式，常被视为对 vLLM、TensorRT-LLM 这类系统的补强方向。（DFlash，arXiv / GitHub）

整体看下来，这几条路线本质都围绕同一个判断展开：投机解码的上限不在多猜几个 token，而在便宜地产生候选，并且让验证几乎不浪费算力。

Medusa 在模型结构上做文章，N-gram 和 suffix 直接绕过神经网络，级联和早退在系统调度上榨成本空间，DFlash 则把瓶颈进一步下沉到 GPU 执行层。不同方案在不同数据分布下表现差异很大，这也是 vLLM 和 Google Research 会强调没有单一方案能稳定统治所有 workload的原因。

---
参考
1. Yaniv Leviathan, Matan Kalman, Yossi Matias. Fast Inference from Transformers via Speculative Decoding. arXiv:2211.17192. (arXiv)
2. Charlie Chen et al. Accelerating Large Language Model Decoding with Speculative Sampling. arXiv:2302.01318. (arXiv)
3. Google Research Blog. Looking back at speculative decoding. (Google Research)
4. NVIDIA Technical Blog. TensorRT‑LLM Speculative Decoding Boosts Inference Throughput by up to 3.6x. (NVIDIA Developer)
5. TensorRT‑LLM 文档. Speculative Sampling. (NVIDIA GitHub)
6. vLLM 文档. Speculative Decoding 与 lossless guarantees。 (vLLM)
7. Y. Li et al. EAGLE‑3: Scaling up Inference Acceleration of Large Language Models via Training time Test.
8. Tianle Cai et al. Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. arXiv:2401.10774. (arXiv)
