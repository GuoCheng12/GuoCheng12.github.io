# Learning Path Recommendation Based on Knowledge Tracing Model andReinforcement Learning



## Abstract

介绍他们使用了某个KT算法和RL算法来**record the knowledge level change during the learning process.**



## Introduction

大概意思就是说，作者认为：大多数现在的学习路径推荐，都是由唯一的维度**（learning costs）**来评判，而且这个唯一的维度也是由**expert**来手动标注。

而作者的算法*Knowledge Tracing based Knowledge Demand Model (KTKDM)* 有下面几个亮点：

![](https://pic.imgdb.cn/item/642d7555a682492fcc855a0d.jpg)



> 没有什么特别关注到的点



## Related Work

> 分别介绍了BKT和DKT



## Proposed Method

### *Algorithm Overview*

先说一下建模部分：

作者提出了他们的model —— KT-KDM  

**KT-KDM**也分为两个不同的模块 一个是**Knowledge Demand Model (KDM)** 另一个是***Knowledge Tracing Model (KTM)***

**KTM** 通过**learner**的n次的**trajectories**来学习到**KCs** 然后把**KCs**送给**KDM**

然后就构建出**MDP**

![](https://pic.imgdb.cn/item/642d7ac3a682492fcc903711.jpg)

具体流程：

![](https://pic.imgdb.cn/item/642d7c19a682492fcc926152.jpg)

> env中 KTM和Learner之间的预处理感觉很厉害



### Knowledge Tracing Model

KTM在使用前也要被训练，之后才能上实战

1. 先将KTM试做为**Learner**与KDM交互 ，因为开始KTM有过去**Learners**很多的trajectories所以可以视作为一个很大的**learner's  knowledge database**，通过KDM 训练出一个新的KTM
2. 在线平台就是实战了，在线部署阶段，真实用户与模型进行交互。用户获取并学习系统中推荐的KC，进行学习。实际学习结果合并到学习历史中，传递给KTM进行预测

![](https://pic.imgdb.cn/item/642da6f0a682492fccde24b3.jpg)

关于KTM是啥 如何实现的，作者的解释是他们的KTM其实是正则化DKT+



> 关于什么是正则化DKT+
>
> - 引入Recurrent Neural Networks(RNNs)和LSTM细胞来模拟学生的知识状态，提出了Deep Knowledge Tracing (DKT)模型，而DKT+模型则进一步考虑了时间上的依赖性。
> - 引入正则化项到原始DKT模型的损失函数中，包括重构误差和波状度量，以增强一致预测。
> - 提出了性能指标来评估知识跟踪的三个方面的性能。



### Learning Path Generation

#### 1. Model

- 该方法具有两个主要组成部分：知识需求模型（KDM）和权重随机选择函数。

- KDM是一个预测网络，它接收学习者掌握程度的输入，并输出一个表示推荐KC的预测概率向量。
- 权重随机选择函数进行推荐选择。

#### 2. Reward

- 奖励函数reward设计为相邻推荐KC的学习者后学习掌握程度之差，对于已经推荐了的KC的减分。

#### 3. Training method

- 利用优势演员-评论家（A2C）RL方法解决连续状态空间、离散动作空间和实验长度有限的问题。



## Experiments

### 

![](https://pic.imgdb.cn/item/642db4efa682492fccef6235.jpg)

- 在ASSITment 2009-2010数据集上的实验与贪心和随机算法进行对比，表明了所提出的KT-KDM方法比随机算法更好，与贪心算法持平，同时运算成本更低。
- 推荐的KC覆盖大多数的学习过程，并且推荐次数的分布相对均匀。
- 该模型在推荐效果和运行时间上都实用可行。



## Conclusion

- a. 该工作的意义：
  - 提出了一种基于知识跟踪模型( KTM)和强化学习模型(RLM)相结合的个性化学习路径推荐方法，该方法动态跟踪学习过程，有效提高了学习效率，并生成了覆盖关键主题的学习路径。
- b. 创新性、性能和工作量：
  - 该方法越过了获取学习成本的步骤，提供了更加个性化的学习路径推荐，可以在未有学习成本的情况下推荐适当的学习路径，最终建立了学习路径推荐框架。
- c. 研究结论（列出点）：
  - 对于推荐算法，本文提出了基于知识需求模型和强化学习的学习路径推荐算法，并在实验中验证了其有效性，建立了一个新的生成个性化学习路径的方向，同时也指出了未来在线学习的训练应该同时训练这两个模型。