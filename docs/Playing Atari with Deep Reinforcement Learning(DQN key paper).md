# Playing Atari with Deep Reinforcement Learning(DQN key paper)

note by GuoChengWu  

## Introduction

![](https://pic2.imgdb.cn/item/6455b8530d2dde577752aeb1.jpg)

作者指出他们使用了一个**卷积神经网络**来应对监督学习在RL上面的弱势，该网络使用 **Q-learning** 算法的变体进行训练，使用**随机梯度下降(SGD)**来更新权重。

Input是一个high dimensional visual input (210 * 160 RGB video at 60Hz)，且这个network不知道任何有关于游戏和模拟器内部的内容

## Background

由于考虑到只观测到当前**游戏状态(Xt)**并不足以分析这一状态的价值，所以考虑分析一整个**sequences（x1,a1,x2...at-1,xt)** 且定义为 **s**，这种形式主义产生了一个庞大但有限的**马尔可夫决策过程 (MDP)**，其中每个序列都是一个不同的状态。因此，我们可以将标准强化学习方法应用于 MDP，只需使用完整序列 st 作为时间 t 的状态表示。

我们的任务是要让agent与环境交互的过程中得到最优解

![](https://pic2.imgdb.cn/item/6455b8700d2dde577752cf04.jpg)

**同样满足贝尔曼公式（就是一个Q-learning）**

**loss function：**

![](https://pic2.imgdb.cn/item/6455b88d0d2dde577752eccf.jpg)

**这个算法是model-free的 也是off-policy的**

## Related work

最著名的的一个任务是TD-gammon  用强化学习来玩西洋双陆棋

![](https://pic2.imgdb.cn/item/6455b8c40d2dde5777532b9a.jpg)

作者指出：off-policy，model-free的function approximator表现的都很差（发散） 

最接近此任务的一个算法——NFQ

![](https://pic2.imgdb.cn/item/6455b8d40d2dde5777533a63.jpg)

**但这个算法用了batch gradient descent 而 DQN用的是 stochastic gradient descent**



## Deep Reinforcement Learning

> 这章介绍了DQN的实现技术

由于近年来DL的发展迅速，作者从CV和SC的模型中学习到，认为SGD会是很好的一个突破口

![](https://pic2.imgdb.cn/item/6455b8e20d2dde57775345a9.jpg)

DQN的核心技术：

- experience replay （we store the agent’s experiences at each time-step
- replay memory





![](https://pic2.imgdb.cn/item/6455b9020d2dde5777536808.jpg)

使用这两个办法一个是很好的增加了在学习过程中的稳定性和准确率，二是这样这么做，一定就是off-policy的了，也改变了当前参数会决定下一个时刻的data sample的参数

![](https://pic2.imgdb.cn/item/6455b9100d2dde5777537668.jpg)

实机演示中：

![](https://pic2.imgdb.cn/item/6455b91e0d2dde57775382e2.jpg)