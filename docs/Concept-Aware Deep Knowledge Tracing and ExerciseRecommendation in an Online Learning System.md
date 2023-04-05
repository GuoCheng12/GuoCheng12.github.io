## *Concept-Aware Deep Knowledge Tracing and Exercise*Recommendation in an Online Learning System



## Abstract

提出一种新的<u>**TK**</u>——**Dynamic Key-Value Memory Network (DKVMN)**

（其实也是一种**exercise-concept** 的 **mapping**)



用这个TK建立MDP ， 接着他们用**RL**在上面进行**policy eval and policy improvement**

> 观看本篇文章，接下来的目的就只是单纯看他的TK设计了



## 1.Introduction

作者设计了的system<u>相比</u>于MOOCs， 重点强调了以下两个特点

> 其实没差

![](https://pic.imgdb.cn/item/642cfaf9a682492fccb86ffd.jpg)

![](https://pic.imgdb.cn/item/642cfac4a682492fccb7e7b4.jpg)

> **知识点的概念特征**  暂时不知道是啥

![](https://pic.imgdb.cn/item/642cfae6a682492fccb838d4.jpg)

![](https://pic.imgdb.cn/item/642cfb12a682492fccb8bf6a.jpg)

## 2. Related work

> DTK和BTK都基本提到了（包括后续有人提出的改进）  不过没提到ACT 



## 3. Background

### 3.1 Intelligent Practice System(IPS)

> 这个就是一个线上OJ系统 分为1 - 7个stages 每个stages 都包括一个knowledge。
>
> 每个knowledge 都包涵三个 **tags** 	这就解释了上面的**知识点概念特征(exercises's knowledge concept properties)**
>
> 我也同时找到了DKVMN的文献
>
> ![](https://pic.imgdb.cn/item/642cfb23a682492fccb8f651.jpg)
>
> 这篇论文提出的模型结合了BKT和DKT的优点：开发概念之间关系的能力和跟踪每个概念状态的能力
>
> 细节在下面讲...

### 3.2 Data Set and Data Pre-Processing

![](https://pic.imgdb.cn/item/642d04b8a682492fccc73a9a.jpg)



## 4. Knowledge Tracing Model

### 4.1 Concept-Aware Memory Structure

> 作者修改 DKVMN 以根据课程的**概念列表**设计其内存结构
>
> 主要是这张图：
>
> ![](https://pic.imgdb.cn/item/642d0506a682492fccc79c09.jpg)
>
> 通过输入问题qt 的到最后知识概念权重
>
> （具体是怎么得到的？回头看论文吧）



### 4.3 Read Process

>  然后，我们使用获得的 KCW 来计算用户当前知识概念状态的加权和，以预测学生在练习中的表现 rt
>
> 我们进一步将 rt 与练习的难度和阶段特征的嵌入连接起来，即 dt 和 gt。结果再经过一个带有激活函数Tanh的全连接层得到一个汇总向量ft，它包含了学生与qt相关的知识状态的所有信息和练习的特征
>
> ![](https://pic.imgdb.cn/item/642d0597a682492fccc86052.jpg)
>
> 最后，ft通过一个全连接层输出学生**正确完成练习qt的概率**。用 p 表示概率
>
> ![](https://pic.imgdb.cn/item/642d05bfa682492fccc8923e.jpg)

### 4.4 Update Process

> 更新这一过程
>
> 
>
> ![](https://pic.imgdb.cn/item/642d05f8a682492fccc8e5a4.jpg)
>
> 作者提出，相比于DKVMN 他们还将**学生做题时间**考虑进去了。会在学生做完题目后，将**时间和正确率**都作为更新参数。
>
> 其他的更新和DKVMN类似

## 5. Exercise Recommendation Based on Reinforcement Learning

> 这里使用了POMDP（部分观测）
>
> state: 学生的潜在知识点掌握情况
>
> action：题目
>
> agent:  推荐策略
>
> env: 学生
>
> 其他我看不懂了55 暂时还没看过POMDP的决策过程和决策算法













 

