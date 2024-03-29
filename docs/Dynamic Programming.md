# Dynamic Programming

**Richard Bellman**

<u>2023/2/27 注释</u>

## Background

- 人类在发展过程中，不断提高自己思考问题，解决问题的能力。

> Man is a decision-making animal and, above all, conscious of this fact. This self-scrutiny has led to continuing efforts to find efficient ways of behaving in the face of complexity and uncertainty.

## Feedback Control

作者让我们考虑这样一个情况：飞船降落月球的场景

<img src="https://pic.imgdb.cn/item/64344bb00d2dde577785381f.jpg" style="zoom:50%;" />

目标就是最短时间最少功耗...到达目的地



那么这个问题有时候很容易被我们理想化了，对此我们不得不随时对这样的trajectory进行监控，由于各种不确定性因素的影响，有时候我们的路径可能变成这样：

<img src="/Users/wuguocheng/Library/Application Support/typora-user-images/image-20230411015156508.png" alt="image-20230411015156508" style="zoom:50%;" />

愚蠢的办法就是将这个路径不停的根据“刚刚制定的路径”不断的修改方案，使得我们的路径回归到原始状态。但是这个显然太愚蠢了。动态规划，最base的地方就是千万不能忘记最主要的目的。

<img src="https://pic.imgdb.cn/item/643454360d2dde57778be3b5.jpg" style="zoom:50%;" />

## Policy Concept

**多阶段决策被视为政策的重复应用。在最小化时间、燃料、成本或最大化利润的意义上最有效的策略称为最优策略。**

基于刚刚的问题，我们应该想到的是这样的策略：

<img src="https://pic.imgdb.cn/item/643454210d2dde57778bd742.jpg" style="zoom:50%;" />

这是DP中最核心的一点。

思考并使用DP解决问题的本质就是要时刻关注当前状态，无论之前有多少束缚，我们在此刻都是要最优化现处的策略。

动态规划是一种数学技术，用于复杂系统中的决策制定，其应用于预测和控制问题

## Application 

这里作者主要想讲述当时年代（此论文写于1966年），迫于计算机的内存和计算能力有限，动态规划问题一直都不太能够实现大部分的Application，但随着技术进步，动态规划变得更加广泛适用。



## Multistage Decision Making 

> 这段话很震撼

作者在这段想表示，当我们如果面临的问题是一个**复杂决策中涉及不确定性时**的问题，我们应该也继续想到**DP**给我们带来的算法好处。首先，对于这样的问题，我们可以**简化假设**，例如：我们可以准确确定系统在任何时候的状态，并且知道何时施加控制以及其效果。然而，这些假设都是理想化的，当它们在科学上限制了我们理解和预测的能力时，我们应该准备修改它们。

在这里，作者仅考虑了因果预测需要修改的情况。如果我们无法准确预测决策的效果，那么有效决策的问题就变得更加困难。此时，“有效”或“最优”的定义也变得模糊。概率理论是一种研究不可预测效果的数学理论，但这个理论并不能涵盖所有类型的随机事件，只能处理特定类型的不确定性。

> 在这里，"有效"（effective）和"最优"（optimal）是指在面对不确定性的情况下，能够帮助我们实现目标的决策策略。这些策略需要考虑到可能的随机事件，以便在多个阶段的决策过程中使我们尽可能接近目标。有效和最优策略可以最大化收益、最小化损失或实现其他特定的目标。
>
> 在前面的掷硬币赌博游戏示例中，"有效"和"最优"策略是指在每次抛硬币时，指导你下注金额的规则，使你能够在10轮游戏中实现将100元增加到200元的目标。这个策略需要在每轮游戏中都能给出最佳的下注建议，以便在整个过程中使你的收益最大化。
>
> 值得注意的是，在面对不确定性时，"有效"和"最优"并不意味着我们可以完全预测决策的结果，而是我们在可用信息和当前情况下，采取了最合适的行动。这些策略旨在使我们在整个决策过程中尽可能地实现目标，而不是仅仅关注单个阶段的结果。

作者提到，在不确定条件下研究决策过程，可以将问题领域转向赌博系统。赌博系统涉及的数学原理与保险公司的精算、华尔街的投资计划以及可靠性理论、库存理论等方面具有相同的抽象性质。

赌博系统需要一种政策，这个政策告诉赌徒在每种可能的情况下应该做出什么决策。动态规划已经在21点等赌博游戏中取得了成功。在涉及随机事件的多阶段决策过程中，灵活的策略非常适用。最优性原理为处理确定性和概率决策过程提供了统一的数学工具。随机这个术语经常用来描述涉及随机事件的过程，因为它没有日常生活的含义。

<u>举个例子:</u>

假设你在一场掷硬币游戏中，每次抛硬币，正面朝上你赢，反面朝上你输。你开始时有100元，你的目标是通过合理的下注策略，在10轮游戏中赢得200元。在这个场景中，你面临的问题就是如何在每次抛硬币时决定下注多少钱，以实现你的目标。

在这个游戏中，有许多不确定因素，比如每次抛硬币的结果是随机的。由于这个不确定性，我们无法确切地预测每次决策的结果。因此，我们需要运用概率理论来帮助我们进行决策。

为了解决这个问题，你可以制定一个策略，根据你当前拥有的金额以及你的目标，告诉你每次应该下注多少。这个策略需要在每轮游戏中都能指导你做出决策，同时考虑到游戏的随机性。动态规划方法可以帮助你找到这样一个策略，因为它可以在每个阶段都优化你的决策，使你在整个过程中达到最优结果。

总之，这段文字主要讨论了在面对不确定性时如何进行复杂决策。通过运用概率理论和动态规划，我们可以在随机事件中找到一种有效的策略来指导我们的决策。

> 恍然大悟，动态规划解决MDP的影子



## Adaptive Control

传统的决策过程我们是在已知情况下做出假设：

<img src="https://pic.imgdb.cn/item/6434ceba0d2dde5777f9577b.jpg" style="zoom:50%;" />

但是作者提出，在很多重要工作中（实际上在所有决策的场景中），我们需要在不完全了解底层系统的基本工作原理的情况下做决策。

这个时候就提出一种新的概念——**“Adaptive Control”**自适应控制

自适应控制是一种**基于经验修改最优策略**的方法。在这种方法中，我们从对最优策略的某些预设概念开始，然后根据经验系统地修改这些策略。自适应控制与心理学家所说的**适应**有密切联系。

举个例子：

> 在无人驾驶汽车的开发过程中，我们无法提前知道所有可能遇到的道路、交通和天气条件。同时，无人驾驶汽车需要在不断变化的环境中做出实时决策。因此，我们需要设计一个能够根据实时数据调整其行为的自适应控制系统。
>
> 在这个例子中，自适应控制系统首先基于预先设定的一组规则（例如遵守交通法规、保持安全距离等），在实际驾驶过程中，系统会收集到各种数据，如其他车辆的位置、速度，道路状况等。基于这些信息，系统会不断调整自己的行为，以适应当前的环境。
>
> 例如，当无人驾驶汽车遇到拥堵时，自适应控制系统可能会根据实时交通数据寻找最佳路线。如果遇到突如其来的恶劣天气，系统可能会降低车速以保证行驶安全。在所有这些情况下，系统都在不断地学习并根据实时数据调整自己的行为。
>
> 这个过程与心理学中的适应概念相似。无人驾驶汽车需要在不断变化的环境中通过实时学习和调整，以确保其决策始终能够满足安全、有效和舒适等目标。



## Hierarchy of Decision Making

作者在这里讨论到了“machine can learning” ？

关于机器能否思考这一问题引发了很多争议，作者在这个问题所理解的意义取决于我们如何定义“机器”、“思考”以及“能”。在这里，“机器”是指现有的商用数字计算机。而为了定义“思考”，我们将其与决策制定联系起来，将思考过程的不同层次与决策过程的不同层次等同起来。根据这种定义，我们可以探讨我们是否能编写一个在指定时间内执行特定决策过程的计算机程序。“能”具有不同的含义，取决于我们允许的时间是2分钟、2小时还是2年，或者仅要求时间是有限的，尽管不可预测。

目前，计算机可以被编程来玩国际象棋或跳棋，但它还不能从**经验中学习**成为国际象棋大师。如果能实现这种能力，将是自适应过程理论的重大突破。

按照这种思路，可以将决策过程分为不同层次。第一层是上文提到的确定性或随机类型的过程。第二层涉及了解系统结构的过程。在每个阶段都需要一个局部策略来做决策，而全局策略则根据经验修改局部策略。这就是关于决策的决策。选择全局策略是关于关于决策的决策的决策。我们可以继续按照这种方式进行。

> 首先，在第一层，我们考虑确定性或随机类型的过程。这些过程通常涉及基于已知信息做出决策。例如，在给定的环境下找到最佳路径。
>
> 接下来，在第二层，我们考虑关于系统结构的学习过程。在这一层，我们需要在每个阶段使用局部策略（基于当前状态和信息做出决策），并根据经验修改这些局部策略。这可以称为关于决策的决策，因为我们在考虑如何在不断变化的情况下调整决策方法。这一层的关键是从**经验中学习**以改进决策方法。
>
> 
>
> (关于什么是系统结构的学习过程，简单来说就是我们试图了解和学习潜在的系统规律，os:说白了就是MDP中的env提供的observation或者reward给我们带来的信息。
>
> 系统结构的学习过程是指在决策过程中，我们试图了解和学习潜在的系统规律、关系和模式。这些规律和模式有助于我们更好地理解系统的行为，从而在将来做出更明智的决策。简而言之，学习系统结构意味着我们试图揭示系统的内在逻辑，以便在决策过程中加以利用。
>
> 例如，假设你正在管理一个复杂的供应链系统，你需要决定如何优化库存水平以满足客户需求。在这个过程中，你可能会发现系统中的某些规律，例如某些产品的需求周期性波动、供应商的交货时间可靠性等。通过学习这些规律，你可以更好地预测未来的需求变化，并相应地调整库存水平以提高整体效率。
>
> 系统结构的学习过程通常涉及从经验和数据中提取信息，以便调整和优化决策策略。这可能包括观察系统的行为、分析历史数据、进行试验等。随着时间的推移，通过这些学习过程，我们可以不断更新和改进我们的决策方法，以更好地适应系统的变化。在这个过程中，局部策略（基于当前状态和信息做出的决策）可以根据所学到的系统结构进行调整和优化。)
>
> 
>
> 再往上，选择全局策略（调整局部策略的方法）可以被认为是关于关于决策的决策的决策。这意味着我们不仅在调整具体决策，还在调整如何调整决策方法本身。这个层次更加复杂，因为它涉及到对整个决策过程的优化和调整。
>
> 总的来说，这个层次结构强调了决策过程在不同层次上的复杂性。在每个层次上，我们需要考虑不同的问题和策略，以便更好地处理各种情况。这种分层方法有助于我们更好地理解决策过程的不同方面，并有可能开发出更加智能和自适应的决策系统。



无论我们采用哪种方式引入**层次结构**，都会遇到一定的问题。例如，在特定的决策过程中如何确定其层次？

>1. 在引入层次结构时，我们需要确保每个层次都具有明确的目标和职责。这将有助于确保在整个决策过程中，各个层次之间的沟通和协作得以顺利进行。
>2. 在设计这些层次时，我们应该考虑在不同层次之间分享信息和知识的方法。有效的信息共享可以提高决策过程的整体效率，并有助于避免重复努力和资源浪费。
>3. 在实际应用中，我们可能需要根据特定问题和环境调整层次结构。在某些情况下，可能需要增加更多层次以处理更复杂的问题，而在其他情况下，可能需要减少层次以简化过程。
>4. 虽然层次结构可以帮助我们更好地理解和解决复杂问题，但它也可能带来一定程度的管理挑战。例如，可能需要更多的协调和沟通来确保各个层次之间的顺利合作。因此，在引入层次结构时，我们需要权衡其优缺点。
>5. 在整个决策过程中，我们需要不断地评估和改进层次结构。这可能包括定期审查各个层次的表现，以及根据需要调整职责和目标。通过这种持续改进的方法，我们可以确保层次结构始终保持适应性，并能够有效地应对新的挑战和变化。



## Conclusion

>  作者这段话引发的思考在如今也是一个具有挑战的问题

这段话提到了人类对大脑过程（包括心理学和生理学方面）的研究兴趣在很大程度上受到了数字计算机的推动。尽管目前尚无完整的理论来解释这些引人入胜的问题，但数字计算机的出现促使人们对大脑过程产生了浓厚兴趣。

作者指出，在处理人类个体和社会时，我们可以在**不完美**的情况下应对，如非理性、不合逻辑、不一致和不完整的行为。然而，在操作计算机时，我们必须满足详细的指令和绝对精确性的严格要求。如果我们理解了人类在面对复杂性、不确定性和非理性时做出有效决策的能力，那么我们可以比现在更有效地利用计算机。

> 神经网络？可解释性？

正是对这一事实的认识激发了神经生理学领域研究的繁荣。越来越多地研究大脑的信息处理方面，我们变得越来越困惑和惊叹。在充分理解和重现这些过程之前，我们还有很长的路要走。

总之，数学家在许多新兴领域面临着数以千计的严峻挑战、难题和困惑，尽管他们可能永远无法解决其中的一些问题，但他们永远不会感到无聊。对数学家来说，这已经足够了。