## Reward的一些选择...

详细看此视频[布朗大学Michael Littman: The Reward Hypothesis](https://d3c33hcgiwev3.cloudfront.net/pJY0r-OvEemJfgqR2HI2sA.processed/full/720p/index.webm?Expires=1681344000&Signature=bMTGyPDgCVVTNjDMzGdOTIcWzytYmyh~JPBXs4r0mYPCoaQGXXStY4Uupr5QG46ZsYZYwyJbIlBIPadjlnaXU6hGd70QqSF5xuOvF7WirDERQYMzV4i47dfJID0mNdAXX6q-DsXa4qvp5S837smng3s8arW8s5uWkQDvW-wOHcg_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

我们知道Agent总是学习如何最大化收益。如果我们想要它为我们做某件事情，我们提供收益的方式必须要使得Agent在最大化收益的同时也同时实现我们的目标。

> You can see what is happening in all of these examples. The agent always learns to maximize its reward. If we want it to do something for us, we must provide rewards to it in such a way that in maximizing them the agent will also achieve our goals. It is thus critical that the rewards we set up truly indicate what we want accomplished. In particular, the reward signal is not the place to impart to the agent prior knowledge about how to achieve what we want it to do.5 For example, a chess-playing agent should be rewarded only for actually winning, not for achieving subgoals such as taking its opponent’s pieces or gaining control of the center of the board. If achieving these sorts of subgoals were rewarded, then the agent might find a way to achieve them without achieving the real goal. For example, it might find a way to take the opponent’s pieces even at the cost of losing the game. The reward signal is your way of communicating to the robot what you want it to achieve, not how you want it achieved.

