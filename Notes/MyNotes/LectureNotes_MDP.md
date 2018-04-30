### Markov Decision Process

##### what is reinforcement learning?

- Learning based on rewards
- Robotics - teach them to play something
- Alpha go :)

##### What have you head of reinforcement learning?

we are gonna consider agents and agents are gonna learn (for example) how to play a game without even knowing what are the rule are.

easiest instance : Markov Decision Process

##### What are the 2 main concepts that we have learned so far?

- [x] Discriminating Learning : 
  - $p(y|x)​$ 
  - How do we train ? 
    - Maximum likelihood 
- [x] Generative 
  - $p(x)$
  - What were the main techniques that we talked about ? 
    - K means, GMM, GANs, VAE and Auto Regressive Type Methods (RNNs)
- [ ] Now - Reinforement learning
  - Examples - [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)  (They used same algorithm to train 49 different Atari games), [Fly stunt manoeuvers in a helicopter](https://www.youtube.com/watch?v=M-QUkgk3HyE), Defeating the world champion at Go, Manage investment portfolio, Control the data centers, make a humanoid robot walk...

##### How do we formulate the task?

###### what is different?

there is no training data! There some interactive environment now. Now we get some form of score. Also , temporal dependency (Action may effect the future) !

###### How its done : 

At each step 't'  - 

* The agent thinks/ knows about being in state : $s_t$ 
* Agent will perform the action : $a_t$
* Then the agent will receive a scalar reward : $r_t \in \mathbb{R}$
* Now the agent find itself in new state : $s_{t+1}$

> Relate this with Atari game: action is move left or move right, the reward is the change in score  and the state is the current position of the agent. 



###### Settings that we need to keep in mind :

1. Deterministic setting - the bot will follow your guidance exactly and will not do anything stupid. 
2. Stochastic setting - The bot will have a stochastic component in its movement. Hence, there is a probability that it will not follow the commands. 

###### Formulating the above things:

The Markov Decision Process have :

* A set of states : $s \in S$
* A set of actions : $a \in A_s$
* A transition probability : $P(s' | s, a)$
* A reward function : $R(s,a,s')$
* `sometimes` A start and terminal states 

> The transition probability will now take care of both of the settings. In deterministic setting, the transition probability will be either zero or 1. In stochastic setting, its a probability distribution over all the possible states to go to. 

This is `Markov` because the transition probability of the current state is independent of future and past.  

$P(S_{t+1} = s' | S_t = s_t, A_t = a_t, S_{t-1} = s_{t-1}, A_{t-1} =a{t-1} ...) = P(S_{t+1} = s' | S_t = s_t, A_t = a_t)$

> **Note** : This is an approximation. we make this approximation to model in a simple way. 

 TODO : pictorial description + Graph



##### What makes the RL different from other paradigms?

* No dataset, No supervisor
* Might be delay in feedback

Now given a full description of a MDP, what do we want now?

We need now a set of rules (policy $\pi^*$) that determines the actions our agent will perform to maximize the expected future reward. 

How can we encode this into a policy?

As we have already defined, a policy is a mapping from state to an action. 

So, $\pi(s) : S \mapsto A_s$ 

Now the question is how to find this optimal policy?

*  We can try exhaustive search : In this case, we have a policy which determines what action to perform for a given state. 
  * Then how many such policies are there : we have $\Pi_{s \in S} |A_s|$  number of possible policies. Short answer :  Very big! Very expensive to do. But nonetheless, guarantees to find the optimal policy.
  * How to evaluate the quality of policy $\pi$? For this we need to compute the expected future reward. Denoted by $V^{\pi}(s_0)$, which means what is the value that you expect to observe when being at state $s_0$ if agent performs according to policy $\pi$.  From this notation, $V^{\pi^*}(s_0)$, is the value with optimal policy. 
* Policy iteration [Howard 1960]
  * Initialize a policy $\pi$ 
  * Repeat until the policy $\pi$ does not change
    * policy evaluation (iteratively): TODO
    * Policy Iteration : TODO 
*  Value iteration (Instead of finding optimal policy, can we look for optimal value) [Shapley (1953), Bellman (1957)]
  * ​



