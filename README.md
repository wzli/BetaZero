# BetaZero
An open source board game AI inspired by AlphaZero, with the goal of reaching superhuman performance via soley rule definitions and self-play.

This project is still in progress ...

##The Plan:
Borrowing from recent algorithms in Deep Rienforcement Learning (such as Deep Q-Network, Policy Gradient, Asynchronous Actor Critic), CNN (such as Inception Network) and Bayesian Neural Networks (such as the MC Dropout approximation), design an elegantly simple archatecture for the purpose of playing 2-play turned based board games.

1) There will be a single deep neural network to approximate the value function of an input game state.
..* An additional policy/action model is omitted by design, given the assumption that the valid action space and state transitions for any given state is already known from provided game rules
..* Deciding on which action to take based on itterating through value predictions of each possible state transition is easily compuatable given the action space for boardgames are relatively small (>400 moves even for a large game like GO)
..* Allow for more tractable mechanism of action/policy selection and hueistically directed value network updates.

2) Combine Exploration and Explotation into a single process during Policy/Action selection
..* borrow algorithms from the multi-arm bandit problem, the goal of minimizing total regret translates to learning efficiency. Eg (UCB, EXP3, Thompson Sampling)

3) Use a Bayesian Based Neural Network archatecture to provide uncertainty estimates along with predictions
..* Pillar of stocastic based exploration algorithms is uncertainty estimates, to favor exploration of action of higher uncertainty
..* Allow for auto pruning of network archatecture and weights based on probability of hidden node being zero, also allows for reducing computational requirement for embedded applications.  
..* Base on recent papers on either MC dropout or variational inference methods

4) Directly approximate the Minimax tree of the game
..* minimax tree is the proven optimal algorithm for 2 player turn based games given enough resources, its a useful hueistic to embeded
..* on action selection, directly predict the value of the tree node corresponding to a given state, instead of generating a large MTCS or a Minimax Tree and truncating on computation bounds using the value predictions at the leaves
..* create learning batches based on individual game roll outs
..*propagate value updates to each state transition in a game batch based on minimax tree update mechanism (a better update huestic than Q-learning which only updates a single state before reward, and policy gradients that update everystate leading to reward)

5) Use for CNN for board translational symetries
..* For game board rotational and reflectional symetries, map all symetrical states to one unique state prior to input
..* To inject custom value hueistinces to the network as aid, add fixed analysis filters at input of CNN layers, for example, a filter that transform's board states to a heatmap of pieces in danger of being captured.
..* Maybe use Inception nets to auto decide filter sizes 

6) The usual Deep learning advances, leaky Relu's, ResNets
..* use conventioanal libraries, Keras, Tensorflow
