import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# hyperparamter / training config
EPSILON_STRATEGY = "GREEDY_DECAY" # GREEDY / GREEDY_DECAY_SOFTMAX
EPSILON_START_VALUE = 0.7
EPSILON_END_VALUE = 0
MAX_GAME_STEPS = 400
TAU = 5 # IAUU softmax policy


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
        self.epsilon = get_epsilon()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def get_epsilon():

    epsilon = []

    if (EPSILON_STRATEGY == "GREEDY_DECAY" ) or (EPSILON_STRATEGY == "GREEDY_DECAY_SOFTMAX"):
        epsilon = np.linspace(EPSILON_START_VALUE, EPSILON_END_VALUE, MAX_GAME_STEPS)
    elif EPSILON_STRATEGY == "GREEDY":
        epsilon = np.linspace(EPSILON_START_VALUE, EPSILON_START_VALUE, MAX_GAME_STEPS)

    return epsilon


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    if self.train: #and random.random() < random_prob:
      
        # computation of Q-values
        # assumptions for value approximation of Q:
        # 1. weights \beta_a are stored in a matrix column-wise -> multiplication of state_features with that matrix
        #    leads to an array
        # 2. weights \beta_a are stored in the same order as defined in ACTIONS 
        state_features = state_to_features(game_state)
        Q_sa = np.argmax(np.matmul(state_features.T, np.asarray(self.action_models_weight)))
        argmax_Q = ACTIONS[np.argmax(Q_sa)]

        if EPSILON_STRATEGY == "GREEDY_DECAY_SOFTMAX":
            # IAUU exploration - improved epsilon-greedy strategy: uses softmax instead of uniform dist
            
            # calculation of probabilities for actions
            numerator = np.exp(Q_sa/TAU)
            denominator = np.sum(numerator)
            probabilities = numerator/denominator
            return np.random.choice(argmax_Q, np.random.choice(ACTIONS,probabilities),p=[1-self.epsilon[game_state.step],self.epsilon[game_state.step]])

        else:
            # uses uniform probability to pick an action with probability epsilon
            return np.random.choice(argmax_Q, np.random.choice(ACTIONS),p=[1-self.epsilon[game_state.step],self.epsilon[game_state.step]])


        # self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        # return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    return np.random.choice(ACTIONS, p=self.model)
    # return self.model.propose_action(game_state)


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
