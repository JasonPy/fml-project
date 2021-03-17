import os
import pickle
import time

import numpy as np
from scipy.spatial.distance import cityblock

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
TRANSFORMER = None

# hyperparameters / training config
EPSILON_STRATEGY = "GREEDY_DECAY_EXPONENTIAL"  # GREEDY_DECAY_LINEAR / GREEDY
EPSILON_START_VALUE = 1.0
EPSILON_END_VALUE = 0.01

SOFTMAX = False  # define whether the argmax or the softmax is used during training
TAU = 5  # for softmax policy

MAX_GAME_STEPS = 401
NUMBER_EPISODES = 100


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

    self.number_of_features = 1475

    # load transformer if feature selection has already been done
    if os.path.isfile("transformer.pt"):
        with open("transformer.pt", "rb") as file:
            global TRANSFORMER
            TRANSFORMER = pickle.load(file)

    if self.train:
        if os.path.isfile("my-saved-model.pt"):
            # train weights from existing file
            with open("my-saved-model.pt", "rb") as file:
                self.model = pickle.load(file)
        else:
            # train weights entirely new based on specific weight setup
            set_weights(self)

        set_epsilon(self)

    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def set_epsilon(self):
    if EPSILON_STRATEGY == "GREEDY":
        self.epsilon = epsilon_decay(0)
    elif EPSILON_STRATEGY == "GREEDY_DECAY_LINEAR":
        self.epsilon = epsilon_decay((EPSILON_START_VALUE - EPSILON_END_VALUE) / NUMBER_EPISODES)
    elif EPSILON_STRATEGY == "GREEDY_DECAY_EXPONENTIAL":
        self.epsilon = stretched_exponential_decay


def epsilon_decay(decay_rate):
    def decrease_epsilon(t):
        return EPSILON_START_VALUE - t * decay_rate

    return decrease_epsilon


def stretched_exponential_decay(t):
    A = 0.6
    B = 0.2
    C = 0.1

    standardized_time = (t - A * NUMBER_EPISODES) / (B * NUMBER_EPISODES)
    cosh = np.cosh(np.exp(-standardized_time))
    epsilon = 1. - (1 / cosh + (t * C / NUMBER_EPISODES))
    if epsilon < 0:
        return 0
    elif epsilon > 1:
        return 1
    return epsilon


def set_weights(self):
    self.model = np.zeros((self.number_of_features, 6))
    self.model[:, 0:4] = np.random.uniform(0.7, 1, (self.number_of_features, 4))
    self.model[:, 4:6] = np.random.uniform(0, 0.2, (self.number_of_features, 2))


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    # computation of Q-values
    state_features = state_to_features(game_state)
    weights = self.model
    Q_sa = np.argmax(np.matmul(state_features.T, weights))
    argmax_Q = ACTIONS[Q_sa]

    if self.train:
        if SOFTMAX:
            # IAUU exploration - improved epsilon-greedy strategy: uses softmax instead of uniform dist
            # calculation of probabilities for actions
            numerator = np.exp(Q_sa / TAU)
            denominator = np.sum(numerator)
            probabilities = numerator / denominator
            return np.random.choice([argmax_Q, np.random.choice(ACTIONS, p=probabilities)],
                                    p=[1 - self.epsilon(game_state["round"]),
                                       self.epsilon(game_state["round"])])
        else:
            # uses uniform probability to pick an action with probability epsilon
            return np.random.choice([argmax_Q, np.random.choice(ACTIONS)],
                                    p=[1 - self.epsilon(game_state["round"]),
                                       self.epsilon(game_state["round"])])

    self.logger.debug("Querying model for action.")
    return argmax_Q


def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return X: A np.array of features
    """
    # start = time.time()

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # field is kept transposed due to using x,y coordinates for other channels
    field_channel = game_state['field'].flatten()

    opponent_channel = np.zeros(game_state['field'].shape)
    for i in game_state['others']:
        opponent_channel[i[3]] = 1
    opponent_channel = opponent_channel.flatten()

    bomb_channel = np.zeros(game_state['field'].shape)
    for i in game_state['bombs']:
        bomb_channel[i[0]] = i[1]
    bomb_channel = bomb_channel.flatten()

    explosion_map_channel = game_state['explosion_map'].flatten()

    coin_channel = np.zeros(game_state['field'].shape)
    for i in game_state['coins']:
        coin_channel[i] = 1
    coin_channel = coin_channel.flatten()

    state_as_features = np.stack(
        [field_channel, bomb_channel, explosion_map_channel, coin_channel, opponent_channel]).reshape(-1)

    # feature vector (initially a list)
    features = []  # np.empty(26)

    features.append(game_state['round'])
    features.append(game_state['step'])

    # append score and if laying bomb is possible
    features.append(game_state['self'][1])
    features.append(int(game_state['self'][2]))

    field = game_state['field'].T
    bombs = np.asarray(game_state['bombs'])
    pos = np.asarray(game_state['self'][3])
    coins = np.asarray(game_state['coins'])
    others = np.asarray(game_state['others'])
    explosion_map = game_state['explosion_map']

    # max distance equals length of diagonal of field
    max_dist = cityblock(np.array([1, 1]), game_state['field'].shape)

    # append own position
    features.append(pos[0])
    features.append(pos[1])

    # distance to others
    others_dists = [max_dist + 1] * 3
    if others.shape[0] > 0:
        for i in range(len(others)):
            dist = cityblock(np.asarray(others[i, 3]), pos)
            others_dists[i] = dist
        others_dists = np.sort(others_dists).tolist()
    features += others_dists

    # distance to bombs
    bomb_dists = [max_dist + 1] * 4
    if bombs.size > 0:
        for i in range(len(bombs)):
            dist = cityblock(np.asarray(bombs[i, 0]), pos)
            bomb_dists[i] = dist
        bomb_dists = np.sort(bomb_dists).tolist()
    features += bomb_dists

    # danger zone to determine if agent would get hit by bombs
    danger_zone = []
    for b in bombs:
        if np.abs(pos[0] - b[0][0]) <= 3 or np.abs(pos[1] - b[0][1]) <= 3:
            if pos[0] == b[0][0]:
                y_pos = pos[1] - b[0][1]
                if y_pos > 0:
                    if np.sum(np.where(field[pos[0], b[0][1]:pos[1]] == -1)) == 0:
                        danger_zone.append(1 / (b[1] + 1))
                    else:
                        danger_zone.append(0)
                else:
                    if np.sum(np.where(field[pos[0], pos[1]:b[0][1]] == -1)) == 0:
                        danger_zone.append(1 / (b[1] + 1))
                    else:
                        danger_zone.append(0)
            elif pos[1] == b[0][1]:
                x_pos = pos[0] - b[0][0]
                if x_pos > 0:
                    if np.sum(np.where(field[b[0][0]:pos[0], pos[1]] == -1)) == 0:
                        danger_zone.append(1 / (b[1] + 1))
                    else:
                        danger_zone.append(0)
                else:
                    if np.sum(np.where(field[pos[0]:b[0][0], pos[1]] == -1)) == 0:
                        danger_zone.append(1 / (b[1] + 1))
                    else:
                        danger_zone.append(0)
        else:
            danger_zone.append(0)
    d = [0, 0, 0, 0]
    d[:len(danger_zone)] = np.sort(danger_zone)
    features += d

    # distance to nearest crate
    crate_indices = np.argwhere(field == 1)
    crate_dists = []
    if crate_indices.shape[0] > 0:
        for i in range(crate_indices.shape[0]):
            crate_dists.append(cityblock(crate_indices[i], pos))
        features.append(np.min(crate_dists))
    else:
        features.append(max_dist + 1)  # set to max dist

    # distance / direction to nearest coin
    coin_dists = []
    if coins.size > 0:
        for i in range(len(coins)):
            dist = cityblock(np.asarray(coins[i]), pos)
            coin_dists.append(dist)
        features.append(np.min(coin_dists))

        # determine coin direction
        coin_dir = np.sign(coins[np.argmin(coin_dists)] - pos)
        features.append(coin_dir[0])
        features.append(coin_dir[1])
    else:
        features.append(max_dist + 1)  # set to max dist
        features.append(0)  # no coin direction
        features.append(0)

    # prevent invalid actions
    # tile to the right
    if pos[0] + 1 > field.shape[0]:
        features.append(-1)
    else:
        features.append(field[pos[0] + 1, pos[1]])

    # tile to the left
    if pos[0] - 1 < field.shape[0]:
        features.append(-1)
    else:
        features.append(field[pos[0] - 1, pos[1]])

    # tile below
    if pos[1] + 1 > field.shape[1]:
        features.append(-1)
    else:
        features.append(field[pos[0], pos[1] + 1])

    # tile above
    if pos[1] - 1 < field.shape[1]:
        features.append(-1)
    else:
        features.append(field[pos[0], pos[1] - 1])

    # check explosion map
    # tile to the right
    if pos[0] + 1 > field.shape[0]:
        features.append(0)
    else:
        features.append(explosion_map[pos[0] + 1, pos[1]])

    # tile to the left
    if pos[0] - 1 < field.shape[0]:
        features.append(0)
    else:
        features.append(explosion_map[pos[0] - 1, pos[1]])

    # tile below
    if pos[1] + 1 > field.shape[1]:
        features.append(0)
    else:
        features.append(explosion_map[pos[0], pos[1] + 1])

    # tile above
    if pos[1] - 1 < explosion_map.shape[1]:
        features.append(0)
    else:
        features.append(explosion_map[pos[0], pos[1] - 1])

    # aggressiveness
    features.append(game_state['step'] * others.shape[0])

    # end = time.time()
    # print(end - start)
    if TRANSFORMER:
        return TRANSFORMER.transform(np.concatenate((state_as_features, np.array(features))))
    else:
        return np.concatenate((state_as_features, np.array(features)))
