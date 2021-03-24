import os
import numpy as np
import torch

from .deep_q_net import DeepQNet
from scipy.spatial.distance import cityblock

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# hyperparameters / training config
EPSILON_STRATEGY = "GREEDY_DECAY_LINEAR"  # GREEDY_DECAY_LINEAR / GREEDY
EPSILON_START_VALUE = 1.0
EPSILON_END_VALUE = 0.01
SOFTMAX = False  # define whether the argmax or the softmax is used during training
TAU = 5  # param for softmax policy

MAX_GAME_STEPS = 401
NUMBER_EPISODES = 10000

# external file locations
MODEL = "../models/pretrain_v"
MODEL_TARGET = "../models/pretrain_target_v"


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
    self.number_of_features = 26
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if self.train:
        self.logger.info("Entering training mode.")

        if os.path.isfile(MODEL):
            self.logger.info("Train based on existing model.")
            with open(MODEL, "rb") as file:
                self.model = DeepQNet(self.number_of_features, 0).to(self.device)
                self.model.load_state_dict(torch.load(file))

            with open(MODEL_TARGET, "rb") as file:
                self.model_target = DeepQNet(self.number_of_features, 0).to(self.device)
                self.model_target.load_state_dict(torch.load(file))
        else:
            print("New Model initialized")
            self.model = DeepQNet(self.number_of_features, 0).to(self.device)
            self.model_target = DeepQNet(self.number_of_features, 0).to(self.device)

        set_epsilon(self)

    else:
        self.logger.info("Loading model from saved state.")
        with open("../models/qnet_0", "rb") as file:
            self.model = DeepQNet(self.number_of_features, 0)
            self.model.load_state_dict(torch.load(file))
            self.model.eval()


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
    """
    Setting up weights randomly.
    """
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
    features = state_to_features(game_state)
    features_tensor = torch.from_numpy(features).float().to(self.device)
    predicted_reward = self.model(features_tensor)
    action = torch.argmax(predicted_reward)
    self.logger.info(f'Selected action: {action}')

    # computation of Q-values

    Q_sa = action
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

    For distance computations the Manhattan distance is used.

    :param game_state:  A dictionary describing the current game board.
    :return features: A np.array of features
    """

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # custom feature vector (initially a list)
    features = []

    # load game state
    field = game_state['field']
    bombs = np.asarray(game_state['bombs'])
    pos = np.asarray(game_state['self'][3])
    coins = np.asarray(game_state['coins'])
    others = np.asarray(game_state['others'])

    # max distance -> longest distance with shortest path -> diagonal
    max_dist = cityblock((1, 1), game_state['field'].shape)

    # features.append(game_state['round'])

    # rel. time of game
    features.append(relative(MAX_GAME_STEPS, game_state['step']))

    # rel. score
    max_score = 9  # 9 coins + 15 opponents
    features.append(relative(max_score, game_state['self'][1]))

    # can bomb
    features.append(int(game_state['self'][2]))

    # distance to others
    others_dists = np.zeros(3)
    if others.shape[0] > 0:
        for i in range(len(others)):
            dist = cityblock(np.asarray(others[i, 3]), pos)
            others_dists[i] = relative(max_dist, dist)
        others_dists = np.sort(others_dists).tolist()
    # features += others_dists
    # TODO: direction to nearest opponent

    # distance to bombs
    bomb_dists = np.zeros(4)
    if bombs.size > 0:
        for i in range(len(bombs)):
            dist = cityblock(np.asarray(bombs[i, 0]), pos)
            bomb_dists[i] = relative(max_dist, dist)
        bomb_dists = np.sort(bomb_dists)
    features += bomb_dists.tolist()

    # danger zone to determine if agent would get hit by bombs
    danger_zone = []
    for b in bombs:
        if np.abs(pos[0] - b[0][0]) <= 3 or np.abs(pos[1] - b[0][1]) <= 3:
            if pos[0] == b[0][0]:
                y_pos = pos[1] - b[0][1]
                if y_pos > 0:
                    if np.sum(np.where(field[pos[0], b[0][1]:pos[1]] == -1)) == 0:
                        danger_zone.append(relative(4, b[1]))
                    else:
                        danger_zone.append(0)
                else:
                    if np.sum(np.where(field[pos[0], pos[1]:b[0][1]] == -1)) == 0:
                        danger_zone.append(relative(4, b[1]))
                    else:
                        danger_zone.append(0)
            elif pos[1] == b[0][1]:
                x_pos = pos[0] - b[0][0]
                if x_pos > 0:
                    if np.sum(np.where(field[b[0][0]:pos[0], pos[1]] == -1)) == 0:
                        danger_zone.append(relative(4, b[1]))
                    else:
                        danger_zone.append(0)
                else:
                    if np.sum(np.where(field[pos[0]:b[0][0], pos[1]] == -1)) == 0:
                        danger_zone.append(relative(4, b[1]))
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
    nearest_crate = None
    if crate_indices.shape[0] > 0:
        for i in range(crate_indices.shape[0]):
            dist = cityblock(crate_indices[i], pos)
            crate_dists.append(relative(max_dist, dist))
        features.append(np.max(crate_dists))
        nearest_crate = crate_indices[np.argmax(crate_dists)]
    else:
        features.append(0)  # set to max dist

    # get indicator on which direction to go for finding a crate
    # indicated for left, right, top, bottom
    crate_dir = np.zeros(4)
    if nearest_crate is not None:
        crate_diff = pos - nearest_crate
        if crate_diff[0] > 0:
            crate_dir[0] = crate_diff[0]
        else:
            crate_dir[1] = np.abs(crate_diff[0])

        if crate_diff[1] > 0:
            crate_dir[2] = crate_diff[1]
        else:
            crate_dir[3] = np.abs(crate_diff[1])
        if np.sum(crate_dir) == 0:
            features += crate_dir.tolist()
        else:
            features += (crate_dir/np.sum(crate_dir)).tolist()
    else:
        features += crate_dir.tolist()

    # distance to nearest coin
    coin_dists = []
    nearest_coin = None
    if coins.size > 0:
        for i in range(len(coins)):
            dist = cityblock(np.asarray(coins[i]), pos)
            coin_dists.append(relative(max_dist, dist))
        features.append(np.max(coin_dists))
        nearest_coin = coins[np.argmax(coin_dists)]
    else:
        features.append(0)  # set to max dist

    xx = np.zeros(4)
    if nearest_coin is not None:
        coin_diff = pos - nearest_coin
        if coin_diff[0] > 0:
            coin_dir[0] = coin_diff[0]
        else:
            coin_dir[1] = np.abs(coin_diff[0])

        if coin_diff[1] > 0:
            coin_dir[2] = coin_diff[1]
        else:
            coin_dir[3] = np.abs(coin_diff[1])
        if np.sum(coin_dir) == 0:
            features += coin_dir.tolist()
        else:
            features += (coin_dir / np.sum(coin_dir)).tolist()
    else:
        features += coin_dir.tolist()

    # look around and check for free tiles
    # tile to the left
    features.append(np.abs(field[pos[0] - 1, pos[1]]))
    # tile to the right
    features.append(np.abs(field[pos[0] + 1, pos[1]]))
    # tile below
    features.append(np.abs(field[pos[0], pos[1] + 1]))
    # tile above
    features.append(np.abs(field[pos[0], pos[1] - 1]))

    # number of opponents hit by a bomb
    hit_counter = 0
    if game_state["self"][2]:
        pos_array = np.array(pos)
        for enemy in others:
            pos_dif = pos_array - enemy[3]
            if pos_dif[0] == 0 or pos_dif[1] == 0 and np.sum(pos_dif) <= 3:
                # check if there is a stone wall somewhere in between the bomb and an opponent
                if pos_dif[0] == 0:
                    y_pos = pos_dif[1]
                    if y_pos > 0:
                        if np.sum(np.where(field[pos[0], enemy[3][0]:pos[1]] == -1)) == 0:
                            hit_counter += 1
                    else:
                        if np.sum(np.where(field[pos[0], pos[1]:enemy[3][0]] == -1)) == 0:
                            hit_counter += 1
                elif pos_dif[1] == 0:
                    x_pos = pos_dif[0]
                    if x_pos > 0:
                        if np.sum(np.where(field[enemy[3][0]:pos[0], pos[1]] == -1)) == 0:
                            hit_counter += 1
                    else:
                        if np.sum(np.where(field[pos[0]:enemy[3][0], pos[1]] == -1)) == 0:
                            hit_counter += 1
    # features.append(relative(len(others),hit_counter))

    # number of crates hit by bomb
    max_crate_hit = 10
    # x-direction
    crates_hit = 0
    for element in field[pos[0]:pos[0] + 4, pos[1]]:
        if element == -1:
            break
        if element == 1:
            crates_hit += 1

    for element in np.flip(field[pos[0] - 3:pos[0], pos[1]]):
        if element == -1:
            break
        if element == 1:
            crates_hit += 1

    for element in field[pos[0], pos[1]:pos[1] + 4]:
        if element == -1:
            break
        if element == 1:
            crates_hit += 1

    for element in np.flip(field[pos[0], pos[1] - 3:pos[1]]):
        if element == -1:
            break
        if element == 1:
            crates_hit += 1
    features.append(relative(max_crate_hit, crates_hit))

    return np.array(features)


def relative(max_dist, dist):
    return (max_dist - dist) / max_dist

# TODO: FEATURES
# how many opponents / crates do we hit  -> 4 - num_hit / 4 -> relative
# scared ghost
# 4 - num of free fields around / 4 -> fluchtweg
# case 1: in danger zone, case 2: not in danger zone
# where to lay bomb best
# train setup good -> take best agent and train that - not only lost but avg reward / collected coins to decide which agent is better
# use papers of ba
# metrics: win ratio, avg score, avg coins, avg kills per game (avg of collected coin per game) -> summarywriter!
# ies ist klar, dass wir keine 50 agents trainieren und dann den besten nehmen können
# we left out bomb featus when coin colletion is learned


# besprechen, TS generieren, pre train, train
# define training process, which data / plots
