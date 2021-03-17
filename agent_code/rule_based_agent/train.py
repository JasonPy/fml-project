import pickle
import random
import numpy as np
from datetime import datetime

import csv
from collections import namedtuple, deque
from typing import List
from enum import Enum
from agent_code.cherry_bomb.callbacks import state_to_features
from agent_code.cherry_bomb.train import reward_from_events, reset_events, get_custom_events, Action

from agent_code.train_data_utils import save_train_data

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import events as e


class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    WAIT = 4
    BOMB = 5


# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 400  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
GAMMA = 0.8

# Custom events
DOUBLE_KILL = "DOUBLE_KILL"
DOUBLE_COIN = "DOUBLE_COIN"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # # prepare output files
    base_dir = "../resources/"
    self.train_data_path = base_dir + 'train_data2.npy'
    # self.csv_features_filename = base_dir + "TS_features_03-16-2021, 19-01-39.csv"
    # self.csv_rewards_filename = base_dir + "TS_rewards" + "_" + datetime_str + ".csv"
    #
    self.event_map = dict.fromkeys([e.KILLED_OPPONENT, e.COIN_COLLECTED], 0)
    # self.reward_writer = csv.writer(open(self.csv_rewards_filename, 'a', newline=''), quoting=csv.QUOTE_NONE)
    #
    # self.feature_writer = csv.writer(open(self.csv_features_filename, 'a', newline=''), delimiter=',',
    #                                  quoting=csv.QUOTE_NONE)
    self.train_list = []

    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.event_map = dict.fromkeys([e.KILLED_OPPONENT, e.COIN_COLLECTED], 0)
    self.reward_per_epoch = 0
    self.number_of_epoch = 1
    self.nearest_opponent = None
    self.nearest_coin = None
    self.last_survivor = False


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    # Add custom events
    if len(events) > 0:
        for ev in events:
            if ev in self.event_map:
                self.event_map[ev] += 1
        events = events + get_custom_events(self, old_game_state, new_game_state)

    old_state_features = state_to_features(old_game_state)
    new_state_features = state_to_features(new_game_state)

    if old_state_features is not None and new_state_features is not None:
        reward_action_array = np.append(reward_from_events(self, events), Action[self_action].value)
        features = np.append(old_state_features, new_state_features)
        features = np.append(reward_action_array, features)
        self.train_list.append(features)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    last_game_state_features = state_to_features(last_game_state)
    features = np.append(last_game_state_features, np.zeros(len(last_game_state_features)))

    reward_action_array = np.append(reward_from_events(self, events), Action[last_action].value)
    features = np.append(reward_action_array, features)
    self.train_list.append(features)

    self.number_of_epoch += 1
    if (self.number_of_epoch == 4000):
        save_train_data(self.train_list, self.train_data_path)
        self.train_list = []

    reset_events(self)
