import pickle
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm

import csv
from collections import namedtuple, deque
from typing import List
from enum import Enum

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cityblock
from agent_code.training_data.train_data_utils import read_train_data, read_h5f, read_rows_h5f
import events as e

from .callbacks import state_to_features
import os
import torch
from torch.utils.tensorboard import SummaryWriter


# assign each action e scalar value
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

# Hyper parameters
TRANSITION_HISTORY_SIZE = 4096  # keep last transitions
USE_TRAIN_SET = True  # use training set for pre-learning
TRAIN_FILE = "../training_data/h5f_train_data.h5"
GAMMA = 0.95  # discount value
GRADIENT_CLIPPING = False
OLD_LOSS = None

# Custom events
DOUBLE_KILL = "DOUBLE_KILL"
DOUBLE_COIN = "DOUBLE_COIN"
LAST_AGENT_ALIVE = "LAST_AGENT_ALIVE"
# IN_SAFE_SPOT = "IN_SAFE_SPOT"
CLOSER_TO_OPPONENT = "CLOSER_TO_OPPONENT"
CLOSER_TO_COIN = "CLOSER_TO_COIN"
FURTHER_FROM_OPPONENT = "FURTHER_FROM_OPPONENT"
FURTHER_FROM_COIN = "FURTHER_FROM_COIN"
MODEL = "Q_value_model"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.event_map = dict.fromkeys([e.KILLED_OPPONENT, e.COIN_COLLECTED, e.INVALID_ACTION, e.BOMB_DROPPED], 0)
    self.reward_per_epoch = 0
    self.number_of_epoch = 0
    self.last_survivor = False

    # prepare output files
    self.datetime = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    # self.csv_rewards = f"./logs/rewards_{self.datetime}.csv"
    # self.csv_actions = f"./logs/actions_{self.datetime}.csv"
    # self.csv_loss = f"./logs/loss_{self.datetime}.csv"
    # self.csv_steps = f"./logs/steps_{self.datetime}.csv"

    self.model_name = MODEL + "_" + datetime.now().strftime('%d-%m_%H-%M')
    os.mkdir('../models/' + self.model_name)
    self.writer = SummaryWriter(
        f'C:/Users/Jason/OneDrive/Master/1. Semester/FML - Fundamentals of Machine Learning/Exercises/Final Project/fml-project/agent_code/models/{self.model_name}/tensorboard')

    # load pre-collected training data
    if USE_TRAIN_SET:
        pre_train_agent(self, TRAIN_FILE, 40000, mini_batch_gradient_descent) # mini_batch_gradient_descent


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
    self.logger.debug(
        f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Prevents adding the step 0
    if self_action is None:
        return

    # Add custom events
    if len(events) > 0:
        for ev in events:
            if ev in self.event_map:
                self.event_map[ev] += 1
        events += get_custom_events(self, old_game_state, new_game_state)

    # state_to_features is defined in callbacks.py
    rewards = reward_from_events(self, events)
    self.transitions.append(Transition(state_to_features(
        old_game_state), self_action, state_to_features(new_game_state), rewards))

    self.reward_per_epoch += rewards
    # append_data_to_csv(self.csv_actions, self.number_of_epoch, events)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """

    reward = reward_from_events(self, events)
    self.logger.debug(
        f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(
        last_game_state), last_action, None, reward))

    # stochastic_gradient_descent(self, alpha=0.0001, epochs=10000)
    mini_batch_gradient_descent(self, *get_transitions_as_matrices(self))

    # Store the model
    save_model(self, f"../models/my-saved-model-{self.datetime}.pt")

    # save rewards in csv
    # append_data_to_csv(self.csv_rewards, self.number_of_epoch, self.reward_per_epoch)
    # append_data_to_csv(self.csv_steps, self.number_of_epoch, last_game_state["steps"])

    self.writer.add_scalar("Rewards", self.reward_per_epoch, self.number_of_epoch)

    self.writer.add_scalars('Events', {
        'invalid_actions': self.event_map[e.INVALID_ACTION],
        'coins_collected': self.event_map[e.COIN_COLLECTED],
        'bombs dropped': self.event_map[e.BOMB_DROPPED],
    }, self.number_of_epoch)

    self.writer.add_scalar("Epsilon", self.epsilon(self.number_of_epoch), self.number_of_epoch)
    self.writer.add_scalar("Steps per epoch", last_game_state["step"], self.number_of_epoch)

    self.number_of_epoch += 1

    # reset custom events
    reset_events(self)


def reward_from_events(self, events: List[str]) -> int:
    """
    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -1,
        e.INVALID_ACTION: -10,

        e.BOMB_DROPPED: -1,
        e.BOMB_EXPLODED: 0,

        e.CRATE_DESTROYED: 30,
        e.COIN_FOUND: 50,
        e.COIN_COLLECTED: 75,

        e.KILLED_OPPONENT: 100,
        e.KILLED_SELF: -400,  # triggered with GOT_KILLED

        e.GOT_KILLED: -300,
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: 150,

        # custom rewards
        DOUBLE_COIN: 40,
        DOUBLE_KILL: 40,
        LAST_AGENT_ALIVE: 175,
        # TODO: IN_SAFE_SPOT: 40,
        CLOSER_TO_OPPONENT: 0.2,
        CLOSER_TO_COIN: 0.2,
        FURTHER_FROM_OPPONENT: -0.2,
        FURTHER_FROM_COIN: -0.2,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def get_custom_events(self, old_game_state: dict, new_game_state: dict) -> List[str]:
    # TODO: more generic by using state to get amount of players
    custom_events = []

    if self.event_map[e.KILLED_OPPONENT] == 2:
        custom_events.append(DOUBLE_KILL)

        # prevent throwing KILLED_OPPONENT multiple times
        self.event_map[e.KILLED_OPPONENT] += 1

    if self.event_map[e.COIN_COLLECTED] == 4:
        custom_events.append(DOUBLE_COIN)

        # prevent throwing COIN_COLLECTED multiple times
        self.event_map[e.COIN_COLLECTED] += 1

    # check if agent is the last survivor
    if len(new_game_state['others']) < 1 and len(new_game_state['others']) < len(
            old_game_state['others']) and self.last_survivor is False:
        self.last_survivor = True
        custom_events.append(LAST_AGENT_ALIVE)

    # Check if distance to nearest opponent has evolved or decreased
    if len(old_game_state['others']) > 0:
        # first step - no distances yet
        min_dist = np.inf
        for o in old_game_state['others']:
            dist = cityblock(np.asarray(o[3]), np.asarray(old_game_state['self'][3]))
            if dist < min_dist:
                min_dist = dist

        # calculate new distances and collect rewards
        self_pos = np.asarray(new_game_state['self'][3])
        is_closer = False
        for o in new_game_state['others']:
            o = np.asarray(o)
            dist = cityblock(np.asarray(o[3]), self_pos)

            if dist < min_dist:
                is_closer = True
                custom_events.append(CLOSER_TO_OPPONENT)
                break
        if not is_closer:
            custom_events.append(FURTHER_FROM_OPPONENT)

    # Check if distance to nearest coin has evolved or decreased
    if len(old_game_state['coins']) > 0:

        # first step - no distances yet
        min_dist = np.inf
        for c in old_game_state['coins']:
            dist = cityblock(np.asarray(c), np.asarray(old_game_state['self'][3]))
            if dist < min_dist:
                min_dist = dist

        # calculate new distances and collect rewards
        self_pos = np.asarray(new_game_state['self'][3])
        is_closer = False
        for c in new_game_state['coins']:
            dist = cityblock(np.asarray(c), self_pos)
            if dist < min_dist:
                is_closer = True
                custom_events.append(CLOSER_TO_COIN)
                break
        if not is_closer:
            custom_events.append(FURTHER_FROM_COIN)

    return custom_events


def reset_events(self):
    self.reward_per_epoch = 0
    self.last_survivor = False
    self.event_map = dict.fromkeys(self.event_map, 0)


def stochastic_gradient_descent(self, states, actions, next_states, rewards, alpha=0.0001, epochs=10):
    # train for each action a model with SGD
    for action in Action:
        mask_of_action = (actions == action.value)

        # check if action was performed
        if np.sum(mask_of_action) > 0:
            states_for_action = states[mask_of_action, :]
            next_states_for_action = next_states[mask_of_action, :]
            rewards_for_action = rewards[mask_of_action]

            y_t = td_q_learning(self, next_states_for_action, rewards_for_action, GAMMA)

            lin_reg = SGDRegressor(max_iter=epochs, learning_rate='optimal', alpha=alpha, tol=1e-3, fit_intercept=False,
                                   warm_start=True)
            lin_reg.fit(states_for_action, y_t, coef_init=self.model[:, action.value])

            self.model[:, action.value] = lin_reg.coef_


def mini_batch_gradient_descent(self, states, actions, next_states, rewards, alpha=0.0001, epochs=10000,
                                batch_size=256):
    if actions.shape[0] > batch_size:
        batch_indices = random.sample(range(0, actions.shape[0]), batch_size)
        states = states[batch_indices, :]
        actions = actions[batch_indices]
        next_states = next_states[batch_indices, :]
        rewards = rewards[batch_indices]

        current_loss = np.zeros(6)

    for action in Action:
        mask_of_action = (actions == action.value)

        # check if action was performed
        if np.sum(mask_of_action) > 0:
            states_for_action = states[mask_of_action, :]
            next_states_for_action = next_states[mask_of_action, :]
            rewards_for_action = rewards[mask_of_action]

            y_t = td_q_learning(self, next_states_for_action, rewards_for_action, GAMMA)
            beta = self.model[:, action.value]

            d_beta_array = np.zeros(epochs)
            d_beta = 0
            for i in range(epochs):
                y_pred = np.matmul(states_for_action, beta)
                # calculate the derivative of the loss function with respect to beta
                y = (y_t - y_pred).reshape(y_t.shape[0], 1)
                d_beta = (alpha / np.sum(mask_of_action)) * np.sum((states_for_action * y).T, axis=1)

                if GRADIENT_CLIPPING:
                    d_beta = np.where(d_beta < -1, -1, d_beta)
                    d_beta = np.where(d_beta > 1, 1, d_beta)
                    # gradient clipping

                # store loss for tensorboard plot
                d_beta_array[i] = np.average(y)
                # update the weights
                beta += d_beta

            current_loss[action.value] = np.average(d_beta_array)

            # append_data_to_csv(self.loss_csv, self.number_of_epoch, d_beta)
            self.model[:, action.value] = beta

    # tensorboard plots
    global OLD_LOSS

    if OLD_LOSS is not None:
        indices_of_missing_actions = np.argwhere(current_loss == 0)
        current_loss[indices_of_missing_actions] = OLD_LOSS[indices_of_missing_actions]

    OLD_LOSS = current_loss

    self.writer.add_scalars('Loss per action', {
        'UP': current_loss[0],
        'RIGHT': current_loss[1],
        'DOWN': current_loss[2],
        'LEFT': current_loss[3],
        'WAIT': current_loss[4],
        'BOMB': current_loss[5],
    }, self.number_of_epoch)

    a = np.average(current_loss)
    self.writer.add_scalar('Loss', np.average(current_loss), self.number_of_epoch)


def td_q_learning(self, next_state, reward, gamma=GAMMA):
    return reward + gamma * np.max(np.matmul(next_state, self.model))


def n_step_td_q_learning(self, next_state, t, n, gamma=GAMMA):
    transitions = list(self.transitions)

    for i in range(1, n + 1):
        reward_sum = gamma ** (i - 1) * transitions[t + i]['reward']
    return reward_sum + gamma ** n * np.max(np.matmul(transitions[t + n]['state'], self.model))


def get_transitions_as_matrices(self):
    """
    Transition 0 isn't relevant -> excluded
    Last transition -> next state nonexistent -> not relevant -> no estimation for y required bc reward is known in last round
    """
    actions = []
    states = []
    next_states = []
    rewards = []

    for transition in self.transitions:
        actions.append(Action[transition.action].value)
        states.append(transition.state[0, :])

        if transition.next_state is not None:
            next_states.append(transition.next_state[0, :])
        else:
            # TODO: another strategy for filling next_state
            next_states.append(np.zeros(self.number_of_features))
        rewards.append(transition.reward)

    actions = np.array(actions)
    states = np.array(states)
    next_states = np.array(next_states)
    rewards = np.array(rewards)

    # standardize data
    states = standardize(states)
    next_states = standardize(next_states)

    return states, actions, next_states, rewards


def append_data_to_csv(filename, epoch, data):
    with open(filename, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([epoch, data])


def save_model(self, file_name):
    with open(file_name, "wb") as file:
        pickle.dump(self.model, file)


def pre_train_agent(self, file, iterations, gd_method):
    rewards, actions, old_state_features, new_state_features = read_h5f(file, "coin_collect_data")

    # TODO: standardize?
    old_state_features = old_state_features
    new_state_features = new_state_features

    for i in tqdm(range(iterations)):
        self.number_of_epoch += 1
        gd_method(self, old_state_features, actions, new_state_features, rewards, epochs=1) # epochs=1 for mini batch
    save_model(self, f"../models/linear-Q-value-pre-trained-model-{self.datetime}.pt")


def standardize(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler.transform(X)
