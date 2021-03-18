import pickle
import random
import numpy as np
from datetime import datetime

import csv
from collections import namedtuple, deque
from typing import List
from enum import Enum

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cityblock
from agent_code.training_data.train_data_utils import read_train_data
import events as e

from .callbacks import state_to_features, get_transformer


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
GAMMA = 0.95  # discount value

# Custom events
DOUBLE_KILL = "DOUBLE_KILL"
DOUBLE_COIN = "DOUBLE_COIN"
LAST_AGENT_ALIVE = "LAST_AGENT_ALIVE"
# IN_SAFE_SPOT = "IN_SAFE_SPOT"
CLOSER_TO_OPPONENT = "CLOSER_TO_OPPONENT"
CLOSER_TO_COIN = "CLOSER_TO_COIN"
FURTHER_FROM_OPPONENT = "FURTHER_FROM_OPPONENT"
FURTHER_FROM_COIN = "FURTHER_FROM_COIN"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.event_map = dict.fromkeys([e.KILLED_OPPONENT, e.COIN_COLLECTED], 0)
    self.reward_per_epoch = 0
    self.number_of_epoch = 1
    self.last_survivor = False

    # prepare output files
    base_dir = "./logs/"
    now = datetime.now()
    self.datetime_str = now.strftime("%m-%d-%Y-%H-%M-%S")
    self.csv_filename = base_dir + "rewards_per_epoch" + "_" + self.datetime_str + ".csv"
    self.csv_actions_filename = base_dir + "actions" + "_" + self.datetime_str + ".csv"

    # load pre-collected training data
    if USE_TRAIN_SET:
        pre_train_agent(self, "../training_data/train_data2.npy", 5000, mini_batch_gradient_descent)


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
    self.transitions.append(Transition(state_to_features(
        old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

    self.reward_per_epoch += reward_from_events(self, events)
    # append_data_to_csv(self.csv_actions_filename, self.number_of_epoch, events)


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

    self.number_of_epoch += 1
    # Store the model
    save_model(self, f"../models/my-saved-model-{self.datetime_str}.pt")

    # append_data_to_csv(self.csv_filename, last_game_state['round'], self.reward_per_epoch)
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


def stochastic_gradient_descent(self, alpha=0.0001, epochs=10000):
    states, actions, next_states, rewards = get_transitions_as_matrices(self)

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
    # states, actions, next_states, rewards = get_transitions_as_matrices(self)

    if actions.shape[0] > batch_size:
        batch_indices = random.sample(range(0, actions.shape[0]), batch_size)
        states = states[batch_indices, :]
        actions = actions[batch_indices]
        next_states = next_states[batch_indices, :]
        rewards = rewards[batch_indices]

    for action in Action:
        mask_of_action = (actions == action.value)

        # check if action was performed
        if np.sum(mask_of_action) > 0:
            states_for_action = states[mask_of_action, :]
            next_states_for_action = next_states[mask_of_action, :]
            rewards_for_action = rewards[mask_of_action]

            y_t = td_q_learning(self, next_states_for_action, rewards_for_action, GAMMA)
            beta = self.model[:, action.value]

            for i in range(epochs):
                y_pred = np.matmul(states_for_action, beta)
                # calculate the derivative of the loss function with respect to beta
                y = (y_t - y_pred).reshape(y_t.shape[0], 1)
                d_beta = (alpha / np.sum(mask_of_action)) * np.sum((states_for_action * y).T, axis=1)

                # update the weights
                beta += d_beta
            self.model[:, action.value] = beta


def batch_gradient_descent(self, learning_rate=0.0001, epochs=1000):
    actionFeatureMap, actionRewardMap = get_transitions_as_matrices(self)

    for action, value in actionFeatureMap.items():
        X = np.array(value)

        # centralize and standardize the features X
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        y_t = np.array(actionRewardMap[action])

        beta = self.action_models[action]
        n = len(X)

        for i in epochs:
            y_pred = X * beta
            # calculate the derivative of the loss function with respect to beta
            D_beta = (-2 / n) * sum((X.T * (y_t - y_pred)), axis=0)

            # update the weights
            beta -= learning_rate * D_beta

        self.action_models[action] = beta


def td_q_learning(self, next_state, reward, gamma=GAMMA):
    return reward + gamma * np.max(np.matmul(next_state, self.model))


def n_step_td_q_learning(self, next_state, t, n, gamma=GAMMA):
    transitions = list(self.transitions)

    for i in range(1, n + 1):
        reward_sum = gamma ** (i - 1) * transitions[t + i]['reward']
    return reward_sum + gamma ** n * np.max(np.matmul(transitions[t + n]['state'], self.model))


def get_transitions_as_matrices(self):
    actions = []
    states = []
    next_states = []
    rewards = []

    """
    Transition 0 isn't relevant -> excluded
    Last transition -> next state nonexistent -> not relevant -> no estimation for y required bc reward is known in last round
    """

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
    rewards, actions, old_state_features, new_state_features = read_train_data(file)

    old_state_features = get_transformer().transform(standardize(old_state_features))
    new_state_features = get_transformer().transform(standardize(new_state_features))

    for i in range(iterations):
        gd_method(self, old_state_features, actions, new_state_features, rewards)
        print(f"{i}/{iterations}")
    save_model(self, f"../models/pre-trained-model-{self.datetime_str}.pt")


def standardize(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler.transform(X)
