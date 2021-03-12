import pickle
import random
import numpy as np
from collections import namedtuple, deque
from typing import List

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import events as e
from .callbacks import state_to_features
from .callbacks import ACTIONS

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 400  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Custom events
DOUBLE_KILL = "DOUBLE_KILL"
DOUBLE_COIN = "DOUBLE_COIN"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.event_map = dict.fromkeys([e.KILLED_OPPONENT,e.COIN_COLLECTED], 0)

    # intialize for each action a model
    self.action_models_weight = {}

    for i in range(len(ACTIONS)):
        self.action_models_weight[ACTIONS[i]] = self.model[:, i]
    print(self.action_models_weight)    


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

    # Add custom events
    print(events)
    if len(events) > 0:
        for ev in events:
            if ev in self.event_map:
                self.event_map[ev] += 1
        events = events + get_custom_events(self.event_map)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(
        old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(
        f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(
        last_game_state), last_action, None, reward_from_events(self, events)))

    stochastic_gradient_descent(self, learning_rate=0.0001, epochs=1000)
    
    for i in range(len(ACTIONS)):
        self.model[:, i] = self.action_models_weight[ACTIONS[i]]
    # print(self.action_models_weight)

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


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
        e.INVALID_ACTION: -5,

        e.BOMB_DROPPED: -1,
        e.BOMB_EXPLODED: 0,

        e.CRATE_DESTROYED: 2,
        e.COIN_FOUND: 1,
        e.COIN_COLLECTED: 3,

        e.KILLED_OPPONENT: 10,
        e.KILLED_SELF: -15,

        e.GOT_KILLED: -10,
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: 12,

        DOUBLE_COIN: 2,
        DOUBLE_KILL: 2
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def get_custom_events(event_map) -> List[str]:
    # TODO: more generic by using state to get amount of players
    custom_events = []
    print(event_map)

    if event_map[e.KILLED_OPPONENT] == 2:
        custom_events.append(DOUBLE_KILL)

    if event_map[e.COIN_COLLECTED] == 2:
        custom_events.append(DOUBLE_COIN)

    return custom_events


def stochastic_gradient_descent(self, learning_rate=0.0001, epochs=1000):
    actionFeatureMap, actionRewardMap = action_maps_initilization(self)

    # train for each action a model with SDG
    for action, value in actionFeatureMap.items():
        if len(value) > 0:

            X = np.array(value)
            y_t = np.array(actionRewardMap[action])

            scaler = StandardScaler()
            print(y_t.shape, y_t)
            print(X.shape, X)
            # X = scaler.fit_transform(X)

            lin_reg = SGDRegressor(max_iter=epochs, alpha=learning_rate, tol=1e-3, warm_start=True)
            lin_reg.fit(X, y_t, coef_init=self.action_models_weight[action])

            self.action_models_weight[action] = lin_reg.coef_        


def batch_gradient_descent(self, learning_rate=0.0001, epochs=1000, batch_size = 64):
    actionFeatureMap, actionRewardMap = action_maps_initilization(self)

    for action, value in actionFeatureMap.items():
        X = np.array(value)
        
        batch_indices = random.sample(range(0,len(X)), batch_size)

        batch = X[batch_indices, :]
        y_t = np.array(actionRewardMap[action])[batch_indices, :]

        # centralize and standardize the features X
        scaler = StandardScaler()
        batch = scaler.fit_transform(batch)

        y_t = np.array(actionRewardMap[action])

        beta = self.action_models[action]
      
        for i in epochs:
            y_pred = batch*beta
            # calculate the derivative of the loss function with respect to beta
            D_beta = (-2/batch_size) * sum((batch.T * (y_t - y_pred)), axis=0)

            # update the weights
            beta -= learning_rate * D_beta

        self.action_models[action] = beta

def gradient_descent(self, learning_rate=0.0001, epochs=1000):
    actionFeatureMap, actionRewardMap = action_maps_initilization(self)

    for action, value in actionFeatureMap.items():
        X = np.array(value)

        # centralize and standardize the features X
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        y_t = np.array(actionRewardMap[action])

        beta = self.action_models[action]
        n = len(X)

        for i in epochs:
            y_pred = X*beta
            # calculate the derivative of the loss function with respect to beta
            D_beta = (-2/n) * sum((X.T * (y_t - y_pred)), axis=0)

            # update the weights
            beta -= learning_rate * D_beta

        self.action_models[action] = beta


def action_maps_initilization(self):
    actionFeatureMap = {}
    actionRewardMap = {}

    for a in ACTIONS:
        actionFeatureMap[a] = []
        actionRewardMap[a] = []

    # fill map with actions as key and their corresponding features/rewards as value
    for transition in self.transitions:
        print(transition)
        actionFeatureMap[transition.action].append(transition.state)
        actionRewardMap[transition.action].append(transition.reward)

    return actionFeatureMap, actionRewardMap
