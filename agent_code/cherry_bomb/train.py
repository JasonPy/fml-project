import os
import random
import numpy as np
from datetime import datetime

from collections import namedtuple, deque
from typing import List
from enum import Enum

from scipy.spatial.distance import cityblock
from tqdm import tqdm

from agent_code.training_data.train_data_utils import read_h5f
from .callbacks import state_to_features
import events as e

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


# assign each action e scalar value
class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    WAIT = 4
    BOMB = 5


# shape of transition buffer entries
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# HYPERPARAMETERS
TRANSITION_HISTORY_SIZE = 32000  # size of buffer
GAMMA = 0.95  # discount value
LEARNING_RATE = 1e-4  # learning rate for RL
UPDATE_CYCLE = 3  # update every
BATCH_SIZE = 128  # batch size of sample from buffer
UPDATE_CYCLE_TARGET = 10000  # how much net is updated

# CUSTOM EVENTS
DOUBLE_KILL = "DOUBLE_KILL"
DOUBLE_COIN = "DOUBLE_COIN"
LAST_AGENT_ALIVE = "LAST_AGENT_ALIVE"
CLOSER_TO_OPPONENT = "CLOSER_TO_OPPONENT"
CLOSER_TO_COIN = "CLOSER_TO_COIN"
FURTHER_FROM_OPPONENT = "FURTHER_FROM_OPPONENT"
FURTHER_FROM_COIN = "FURTHER_FROM_COIN"

# apply pre training
PRE_TRAIN = False

# output filenames (saved in models folder)
MODEL = "testmodel"


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
    self.model_name = MODEL + "_" + datetime.now().strftime('%d-%m_%H-%M')
    os.mkdir('../models/' + self.model_name)

    # set random seed
    random.seed(1)

    # setup Q- Network
    self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    # Initialize time step (for updating every UPDATE_EVERY steps)
    self.update_step = 0
    self.update_step_target = 0
    self.running_loss = 0

    self.writer = SummaryWriter(
        f'C:/Users/Jason/OneDrive/Master/1. Semester/FML - Fundamentals of Machine Learning/Exercises/Final Project/fml-project/agent_code/models/{self.model_name}/tensorboard')

    if PRE_TRAIN:
        pre_train_agent(self, "../training_data/h5f_crate_train_data.h5", 1000000)


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

    self.update_step += 1
    self.update_step_target += 1

    if self.update_step % UPDATE_CYCLE == 0:
        self.update_step = 0

        if len(self.transitions) > BATCH_SIZE:
            xp = sample_transitions(self)
            learn(self, xp)

    # update target net only
    if self.update_step_target % UPDATE_CYCLE_TARGET == 0:
        self.update_step_target = 0
        update_target_net(self)


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

    reward = reward_from_events(self, events)
    self.transitions.append(Transition(state_to_features(
        last_game_state), last_action, None, reward))

    # Store the model
    save_model(self, f"../models/{self.model_name}/model", f"../models/{self.model_name}/target")

    self.writer.add_scalar("Rewards", self.reward_per_epoch, self.number_of_epoch)

    self.writer.add_scalars('Events', {
        'invalid_actions': self.event_map[e.INVALID_ACTION],
        'coins_collected': self.event_map[e.COIN_COLLECTED],
        'bombs dropped': self.event_map[e.BOMB_DROPPED],
    }, self.number_of_epoch)

    self.writer.add_scalar("Epsilon", self.epsilon(self.number_of_epoch), self.number_of_epoch)

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


def learn(self, xp):
    # samples from transition buffer
    states, next_states, actions, rewards = xp

    # define loss criteria
    loss_criterion = torch.nn.MSELoss()

    # use training and evaluation flag
    self.model.train()
    self.model_target.eval()

    # retrieve q value for each state
    predicted_targets = self.model(states).gather(1, actions)

    # compute future reward (TD-Q-learning)
    labels_next = self.model_target(next_states).max(1)[0].unsqueeze(1)
    labels = rewards + GAMMA * labels_next

    # compute loss
    loss = loss_criterion(predicted_targets, labels).to(self.device)

    # more stable gradient updates by using gradient clipping
    nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1)

    # clearing the previous data
    self.optimizer.zero_grad()

    # apply backpropagation
    loss.backward()

    self.optimizer.step()

    self.writer.add_scalar('Loss', loss.item(), self.number_of_epoch)


def update_target_net(self):
    """
    alternating optimization every UPDATE_CYCLE steps
    """
    for target_param, local_param in zip(self.model_target.parameters(),
                                         self.model.parameters()):
        target_param.data.copy_(local_param.data)


def reset_events(self):
    """
    reset rewards and events per epoch
    """
    self.reward_per_epoch = 0
    self.last_survivor = False
    self.event_map = dict.fromkeys(self.event_map, 0)


def sample_transitions(self):
    """
    Transition 0 isn't relevant -> excluded
    Last transition -> next state nonexistent -> not relevant -> no estimation for y required bc reward is known in last round
    """
    actions = np.empty((BATCH_SIZE, 1))
    states = np.empty((BATCH_SIZE, self.number_of_features))
    next_states = np.empty((BATCH_SIZE, self.number_of_features))
    rewards = np.empty((BATCH_SIZE, 1))

    # use random samples of buffer
    samples = random.sample(self.transitions, k=BATCH_SIZE)

    for t in range(len(samples)):
        actions[t] = Action[samples[t].action].value
        states[t, :] = samples[t].state

        if samples[t].next_state is not None:
            next_states[t, :] = samples[t].next_state
        else:
            # TODO: another strategy for filling next_state
            next_states[t, :] = np.zeros(self.number_of_features)
        rewards[t] = samples[t].reward

    actions = torch.from_numpy(actions).long().to(self.device)
    states = torch.from_numpy(states).float().to(self.device)
    next_states = torch.from_numpy(next_states).float().to(self.device)
    rewards = torch.from_numpy(rewards).float().to(self.device)

    # TODO: standardize or normalize data?

    return states, next_states, actions, rewards


def save_model(self, file_name, file_name_target):
    """
    save target and actual model
    """
    torch.save(self.model.state_dict(), file_name)
    torch.save(self.model_target.state_dict(), file_name_target)


def pre_train_agent(self, file, iterations):
    """
    pre train based on expert data to get good starting weights
    """
    rewards, actions, old_state_features, new_state_features = read_h5f(file, "coin_collect_data")

    num_elements = new_state_features.shape[0]
    indices = np.arange(num_elements).tolist()

    for i in tqdm(range(iterations)):
        self.number_of_epoch += 1
        indices = random.sample(indices, BATCH_SIZE)
        batch_next_states = new_state_features[indices]
        batch_states = old_state_features[indices]
        batch_rewards = rewards[indices]
        batch_actions = actions[indices]

        tensor_actions = torch.from_numpy(batch_actions[:, np.newaxis]).long().to(self.device)
        tensor_states = torch.from_numpy(batch_states).float().to(self.device)
        tensor_next_states = torch.from_numpy(batch_next_states).float().to(self.device)
        tensor_rewards = torch.from_numpy(batch_rewards[:, np.newaxis]).float().to(self.device)

        if self.update_step % UPDATE_CYCLE == 0:
            self.update_step = 0
            learn(self, (tensor_states, tensor_next_states, tensor_actions, tensor_rewards))

        #if self.update_step_target % UPDATE_CYCLE_TARGET == 0:
        #    self.update_step_target = 0
        #    update_target_net(self)

        self.update_step += 1
        # self.update_step_target += 1

    update_target_net(self)
    save_model(self, file_name="../models/pretrain_crate_no_update", file_name_target="../models/pretrain_crate_no_update_target")
