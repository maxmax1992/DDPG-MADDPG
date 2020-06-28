# edited from stable baselines
import random
import torch
# import numpy as np

class ReplayBuffer(object):
    def __init__(self, size, n_agents, device):
        """
        Implements a ring buffer (FIFO).
        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.device = device
        self.n_agents = n_agents

    def __len__(self):
        return len(self._storage)

    @property
    def storage(self):
        """[(Union[np.ndarray, int], Union[np.ndarray, int], float, Union[np.ndarray, int], bool)]: content of the replay buffer"""
        return self._storage

    @property
    def buffer_size(self):
        """float: Max capacity of the buffer"""
        return self._maxsize

    def can_sample(self, n_samples):
        """
        Check if n_samples samples can be sampled
        from the buffer.
        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self):
        """
        Check whether the replay buffer is full or not.
        :return: (bool)
        """
        return len(self) == self.buffer_size

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer
        :param obs_t: (Union[np.ndarray, int]) the last observation
        :param action: (Union[np.ndarray, int]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Union[np.ndarray, int]) the current observation
        :param done: (bool) is the episode done
        """
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        # gc.collect()
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def to_tensor(self, nparr, is_number=False):
        # __import__('ipdb').set_trace()
        if is_number:
            return torch.tensor([nparr]).to(self.device).float()
        return torch.from_numpy(nparr).to(self.device).float()

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = \
            [[] for i in range(self.n_agents)], \
            [[] for i in range(self.n_agents)], \
            [[] for i in range(self.n_agents)], \
            [[] for i in range(self.n_agents)], \
            [[] for i in range(self.n_agents)]
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            # __import__('ipdb').set_trace()    
            for agent_i in range(self.n_agents):
                obses_t[agent_i].append(self.to_tensor(obs_t[agent_i]))
                actions[agent_i].append(self.to_tensor(action[agent_i]))
                rewards[agent_i].append(self.to_tensor(reward[agent_i], True))
                obses_tp1[agent_i].append(self.to_tensor(obs_tp1[agent_i]))
                # __import__('ipdb').set_trace()
                dones[agent_i].append(self.to_tensor(done[agent_i], True))
        
        # __import__('ipdb').set_trace()
        return [torch.stack(obses_t[i]).to(self.device).float() for i in range(self.n_agents)], \
            [torch.stack(actions[i]).to(self.device).float() for i in range(self.n_agents)], \
            [torch.stack(rewards[i]).to(self.device).float() for i in range(self.n_agents)], \
            [torch.stack(obses_tp1[i]).to(self.device).float() for i in range(self.n_agents)], \
            [torch.stack(dones[i]).to(self.device).float() for i in range(self.n_agents)]

    def sample(self, batch_size, **_kwargs):
        """
        Sample a batch of experiences.
        :param batch_size: (int) How many transitions to sample.
        :return:
            tuple(state_agent1, states_agent2 ... agentN), ...other tuples
        """
        # print("MEM_SIZE: ", len(self))
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
