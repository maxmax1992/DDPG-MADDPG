import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from gymnasium import spaces
import gymnasium as gym

class PettingZooEnvWrapper:
    """
    A wrapper for PettingZoo environments to make them compatible with the MADDPG code
    that was originally designed for multiagent library.
    """
    
    def __init__(self, env):
        """
        Initialize the wrapper with a PettingZoo environment.
        
        Args:
            env: A PettingZoo environment
        """
        self.env = env
        self.env.reset()
        
        # Store agent names for consistent ordering
        self.agents = list(self.env.possible_agents)
        self.n_agents = len(self.agents)
        
        # Create observation and action spaces for each agent
        self.observation_space = []
        self.action_space = []
        
        for agent in self.agents:
            self.observation_space.append(self._convert_obs_space(self.env.observation_space(agent)))
            self.action_space.append(self._convert_action_space(self.env.action_space(agent)))
    
    def _convert_obs_space(self, obs_space):
        """Convert PettingZoo observation space to Box space with shape (dim,)"""
        if isinstance(obs_space, spaces.Box):
            # Flatten Box spaces to 1D
            return gym.spaces.Box(
                low=obs_space.low.flatten(),
                high=obs_space.high.flatten(),
                dtype=obs_space.dtype
            )
        elif isinstance(obs_space, spaces.Discrete):
            # Convert Discrete to one-hot vector Box
            return gym.spaces.Box(
                low=0, 
                high=1, 
                shape=(obs_space.n,),
                dtype=np.float32
            )
        else:
            raise NotImplementedError(f"Unsupported observation space: {type(obs_space)}")
    
    def _convert_action_space(self, act_space):
        """Convert PettingZoo action space to Discrete space for MADDPG"""
        if isinstance(act_space, spaces.Discrete):
            # Just return the Discrete space as is
            return act_space
        elif isinstance(act_space, spaces.Box):
            # For continuous action spaces, we need to discretize or handle differently
            # For now, we'll raise an error as original code expects discrete actions
            raise NotImplementedError("Continuous action spaces not supported yet")
        else:
            raise NotImplementedError(f"Unsupported action space: {type(act_space)}")
    
    def _process_obs(self, obs_dict):
        """Process the observation dictionary from PettingZoo to match MADDPG format"""
        observations = []
        
        # Ensure observations are in the same order as self.agents
        for agent in self.agents:
            if agent in obs_dict:
                obs = obs_dict[agent]
                
                # Handle different observation types
                if isinstance(obs, np.ndarray):
                    observations.append(obs.flatten())
                elif isinstance(self.env.observation_space(agent), spaces.Discrete):
                    # Convert discrete observation to one-hot
                    one_hot = np.zeros(self.env.observation_space(agent).n)
                    one_hot[obs] = 1.0
                    observations.append(one_hot)
                else:
                    observations.append(obs)
            else:
                # Agent is done, use zeros
                obs_shape = self.observation_space[self.agents.index(agent)].shape
                observations.append(np.zeros(obs_shape, dtype=np.float32))
                
        return observations
    
    def reset(self):
        """Reset the environment and return initial observations"""
        # Reset the environment
        result = self.env.reset()
        
        # Handle different return types for reset()
        if result is None:
            # If reset doesn't return anything, we need to handle it differently
            # First create an empty observation dictionary
            obs_dict = {}
            for agent in self.env.agents:
                # Get the first observation for each agent separately
                # This is a fallback method used in older PettingZoo versions
                if hasattr(self.env, 'observe'):
                    obs_dict[agent] = self.env.observe(agent)
                else:
                    # If there's no observe method, initialize with zeros
                    obs_space = self.observation_space[self.agents.index(agent)]
                    obs_dict[agent] = np.zeros(obs_space.shape, dtype=np.float32)
        elif isinstance(result, tuple) and len(result) == 2:
            # Newer PettingZoo API returns (obs_dict, info)
            obs_dict, _ = result
        else:
            # Assume result is just the obs_dict
            obs_dict = result
        
        return self._process_obs(obs_dict)
    
    def step(self, actions):
        """
        Take a step in the environment.
        
        Args:
            actions: List of actions, one per agent
            
        Returns:
            observations: List of observations, one per agent
            rewards: List of rewards, one per agent
            dones: List of done flags, one per agent
            infos: Dictionary of info objects
        """
        # The PettingZoo API requires stepping through one agent at a time
        rewards = {agent: 0.0 for agent in self.agents}
        dones = {agent: False for agent in self.agents}
        obs_dict = {}
        infos = {}
        
        for i, agent in enumerate(self.env.agents):
            if i < len(actions):  # Make sure we have an action for this agent
                # Convert one-hot encoded action to index if needed
                action = actions[i]
                if isinstance(action, np.ndarray) and len(action.shape) > 0:
                    if action.shape[0] > 1:  # It's one-hot encoded
                        action = np.argmax(action)
                    else:
                        action = int(action)
                
                try:
                    # Step the environment with this agent's action
                    self.env.step(action)
                    
                    # Collect observation, reward, and done status for this agent
                    obs_dict[agent] = self.env.observe(agent)
                    rewards[agent] = self.env.rewards[agent]
                    dones[agent] = self.env.terminations[agent] or self.env.truncations[agent]
                except Exception as e:
                    # If an error occurs, use fallback values
                    obs_space = self.observation_space[self.agents.index(agent)]
                    obs_dict[agent] = np.zeros(obs_space.shape, dtype=np.float32)
                    rewards[agent] = 0.0
                    dones[agent] = True
        
        # Process observations, rewards, and dones to match MADDPG format
        observations = self._process_obs(obs_dict)
        rewards_list = [rewards.get(agent, 0.0) for agent in self.agents]
        dones_list = [dones.get(agent, False) for agent in self.agents]
        
        return observations, rewards_list, dones_list, infos
    
    def render(self):
        """Render the environment"""
        return self.env.render() 