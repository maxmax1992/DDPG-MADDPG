"""
Configuration file for MADDPG implementation.
This allows running train_maddpg.py without command-line arguments.
"""

class MaddpgConfig:
    """Default configuration for MADDPG training"""
    
    # Environment
    env = "simple_speaker_listener"  # Options: "simple", "simple_speaker_listener", "simple_spread"
    
    # Training parameters
    n_eps = 10000  # Number of episodes
    T = 25  # Maximum timesteps per episode
    
    # Rendering
    render_freq = 0  # Render frequency, 0 = no rendering
    
    # Logging
    use_writer = False  # Use tensorboard writer
    lograte = 500  # Log frequency
    
    # Algorithm settings
    train_freq = 100  # Training frequency
    buffer_length = int(1e6)  # Buffer size
    batch_size = 1024  # Batch size for training
    exp_name = "maddpg_pettingzoo"  # Experiment name
    algo = "maddpg"  # Algorithm: "maddpg" or "ddpg"
    
    # Additional options
    all_obs = False  # Use all observations for every agent
    single_q = False  # Use single Q-value network
    use_sac = False  # Use soft actor-critic 
    sac_alpha = 0.01  # SAC alpha parameter
    use_td3 = False  # Use TD3 algorithm
    use_ounoise = False  # Use OUNoise

# Create a default config object for easy access
default_config = MaddpgConfig() 