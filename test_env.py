"""
Test script to debug the PettingZoo environment.
"""
from utils import make_multiagent_env

def main():
    # Create the environment
    env_name = "simple"
    print(f"Creating environment: {env_name}")
    env = make_multiagent_env(env_name)
    
    # Reset the environment
    obs = env.reset()
    print(f"Observations after reset: {obs}")
    
    # Check action space for each agent
    for agent in env.agents:
        print(f"Agent: {agent}")
        print(f"  Action space: {env.env.action_space(agent)}")
        print(f"  Sample action: {env.env.action_space(agent).sample()}")
    
    # Try stepping with valid actions
    print("\nTrying to step with valid actions...")
    actions_dict = {}
    for i, agent in enumerate(env.agents):
        action = env.env.action_space(agent).sample()
        actions_dict[agent] = action
        print(f"  Agent {agent}: action = {action}")
    
    try:
        next_obs, rewards, dones, info = env.step(actions_dict)
        print("Step successful!")
        print(f"  Next observations: {next_obs}")
        print(f"  Rewards: {rewards}")
        print(f"  Dones: {dones}")
    except Exception as e:
        print(f"Error in environment step: {e}")
    
    # Close the environment
    env.env.close()

if __name__ == "__main__":
    main() 