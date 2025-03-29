"""
Test script to debug the PettingZoo environment directly.
"""
from pettingzoo.mpe import simple_v3

def main():
    # Create the environment
    env = simple_v3.env()
    env.reset()
    
    print(f"Agents: {env.agents}")
    print(f"Possible agents: {env.possible_agents}")
    
    # Check action space for each agent
    for agent in env.agents:
        print(f"Agent: {agent}")
        print(f"  Action space: {env.action_space(agent)}")
        print(f"  Sample action: {env.action_space(agent).sample()}")
    
    # Try stepping with valid actions
    print("\nTrying to step with valid actions...")
    
    for agent in env.agents:
        # Get a valid action
        action = env.action_space(agent).sample()
        print(f"  Agent {agent}: action = {action}")
        
        try:
            env.step(action)  # PettingZoo expects actions to be passed one at a time
            print(f"  Step successful for agent {agent}")
        except Exception as e:
            print(f"  Error in environment step for agent {agent}: {e}")
    
    print("\nCollecting all observations, rewards, etc.")
    for agent in env.agents:
        print(f"  Agent {agent}:")
        print(f"    Observation: {env.observe(agent)}")
        print(f"    Reward: {env.rewards[agent]}")
        print(f"    Terminated: {env.terminations[agent]}")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main() 