import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from mnist_trainer import MNIST_trainer_wrapper

class CustomOptimizationEnv():

    # create elements of the world
    def reset(self):
        
        world = MNIST_trainer_wrapper()
        # make initial conditions
        # self.reset_world(world)
        return world.
        
    # create initial conditions of the world
    def step(self, action):
        pass

    def reward(self, agent, world):
        return world.last_reward

    def observation(self, agent, world):
        pass
        # raise NotImplementedError

# from mnist_trainer import MNIST_trainer_wrapper

# trainer = MNIST_trainer_wrapper()
# trainer.set_model(0.5, 0.5, 0.03)
# print(trainer.train_for_epochs())
# trainer.set_model(0.5, 0.5, 10)
# print(trainer.train_for_epochs())

# https://github.com/openai/multiagent-particle-envs/blob/master/make_env.py
def make_multiagent_env(scenario_name="", scenario=None, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.
    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)
    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world,
                            scenario.reward, scenario.observation, scenario.
                            benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world,
                            scenario.reward, scenario.observation)
    return env

env = make_multiagent_env(scenario=CustomOptimizationScenario())
env.reset()