env = gym.make("BipedalWalker-v2")

for i in range(100):
    state = env.reset()
    for i in range(100)