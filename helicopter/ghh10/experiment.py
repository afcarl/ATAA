import random

from mb_agent import Agent
from helicopter.helicopter import Helicopter as Environment
from helicopter.helicopter import XcellTempest


def init_env():
  """ Initialize helicopter environment. """
  try:
    lmbda = float(sys.argv[1])
    params = [random.gauss(v, lmbda*v) for v in XcellTempest.params]
    return Environment(params, XcellTempest.noise_std)
  except:
    print 'Usage: experiment FLOAT'
    exit()


def main():
  """ Main function, runs the experiment. """
  agent = Agent()
  env = init_env()
  for i in range(1000):
    agent.start()
    state, reward = env.reset()
    while not env.terminal:
      action = agent.step(state, reward)
      state, reward = env.update(action)
    agent.end(reward)


if __name__ == '__main__':
  main()
