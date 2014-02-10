import mdp
from environment import Environment
from agents.mf_agent import Agent


def init_env():
  """ Initialize helicopter environment. """
  try:
    index = int(sys.argv[1])
    params = mdp.test[index]
    return Environment(params)
  except:
    print 'Usage: experiment [0-99]'
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
