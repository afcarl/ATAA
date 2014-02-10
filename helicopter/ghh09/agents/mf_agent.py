import sys

from genome import Genome
from organism import Organism


PREFIX = '/home/koppejan/projects/helicopter/ghh09/policies/'
POLICIES = [Genome.open(PREFIX + 'mdp%i.net' % i) for i in range(10) if i != 2]


class Agent:
  """ Model-Free agent, chooses best policy from a set of pre-trained policies. """
  def __init__(self):
    """ Initialize agent. """
    self.pool = [Organism(genome) for genome in POLICIES]
    self.episode = 0

  def start(self):
    """ Start a new episode """
    self.set_policy()
    self.episode += 1
    self.reward = 0
    self.steps = 0

  def step(self, state, reward):
    """ Choose an action based on the current state. """
    self.steps += 1
    self.reward += reward
    return self.org.policy.propagate(state, 1)

  def end(self, reward):
    """ Ends the current episode """
    self.reward += reward
    self.org.evals.append(self.reward)
    print '%i %i %.2f' % (self.episode, self.steps, self.reward)

  def set_policy(self):
    """ Set control policy based on current episode. """
    if self.episode < len(self.pool):
      self.org = self.pool[self.episode]
    else:
      self.org = min(self.pool) # Minimizing error

