import sys
import random

sys.path.append('/home/koppejan/projects/helicopter')

import functions
from string import letters, digits
from eonn.genome import Genome
from eonn.organism import Organism
from helicopter import ghh, model

TOKENS = letters + digits
PREFIX = '/home/koppejan/projects/helicopter/ghh09/policies/'

class Agent:
  """ Model-Based agent, evolves a specialed policy using learned model. """
  def __init__(self):
    """ Initialize agent. """
    self.episode = 0
    self.org = Organism(Genome.open(PREFIX + 'generic.net'))
    self.log = open('log_%s.txt' % ''.join(random.sample(TOKENS, 10)), 'w')

  def start(self):
    """ Start a new episode """
    self.episode += 1
    self.reward = 0
    self.steps = 0

  def step(self, state, reward):
    """ Choose an action based on the current state. """
    self.steps += 1
    self.reward += reward
    action = self.org.policy.propagate(state, 1)
    if not self.log.closed:
      self.log.write('%+.10f ' * 12 % tuple(state))
      self.log.write('%+.10f ' *  4 % tuple(action) + '\n')
    return action

  def end(self, reward):
    """ Ends the current episode """
    self.reward += reward
    self.org.evals.append(self.reward)
    print '%i %i %.2f' % (self.episode, self.steps, self.reward)
    if self.episode == 1:
      self.log.close()
      self.evolve_policy()

  def evolve_policy(self):
    """ Learn a model from flightdata and evolve specialized policies. """
    params = model.estimate_params(self.log.name)
    noise_std = model.estimate_std(self.log.name, params)
    heli = ghh.Helicopter(params, noise_std, 0.1)
    genome = Genome.open(PREFIX + 'baseline.net')
    self.org = functions.evolve(heli, genome, epochs=500)
