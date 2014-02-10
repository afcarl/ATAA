import sys
import random

sys.path.append('/home/koppejan/projects/helicopter')

import functions
from string import letters, digits
from eonn.genome import Genome
from eonn.organism import Organism
from helicopter import ghh, model
from helicopter.quaternion import quaternion_from_orientation

TOKENS = letters + digits
PREFIX = '/home/koppejan/helicopter/helicopter/ghh09/policies/'
POLICIES = [Genome.open(PREFIX + 'mdp%i.net' % i) for i in (3, 6)]


class Agent:
  """ Hybrid agent, a combination of model-free and model-based learning. """
  def __init__(self):
    """ Initialize agent. """
    self.episode = 0
    self.pool = [Organism(genome) for genome in POLICIES]
    self.backup = Organism(Genome.open(PREFIX + 'generic.net'))
    self.log = open('log_%s.txt' % ''.join(random.sample(TOKENS, 10)), 'w')

  def start(self):
    """ Start a new episode """
    self.set_policy()
    self.episode += 1
    self.reward = 0
    self.steps = 0
    self.flag = ''

  def step(self, state, reward):
    """ Choose an action based on the current state. """
    self.steps += 1
    self.reward += reward
    self.crash_control(state)
    action = self.org.policy.propagate(state, 1)
    if not self.log.closed:
      self.log.write('%+.10f ' * 12 % tuple(state))
      self.log.write('%+.10f ' *  4 % tuple(action) + '\n')
    return action

  def end(self, reward):
    """ Ends the current episode """
    self.reward += reward
    self.org.evals.append(self.reward) # TODO
    print '%i %i %.2f %s' % (self.episode, self.steps, self.reward, self.flag)
    if self.episode == 1:
      self.log.close()
      self.evolve_policy(3)

  def set_policy(self):
    """ Set control policy based on current episode. """
    if self.episode < len(self.pool):
      self.org = self.pool[self.episode]
    else:
      self.org = min(self.pool) # Minimizing error

  def evolve_policy(self, n=1):
    """ Learn a model from flightdata and evolve specialized policies. """
    params = model.estimate_params(self.log.name)
    noise_std = model.estimate_std(self.log.name, params)
    heli = ghh.Helicopter(params, noise_std, 0.1)
    genome = Genome.open(PREFIX + 'baseline.net')
    for i in range(n):
      champion = functions.evolve(heli, genome, epochs=500)
      champion.evals = list()
      self.pool.append(champion)

  def crash_control(self, state):
    """ Check for dangerous states. """
    qw = quaternion_from_orientation(state[9:])[-1]
    if qw < 0.99 and any([abs(v) > 0.5 for v in state[:9]]):
      self.org.evals.append(1000000)
      self.org = self.backup
      self.flag = '*'
