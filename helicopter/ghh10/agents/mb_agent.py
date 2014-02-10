import sys
import glob
import random
import math
from string import letters, digits

sys.path.append('/home/koppejan/projects/helicopter')

import functions
from helicopter.helicopter import Helicopter

from eonn import *
from eonn.genome import Genome
from eonn.organism import Organism
from eonn.organism import spawn

PREFIX = '/home/koppejan/projects/helicopter/ghh10/policies/race/'
POLICIES = glob.glob(PREFIX + 'policy*.net')

SAFE_LIMITS = tuple([5.0] * 3 + [20.0] * 3 + [4 * math.pi] * 3 + [1.0] * 3)

eonn.keep = 49
eonn.mutate_prob = 0.75
eonn.mutate_frac = 0.10
eonn.mutate_std  = 0.80
eonn.mutate_repl = 0.25


def scale(value, lbx, ubx, lby , uby):
    return lby  + ((value - lbx) * (uby - lby)) / (ubx - lbx)

def normalize(value, limit):
  # Scale value to a range of [-limit, limit] to [-1, 1].
  return -1.  + ((value + limit) * 2.) / (limit * 2.)

class Agent:
  """ Model-Based agent, evolves a specialized policy using learned model. """
  def __init__(self):
    """ Initialize agent. """
    self.episode = 0
#    self.org = Organism(Genome.open(PREFIX + 'generic.net'))
    self.org = Organism(Genome.open(random.choice(POLICIES)))
    self.pool = [self.org]
    self.log = open('%s.dat' % ''.join(random.sample(letters + digits, 10)), 'w')

  def start(self):
    """ Start a new episode """
    self.episode += 1
    self.reward = 0
    self.steps = 0
    self.crashed = False

  def step(self, state, reward):
    """ Choose an action based on the current state. """
    self.steps += 1
    self.reward += reward
    if self.episode > 1 and not self.crashed:
      self.crash_control(state)
    action = self.org.policy.propagate(state, 1)
    if not self.log.closed:
      self.log.write('%+.10f ' * 12 % tuple(state))
      self.log.write('%+.10f ' *  4 % tuple(action) + '\n')
    return action

  def end(self, reward):
    """ Ends the current episode """
    self.reward += reward
    self.org.evals.append(reward)
    print '%i %i %.2f' % (self.episode, self.steps, self.reward)
    if self.episode == 1:
      self.log.close()
      self.estimate_model()
    if self.episode <= 3:
      self.evolve_policy()
    else:
      self.org = min(self.pool)

  def crash_control(self, state):
    if sum([normalize(v, SAFE_LIMITS[i]) for i, v in enumerate(state)]) > 0.15:
      self.crashed = True
      self.org.evals.append(100000)
      self.org = Organism(Genome.open(PREFIX + 'baseline.net'))

  def estimate_model(self):
    """ Learn a model from flightdata. """
    params = functions.estimate_params(self.log.name)
    noise_std = functions.estimate_std(self.log.name, params)
    self.model = Helicopter(params, noise_std, 0.1)

  def evolve_policy(self):
    """ Evolve a specialized policy using learned model. """
    pool = spawn(Genome.open(PREFIX + 'model.net'), 50)
    feval = functions.Evaluator(self.model)
    self.org = max(eonn.optimize(pool, feval.call, 2500, verbose=False))
    self.org.evals = []
    self.pool.append(self.org)


