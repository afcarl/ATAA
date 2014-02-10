import math

from eonn import eonn
from eonn.genome import Genome
from eonn.organism import Pool
from helicopter.helicopter import Helicopter, XcellTempest


def hover(policy):
  """ Helicopter evaluation function. """
  state, sum_error = heli.reset()
  while not heli.terminal:
    action = policy.propagate(state, 1)
    state, error = heli.update(action)
    sum_error += error
  return 1 / math.log(sum_error)


if __name__ == '__main__':
  heli = Helicopter(XcellTempest.params, XcellTempest.noise_std)
  pool = Pool.spawn(Genome.open('baseline.net'), 20)
  # Set evolutionary parameters
  eonn.keep = 15
  eonn.mutate_prob = 0.9
  eonn.mutate_frac = 0.1
  eonn.mutate_std = 0.8
  eonn.mutate_repl = 0.15
  # Evolve population
  pool = eonn.optimize(pool, hover)
  champion = max(pool)
  # Print results
  print '\nerror:', math.exp(1 / hover(champion.policy))
  print '\ngenome:\n%s' % champion.genome
