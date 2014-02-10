import sys
sys.path.append('/home/koppejan/projects/site-packages/')

import math
import random
from helicopter.helicopter import *
from eonn import eonn
from eonn import organism
from eonn.genome import *
import stats
from numpy import *
import random
from string import letters, digits

PROTOTYPE = '/home/koppejan/projects/helicopter/ghh10/baseline.net'

def hover(policy):
  """ Helicopter evaluation function. """
  env = init_env(0.3)
  state, sum_error = env.reset()
  while not env.terminal:
    action = policy.propagate(state, 1)
    state, error = env.update(action)
    sum_error += error
  return 1 / math.log(sum_error)


def init_env(lmbda):
  """ Generates a random helicopter environment. """
  params = [random.gauss(v, lmbda*v) for v in XcellTempest.params]
  noise_std = XcellTempest.noise_std
  return Helicopter(params, noise_std)


def optimize(pool, fselect, feval, maxevals):
  """ Evolve the population for at most 'maxevals' evaluations. """
  champion = None
  while maxevals > 0:
    parents, evals = fselect(pool, feval)
    champion = max(parents)
    maxevals -= evals
    pool = epoch(parents, len(pool))
  return pool, champion


def epoch(pool, size):
  offspring = []
  for i in range(size):
    mom, dad = random.sample(pool, 2)
    offspring.append(eonn.reproduce(mom, dad))
  return offspring


def evaluate(pool, feval, n=1):
  for org in pool:
    for i in range(n):
      org.evals.append(feval(org.policy))


def set_bounds(pool, alpha=0.5):
  """ Set lower- and upper bound on fitness with confidence level alpha. """
  for org in pool:
    org.lb, org.ub = cint(org.evals, alpha)


def cint(evals, alpha=0.5):
  """ Compute confidence interval around the mean for the given data """
  n = len(evals)
  m, se = mean(evals), std(evals)/sqrt(n-1)
  c = stats.cached_tinv(1-alpha, n-1) * se
  return m-c, m+c


def fixed_resampling(pool, feval, mu=10, k=10):
  """ Fixed resampling, each individual is evaluated k times. """
  evaluate(pool, feval, k)
  return sorted(pool, reverse=True)[:mu], len(pool)*k


def race_resampling(pool, feval, mu=10, alpha=0.5, k=10):
  """ Selection races, evaluate each individual at most k times. """
  t = 1
  U, S, D = list(pool), [], [] # Undecided, selected & discarded
  evaluate(U, feval)
  n = evals = len(U)
  while len(S) < mu and t < k:
    t += 1
    evaluate(U, feval)
    set_bounds(U, alpha)
    evals += len(U)
    for i, x in enumerate(U):
      if sum([x.lb > y.ub for y in U]) >= n - mu - len(D):
        S.append(U.pop(i))
      elif sum([x.ub < y.lb for y in U]) >= mu - len(S):
        D.append(U.pop(i))
  if len(S) < mu:
    S.extend(sorted(U, reverse=True)[:mu - len(S)])
  return S, evals


def main():
  output = open('log_%s.txt' % ''.join(random.sample(letters+digits, 10)), 'w')
  pool = organism.spawn(Genome.open(PROTOTYPE), 30)
  for i in range(25):
    pool, champion = optimize(pool, race_resampling, hover, 20000)
    output.write('%i %.3f\n' % ((i+1) * 20000, champion.fitness))
    output.flush()
  output.close()



if __name__ == '__main__':
  main()
