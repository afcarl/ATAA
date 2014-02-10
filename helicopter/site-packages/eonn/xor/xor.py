from eonn import eonn
from eonn.genome import Genome
from eonn.organism import Pool


def xor(policy, verbose=False):
  """ XOR evaluation function. """
  err = 0.0
  input = [(i, j) for i in range(2) for j in range(2)]
  for i in input:
    output = policy.propagate(i, 1);
    err += (output[0] - (i[0] ^ i[1]))**2
    if verbose:
      print i, '-> %.4f' % output[0]
  return 1.0 / err


if __name__ == '__main__':
  pool = Pool.spawn(Genome.open('xor.net'), 30)
  # Set evolutionary parameters
  eonn.keep = 1
  eonn.mutate_prob = 0.9
  eonn.mutate_frac = 0.25
  eonn.mutate_std = 0.8
  eonn.mutate_repl = 0.2
  # Evolve population
  pool = eonn.optimize(pool, xor)
  champion = max(pool)
  # Print results
  print '\noutput:'
  xor(champion.policy, True)
  print '\ngenome:\n%s' % champion.genome
