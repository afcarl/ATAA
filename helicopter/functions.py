import sys
import math

sys.path.append('/home/koppejan/projects/site-packages')

from eonn import eonn
from eonn.genome import Genome
from eonn.organism import spawn

from numpy import *
from numpy.linalg import *
from helicopter.helicopter import *
from helicopter.quaternion import *


class Evaluator:
  """ Policy evaluator, provides a fitness function for EONN. """
  def __init__(self, heli):
    self.heli = heli
    self.calls = 0

  def call(self, policy):
    state, reward = self.heli.reset()
    while not self.heli.terminal:
      action = policy.propagate(state, 1)
      state, error = self.heli.update(action)
      reward += error
    self.calls += 1
    return 1 / math.log(reward)


def evaluate(heli, policy):
  """ Helicopter evaluation function. """
  feval = Evaluator(heli)
  return feval.call(policy)


def evolve(heli, genome, popsize=50, epochs=100, keep=49, mutate_prob=.75, mutate_frac=.1, mutate_std=.8, mutate_repl=.25, verbose=False):
  """ Evolve a specialized policy for the given helicopter environment. """
  # Set evolutionary parameters
  eonn.keep = keep
  eonn.mutate_prob = mutate_prob
  eonn.mutate_frac = mutate_frac
  eonn.mutate_std = mutate_std
  eonn.mutate_repl = mutate_repl
  # Evolve population and return champion
  feval = Evaluator(heli)
  pool = spawn(genome, popsize)
  return max(eonn.optimize(pool, feval.call, epochs, 1, verbose))


def preprocess(flightdata):
  """ Preprocess flightdata for linear regression. """
  data = loadtxt(flightdata)
  diff = list()
  for i in xrange(len(data) - 1):
    v0, q0 = extract_vq(data[i])
    v1, q1 = extract_vq(data[i + 1])
    delta = (array(v1) - array(v0)) / 0.1 - array([0, 0, 9.81])
    delta = inverse_rotate(delta.tolist(), q0)                        # u, v & w
    delta.extend([None] * 3)                                          # x, y & z
    delta.extend((data[i + 1][P:ROLL] - data[i][P:ROLL]) / 0.1)       # p, q & r
    diff.append(delta)
  return data[:-1], array(diff)


def estimate_wind(flightdata):
  """ Estimate static wind (assume all other model parameters are known). """
  data, diff = preprocess(flightdata)
  n = len(data)
  wind0 = [((diff[i][U] / -0.18) - data[i][U]) / 5.0 for i in xrange(n)]
  wind1 = [((diff[i][V] + 0.54) / -0.43 - data[i][V]) / 5.0 for i in xrange(n)]
  return float(average(wind0)), float(average(wind1))


def estimate_params(flightdata):
  """ Estimate all model parameters (assume there's no wind). """
  data, diff = preprocess(flightdata)
  u = regress(data[:, U], diff[:, U], True)[:1] # u = [?, 0.0]
  v = regress(data[:, V], diff[:, V], True)
  w = regress(vstack([data[:, W], data[:, COLL+12]]), diff[:, W])
  p = regress(vstack([data[:, P], data[:, AILR+12]]), diff[:, P])
  q = regress(vstack([data[:, Q], data[:, ELEV+12]]), diff[:, Q])
  r = regress(vstack([data[:, R], data[:, RUDD+12]]), diff[:, R])
  return [float(x) for x in concatenate([u, v, w, p, q, r])]


def estimate_std(flightdata, params):
  """ Estimate standard deviations for the Gaussian noise variables. """
  data = loadtxt(flightdata)
  error = array([0.0] * 9)
  next_state = array([0.0] * 9) # Predicted next state
  for oa in data:
    cur_state, q, action = extract_sa(oa)
    error += (next_state - array(cur_state))**2
    next_state, q = update_state(cur_state, q, action, params)
  std_devs = sqrt(error / (len(data) - 1))
  std_devs = concatenate(array_split(std_devs, 3)[::2]) / (0.1 / 2)
  return [float(x) for x in std_devs]


def extract_vq(oa):
  """ Extract velocity and q vector in world frame from observation. """
  q = quaternion_from_orientation(oa[ROLL:AILR+12].tolist())
  v = rotate(oa[U:X].tolist(), q) # TODO!!!!
  return v, q


def extract_sa(oa):
  """ Extract true state, q and action from observation. """
  state = [0.0] * 9
  q = quaternion_from_orientation(oa[ROLL:AILR+12].tolist())
  state[U:X] = rotate(oa[U:X].tolist(), q)
  state[X:P] = rotate(oa[X:P].tolist(), q)
  state[P:ROLL] = oa[P:ROLL].tolist()
  action = oa[AILR+12:].tolist()
  return state, q, action


def update_state(state, q, action, params):
  """ Update the current state by integrating the accelerations. """
  # Saturate all the action features
  action = [min(max(f, -1.0), 1.0) for f in action]
  # Update position (x, y, z)
  state[X] += 0.1 * state[U]
  state[Y] += 0.1 * state[V]
  state[Z] += 0.1 * state[W]
  # Rotate velocity- and wind vector from world to heli frame
  velocity = inverse_rotate(state[U:], q)
  # Compute velocity delta in heli frame
  delta = [params[U_DRAG] * velocity[U],
           params[V_DRAG] * velocity[V] + params[SIDE_THRUST],
           params[W_DRAG] * velocity[W] + params[W_COLL] * action[COLL]]
  # Rotate delta vector to world frame
  delta = rotate(delta, q)
  # Update velocity (u, v, w)
  state[U] += 0.1 *  delta[0]
  state[V] += 0.1 *  delta[1]
  state[W] += 0.1 * (delta[2] + 9.81)
  # Update orientation (roll, pitch, yaw)
  q_tmp = quaternion_from_rotation([0.1 * v for v in state[P:]])
  q = multiply(q, q_tmp)
  # Compute angular velocity delta
  delta = [params[P_DRAG] * state[P] + params[P_AILR] * action[AILR],
           params[Q_DRAG] * state[Q] + params[Q_ELEV] * action[ELEV],
           params[R_DRAG] * state[R] + params[R_RUDD] * action[RUDD]]
  # Update angular rates (p, q, r)
  state[P] += 0.1 * delta[0]
  state[Q] += 0.1 * delta[1]
  state[R] += 0.1 * delta[2]
  # Return updated state and quaternion
  return state, q


def regress(x, y, constant=False):
  """ Solve ax + b = y using linear regression. """
  if constant:
    x = vstack([x, ones(len(x))]).T
  else:
    x = x.T
  return lstsq(x, y)[0]
