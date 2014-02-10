""" Helicopter simulator for the generalized hover regime version 2008 """

from helicopter.helicopter import *

# Simulator operates at 100Hz
DT = 0.01


def static_wind(params):
  """ Return a static wind force for the NS and EW directions. """
  assert all(map(lambda x: -1 <= x <= 1, params))
  return [params[0] * 5.0, params[1] * 5.0, 0.0]


class Environment(Helicopter):
  """ Simulates the behavior of a helicopter in stationary flight. """
  def __init__(self, params, steps=6000):
    assert len(params) == 2
    self.wind = static_wind(params)
    Helicopter.__init__(self, XcellTempest.params, XcellTempest.noise_std, DT, steps)

  def _update_state(self, action):
    """ Update state features. """
    # Saturate all the action features
    action = [min(max(f, -1.0), 1.0) for f in action]
    # Update state (inegrate at 100HZ)
    for i in range(int(0.1 / DT)):
      # Update position (x, y, z)
      self.state[X] += DT * self.state[U]
      self.state[Y] += DT * self.state[V]
      self.state[Z] += DT * self.state[W]
      # Rotate velocity- and wind vector from world to heli frame
      velocity = inverse_rotate(self.state[U:], self.q)
      wind = inverse_rotate(self.wind, self.q)
      # Compute velocity delta in heli frame
      delta = [self.params[U_DRAG] * (velocity[U] + wind[U]) + self.noise[0],
               self.params[V_DRAG] * (velocity[V] + wind[V]) + \
               self.params[SIDE_THRUST] + self.noise[1],
               self.params[W_DRAG] * velocity[W] + \
               self.params[W_COLL] * action[COLL] + self.noise[2]]
      # Rotate delta vector to world frame
      delta = rotate(delta, self.q)
      # Update velocity (u, v, w)
      self.state[U] += DT *  delta[0]
      self.state[V] += DT *  delta[1]
      self.state[W] += DT * (delta[2] + 9.81)
      # Update orientation (roll, pitch, yaw)
      q = quaternion_from_rotation([DT * v for v in self.state[P:]])
      self.q = multiply(self.q, q)
      # Compute angular velocity delta
      delta = [self.params[P_DRAG] * self.state[P] + \
               self.params[P_AILR] * action[AILR] + self.noise[3],
               self.params[Q_DRAG] * self.state[Q] + \
               self.params[Q_ELEV] * action[ELEV] + self.noise[4],
               self.params[R_DRAG] * self.state[R] + \
               self.params[R_RUDD] * action[RUDD] + self.noise[5]]
      # Update angular rates (p, q, r)
      self.state[P] += DT * delta[0]
      self.state[Q] += DT * delta[1]
      self.state[R] += DT * delta[2]

