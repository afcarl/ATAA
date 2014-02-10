""" Helicopter simulator for the generalized hover regime version 2009 """

from helicopter.helicopter import *

# Wind indices
AMP = 0
FREQ = 1
PHASE = 2
CENTER = 3


# Simulator operates at 100Hz
DT = 0.01


def wind_wave(params):
  """ Return sinusoidal wind wave for given parameters. """
  assert all(map(lambda x: 0 <= x <= 1, params))
  maxf, freq, phase, center = params
  wave = [0.0] * 4
  wave[AMP] = (maxf - center) * 5
  wave[FREQ] = 20*math.pi * freq
  wave[PHASE] = phase * (wave[FREQ] / (freq*10)) if freq != 0 else 0.0
  wave[CENTER] = center * 5
  return wave


def wind_force(wave, time):
  """ Returns wind force at given time. """
  assert len(wave) == 4
  return wave[AMP] * math.sin(wave[FREQ] * time + wave[PHASE]) + wave[CENTER]


class Environment(Helicopter):
  """ Simulates the behavior of a helicopter in stationary flight. """
  def __init__(self, params, steps=6000):
    assert len(params) == 8
    self.wave_ns = wind_wave(params[:4])
    self.wave_ew = wind_wave(params[4:])
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
      # Compute wind in world frame
      time = self.steps * 0.1 + i * DT
      # Uncomment the next two lines to match the original 2009 rl-competition
      # domain. Note though that they introduce rounding errors.
      time = int(time * 100)
      time /= 100.0
      wind = [wind_force(self.wave_ns, time), wind_force(self.wave_ew, time), 0.0]
      # Rotate velocity- and wind vector from world to heli frame
      velocity = inverse_rotate(self.state[U:], self.q)
      wind = inverse_rotate(wind, self.q)
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

