"""Implementation of the pitch controller for the NREL 5MW wind turbine
controller.

"""

import numpy as np
import yaml

from .util import saturate


class PitchController:
    """Time-stepping pitch controller for the NREL 5MW wind turbine.

    It expects the following parameters:

    * ``proportional gain``: Proportional gain of the PI controller
    * ``integral gain``: Integral gain of the PI controller
    * ``pitch schedule doubled angle``: This is the angle at which the pitch
      controller gain is halved.
    * ``pitch angle min``: Minimum pitch angle limit
    * ``pitch angle max``: Maximum pitch angle limit
    * ``pitch rate limit``: Maximum pitch rate limit (up or down)
    * ``rated speed``: The generator speed setpoint for the controller
    * ``speed filter corner freq``: The frequency of the generator speed filter.

    """
    def __init__(self, timestep, params):
        self.params = params
        self.timestep = timestep
        self.reset()

    def reset(self):
        """Reset the controller state."""
        # Values from the previous timestep
        self.last_time = None
        self.speed_error_int = None
        self.pitch_demand = None
        self.speed_filtered = None

    def get_scheduled_gain(self, pitch):
        """Calculate the gain schedule factor."""
        GK = 1.0 / (1.0 + pitch / self.params['pitch schedule doubled angle'])
        return GK

    def initialise(self, time, measured_speed, measured_pitch):
        """Initialise the controller.

        Args:
            time (float): current timestamp
            measured_speed (float): current measured generator speed
            measured_pitch (float): current measured pitch angle

        """
        self.last_time = time - self.timestep
        self.speed_filtered = measured_speed
        self.pitch_demand = measured_pitch

        # Initialise integral speed error. This will ensure that the
        # pitch angle is unchanged if the initial speed_error is zero
        GK = self.get_scheduled_gain(measured_pitch)
        self.speed_error_int = (measured_pitch /
                                (GK * self.params['integral gain']))

    def get_pitch_demand(self, speed_error, speed_error_int, GK):
        # Compute the pitch commands associated with the proportional
        # and integral gains:
        demand_p = GK * self.params['proportional gain'] * speed_error
        demand_i = GK * self.params['integral gain'] * speed_error_int

        # Superimpose the individual commands to get the total pitch command;
        # saturate the overall command using the pitch angle limits:
        demand = saturate(demand_p + demand_i,
                          self.params['pitch angle min'],
                          self.params['pitch angle max'])

        return demand

    def step(self, time, measured_speed, measured_pitch):
        """Step the controller forwards to the next timestep.

        This is the main method of the controller class.

        Args:
            time (float): the current timestamp
            measured_speed (float): current measured generator speed
            measured_pitch (float): current measured pitch angle

        """

        # First run?
        if self.last_time is None:
            self.initialise(time, measured_speed, measured_pitch)

        # Check if enough time has elapsed
        elapsed_time = time - self.last_time
        if elapsed_time < self.timestep:
            return

        # Update filtered speed
        alpha = np.exp(-elapsed_time * self.params['speed filter corner freq'])
        self.speed_filtered = ((1 - alpha) * measured_speed +
                               alpha * self.speed_filtered)

        # Compute the current speed error and its integral
        # w.r.t. time; saturate the integral term using the pitch
        # angle limits:
        GK = self.get_scheduled_gain(self.pitch_demand)
        speed_error = self.speed_filtered - self.params['rated speed']
        self.speed_error_int += (speed_error * elapsed_time)
        self.speed_error_int = saturate(
            self.speed_error_int,
            self.params['pitch angle min'] / (GK*self.params['integral gain']),
            self.params['pitch angle max'] / (GK*self.params['integral gain']))

        # Saturate the overall commanded pitch using the pitch rate limit:
        demand = self.get_pitch_demand(speed_error, self.speed_error_int, GK)
        pitch_rate = saturate((demand - measured_pitch) / elapsed_time,
                              -self.params['pitch rate limit'],
                              +self.params['pitch rate limit'])
        self.pitch_demand = measured_pitch + pitch_rate * elapsed_time
        self.last_time = time

    @classmethod
    def from_yaml(cls, filename):
        """Read controller params from 'controller' section of YAML file"""
        with open(filename) as f:
            config = yaml.safe_load(f)
        c = config['controller']
        return cls(c['timestep'], c['pitch controller'])
