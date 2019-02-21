"""Implementation of the NREL 5MW wind turbine torque controller.

"""

import numpy as np
import yaml

from .util import saturate


class TorqueController:
    """Time-stepping torque controller for the NREL 5MW wind turbine.

    It expects the following parameters:

    * ``rated speed``: The generator speed setpoint for the controller

    * ``rated power``: The generator power setpoint for the controller, for
      constant power mode

    * ``slip percent``: Generator slip rate, to calculate the synchronous speed

    * ``opt constant``: the k coefficient for the optimal speed control region

    * ``speed filter corner freq``: The frequency of the generator speed filter.

    * ``cut in speed``: cut in generator speed

    * ``opt min speed``: minimum generator speed for optimal control (linear
      ramp between cut in speed and this speed)

    * ``torque max``: maximum generator torque

    * ``torque rate limit``: Maximum torque rate limit (up or down)

    Optional parameters:

    * ``constant torque``: control for this constant torque above rated,
      instead of constant power.

    """

    def __init__(self, timestep, params):
        self.params = params
        self.timestep = timestep

        # Calculate maximum optimum-torque speed to achieve slope
        Qrated = self.params['rated power'] / params['rated speed']
        sync_speed = params['rated speed'] / (1 + params['slip percent']/100)
        slope25 = Qrated / (self.params['rated speed'] - sync_speed)
        kopt = params['opt constant']
        params['opt max speed'] = (
            (slope25 - np.sqrt(slope25*(slope25 - 4*kopt*sync_speed))) /
            (2 * kopt))

        # For Hywind: optionally use constant torque instead of constant power
        self.constant_torque = params.get('constant torque', None)
        assert self.constant_torque is None or self.constant_torque > 0

        # Check values
        assert params['speed filter corner freq'] > 0
        assert timestep > 0
        assert params['slip percent'] > 0
        assert params['opt constant'] > 0
        assert params['torque rate limit'] > 0
        assert (0 < params['cut in speed']
                  < params['opt min speed']
                  < params['opt max speed']
                  < params['rated speed'])
        assert (0 < (kopt * params['rated speed']**2)
                  < Qrated
                  < params['torque max'])

        self.reset()

    def reset(self):
        """Reset the controller state."""
        # Values from the previous timestep
        self.last_time = None
        self.torque_demand = None
        self.speed_filtered = None

    def _optQ(self, speed):
        return self.params['opt constant'] * speed**2

    def get_torque(self, spd, const_power):
        Vin = self.params['cut in speed']
        Vo1 = self.params['opt min speed']
        Vo2 = self.params['opt max speed']
        Vrated = self.params['rated speed']
        Qrated = self.params['rated power'] / Vrated

        if spd >= Vrated or const_power:
            # Region 3 - constant power
            if spd <= 0:
                # Needed for harmonic linearisation
                torque = self.params['torque max']
            elif self.constant_torque is not None:
                torque = self.constant_torque
            else:
                torque = self.params['rated power'] / spd
        elif spd < Vo1:
            # Region 1 to 1.5 - linear ramp from cut-in to optimal region
            torque = np.interp(spd, [Vin, Vo1], [0, self._optQ(Vo1)])
        elif spd < Vo2:
            # Region 2 - optimal control
            torque = self._optQ(spd)
        else:
            # Region 2.5 - linear ramp
            torque = np.interp(spd, [Vo2, Vrated], [self._optQ(Vo2), Qrated])

        # Limit to maximum torque
        torque = saturate(torque, 0, self.params['torque max'])

        return torque

    def initialise(self, time, measured_speed):
        """Initialise the controller.

        Args:
            time (float): current timestamp
            measured_speed (float): current measured generator speed

        """

        self.last_time = time - self.timestep
        self.speed_filtered = measured_speed

    def step(self, time, measured_speed, force_constant_power):
        """Step the controller forwards to the next timestep.

        This is the main method of the controller class.

        Args:
            time (float): the current timestamp
            measured_speed (float): current measured generator speed
            force_constant_power (bool): force constant power mode?

        """
        # First run?
        if self.last_time is None:
            self.initialise(time, measured_speed)

        # Check if enough time has elapsed
        elapsed_time = time - self.last_time
        if elapsed_time < self.timestep:
            return

        # Update filtered speed
        alpha = np.exp(-elapsed_time * self.params['speed filter corner freq'])
        self.speed_filtered = ((1 - alpha) * measured_speed +
                               alpha * self.speed_filtered)

        # Choose the desired torque & limit
        torque = self.get_torque(self.speed_filtered, force_constant_power)

        # Saturate the commanded torque using the rate limit
        if self.torque_demand is not None:
            rate = saturate((torque - self.torque_demand) / elapsed_time,
                            -self.params['torque rate limit'],
                            +self.params['torque rate limit'])
            torque = self.torque_demand + rate * elapsed_time

        self.torque_demand = torque
        self.last_time = time

    @classmethod
    def from_yaml(cls, filename):
        """Read controller params from 'controller' section of YAML file"""
        with open(filename) as f:
            config = yaml.safe_load(f)
        c = config['controller']
        return cls(c['timestep'], c['torque controller'])
