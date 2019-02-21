"""Combined controller, including both torque and pitch control.

"""

import numpy as np
import yaml

from .torque_controller import TorqueController
from .pitch_controller import PitchController


class CombinedController:
    """Controller combining both torque and pitch control.

    Args:
        torque_params (dict): passed to the :class:`TorqueController`
        pitch_params (dict): passed to the :class:`PitchController`
        torque_timestep (float): timestep for the torque controller
        pitch_timestep (float, optional): timestep for the pitch controller.
            Defaults to the same as ``torque_timestep``.
        const_power_min_pitch (float, optional).
            The minimum pitch angle to start forcing constant power mode for
            the torque controller. Default 0.

    """
    def __init__(self, torque_params, pitch_params, torque_timestep,
                 pitch_timestep=None, const_power_min_pitch=0):
        if pitch_timestep is None:
            pitch_timestep = torque_timestep
        self.const_power_min_pitch = const_power_min_pitch
        self.c_torque = TorqueController(torque_timestep, torque_params)
        self.c_pitch = PitchController(pitch_timestep, pitch_params)

    def step(self, time, measured_speed, measured_pitch):
        """Step both controllers forwards in time.

        This is the main method of the controller class.

        Args:
            time (float): the current timestamp
            measured_speed (float): current measured generator speed
            measured_pitch (float): current measured pitch angle

        """
        self.c_pitch.step(time, measured_speed, measured_pitch)
        force_constant_power = (self.c_pitch.pitch_demand >=
                                self.const_power_min_pitch)
        self.c_torque.step(time, measured_speed, force_constant_power)

    @property
    def torque_demand(self):
        """The current torque demand from the torque controller."""
        return self.c_torque.torque_demand

    @property
    def pitch_demand(self):
        """The current pitch demand from the pitch controller."""
        return self.c_pitch.pitch_demand

    @classmethod
    def from_yaml(cls, filename):
        """Read controller params from 'controller' section of YAML file"""
        with open(filename) as f:
            config = yaml.safe_load(f)
        c = config['controller']
        torque_params = c['torque controller']
        pitch_params = c['pitch controller']
        timestep = c['timestep']
        const_power_min_pitch = c['force const power above pitch']
        return cls(torque_params, pitch_params, timestep, timestep, const_power_min_pitch)
