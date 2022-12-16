import numpy as np
from friction import PlanarFriction
from friction_simple import PlanarFrictionReduced
import matplotlib.pyplot as plt
import time

class CustomHashList(object):
    def __init__(self, num_segments):
        self.num_segments = num_segments
        self.scl = 2*self.num_segments/np.pi
        self.list = [0]*self.num_segments
        self.vel_scale = 0.01

    def get_closest_samples(self, vel, omega):
        a = np.arctan2(abs(vel/self.vel_scale), abs(omega)) * self.scl
        idx = int(a)
        if idx == self.num_segments:
            return self.list[idx-1][1], self.list[idx-1][1], 1
        return self.list[idx][0], self.list[idx][1], a

    def get_interpolation(self, vel, omega):
        v1, v2, a = self.get_closest_samples(vel, omega)
        r = a - int(a)
        return r * v2 + (1 - r) * v1

    def get_ratio_pairs(self, idx):
        r1 = np.tan(idx/self.scl)
        r2 = np.tan((idx+1)/self.scl)
        return r1, r2

    def add_to_list(self, idx, v1, v2):
        self.list[idx] = [v1, v2]

    def initialize(self, friction_model, p_x_y):
        f1 = friction_model.steady_state(vel_vec={'x': 1, 'y': 0, 'tau': 0}, p_x_y=p_x_y)
        f2 = friction_model.steady_state(vel_vec={'x': 0, 'y': 0, 'tau': 1}, p_x_y=p_x_y)
        f_max = np.linalg.norm([f1['x'], f1['y']])
        tau_max = abs(f2['tau'])

        for i in range(self.num_segments):
            r1, r2 = self.get_ratio_pairs(i)
            w1 = np.sqrt(1/(1 + r1**2))
            v1 = self.vel_scale * np.sqrt(1 - w1**2)
            f = friction_model.steady_state(vel_vec={'x': v1, 'y': 0, 'tau': w1}, p_x_y=p_x_y)
            f1 = np.array([np.linalg.norm([f['x'], f['y']])/f_max, abs(f['tau'])/tau_max])

            w2 = np.sqrt(1 / (1 + r2 ** 2))
            v2 = self.vel_scale * np.sqrt(1 - w2 ** 2)
            f = friction_model.steady_state(vel_vec={'x': v2, 'y': 0, 'tau': w2}, p_x_y=p_x_y)
            f2 = np.array([np.linalg.norm([f['x'], f['y']]) / f_max, abs(f['tau']) / tau_max])
            self.add_to_list(i, f1, f2)