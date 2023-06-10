import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

class Animate(object):
    def __init__(self, data, interval=10):
        """

        :param data: dict()
        :param interval: int
        """
        self.fig = plt.figure(figsize=(6,6))
        self.g = -9.82
        self.m = 0.2
        plt.grid()
        self.ax = self.fig.add_subplot(111)
        #self.ax.set_aspect('equal')


        self.box_width = 0.15
        self.box_height = 0.08

        self.object = patches.Rectangle((0, 0), self.box_width, self.box_height, fc='y')

        self.circle = plt.Circle((0, 0), 0.01, color='r')
        self.arrow_mg = plt.arrow(0, 0, 0, -0.1, width=0.003)
        self.arrow_ff = plt.arrow(0, 0, 0, 0.1, width=0.003)

        self.data = data
        self.interval = interval

    def init_anim(self):
        self.ax.add_patch(self.object)
        self.ax.add_patch(self.circle)
        self.ax.add_patch(self.arrow_mg)
        self.ax.add_patch(self.arrow_ff)
        return self.object, self.circle, self.arrow_mg, self.arrow_ff,

    def animate(self, i):
        j = self.interval*i
        x_, y_ = transform_box_origin(x=self.data['p_x'][j], y=self.data['p_y'][j], theta=self.data['theta'][j],
                                    box_width_=self.box_width, box_height_=self.box_height)
        self.object.set_xy([x_, y_])
        self.object.set_angle(np.rad2deg(self.data['theta'][j]))

        # there is no way to update position on arrow objects :(
        self.arrow_mg.remove()
        self.arrow_mg = plt.arrow(self.data['p_x'][j], self.data['p_y'][j], 0,
                                  self.g * self.m / 50, width=0.003)
        self.ax.add_patch(self.arrow_mg)

        self.arrow_ff.remove()
        self.arrow_ff = plt.arrow(0, 0, self.data['f_x'][j] / 50, self.data['f_y'][j] / 50, width=0.003)
        self.ax.add_patch(self.arrow_ff)

        self.ax.set_ylim(-0.2, 0.2)
        self.ax.set_xlim(-0.2, 0.2)

        return self.object, self.circle, self.arrow_mg, self.arrow_ff,

    def run(self):
        print(type(self.data), type(self.interval))
        frames = np.floor(len(self.data['p_x'])/self.interval).astype(int)
        anim = animation.FuncAnimation(self.fig, self.animate,
                                       init_func=self.init_anim,
                                       frames=frames,
                                       interval=10,
                                       blit=True)
        plt.show()


def transform_box_origin(x, y, theta, box_width_, box_height_):
    r = np.sqrt((box_width_/2)**2 + (box_height_/2)**2)
    a = np.arctan(box_height_/box_width_)
    x_ = x - np.cos(theta + a)*r
    y_ = y - np.sin(theta + a)*r
    return x_, y_