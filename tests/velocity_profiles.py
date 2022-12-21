import numpy as np


def vel_gen_1(t):
    vel = {'x': 0.01, 'y': 0, 'tau': 3}
    return vel


def vel_gen_2(t):
    vx = 1e-2 *t
    vel = {'x': vx, 'y': 0, 'tau': 0}
    return vel


def vel_gen_3(t):
    if t < 0.1:
        vx = t*0.1
        tau = 0
        vy = t*0.1
    else:
        vx = 0.1*0.1
        vy = 0.1*0.1
        tau = (t-0.1)*30
    vel = {'x': vx, 'y': vy, 'tau': tau}
    return vel


def vel_gen_4(t):
    angle = np.pi * 45/180
    tau = 0
    if t < 0.1:
        vn = t
    elif t < 0.2:
        vn = 0.1 - (t - 0.1)
    elif t < 0.25:
        vn = 0
    elif t< 0.35:
        vn = 0
        tau = (t - 0.25) * 30
    elif t < 0.45:
        vn = 0
        tau = 30*0.1
    elif t < 0.75:
        vn = t - 0.45
        tau = 30 * 0.1
    elif t < 0.8:
        tau = 30 * 0.1
        vn = 0.3
    elif t < 0.9:
        tau = 30 * 0.1
        vn = 0.3 - 1.5*(t - 0.8)
    elif t < 1:
        vn = 0.15 - 1.5*(t - 0.9)
        tau = 30 * (0.1 - (t - 0.9))
    else:
        vn = 0
        tau = 0
    vx = np.cos(angle)*vn
    vy = np.sin(angle)*vn

    vel = {'x': vx, 'y': vy, 'tau': tau}
    return vel


def vel_gen_5(t):
    angle = np.pi * 45/180
    tau = 0
    if t < 0.1:
        vn = t

    elif t < 0.2:
        vn = 0.1 - (t - 0.1)
    elif t < 0.25:
        vn = 0
    elif t< 0.35:
        vn = 0
        tau = (t - 0.25) * 30
    elif t < 0.45:
        vn = 0
        tau = 30*0.1
    elif t < 0.47:
        vn = t - 0.45
        tau = 30 * 0.1
    elif t < 0.6:
        tau = 30 * 0.1
        vn = 0.02
    elif t < 0.7:
        tau = 30 * 0.1
        vn = 0.02 + (t - 0.6)
    elif t < 0.9:
        vn = 0.12 - (t-0.7)
        tau = 30 * 0.1
    elif t < 8:
        vn = -0.08
        tau = 30 * 0.1
    else:
        vn = 0
        tau = 0
    vx = np.cos(angle)*vn
    vy = np.sin(angle)*vn

    vel = {'x': vx, 'y': vy, 'tau': tau}
    return vel


def vel_gen_6(t):
    vx = np.cos(t)
    vy = np.cos(2*t)
    tau = np.sin(1.1*t)
    vel = {'x': vx, 'y': vy, 'tau': tau}
    return vel

