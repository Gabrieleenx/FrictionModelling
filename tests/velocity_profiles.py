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


def vel_num_cells(t):
    cor_max = 0.015

    t1 = 0.6
    t2 = 0.8
    t3 = 1.4
    t4 = 1.6
    t5 = 2.2
    t6 = 2.4
    t7 = 3.0
    t8 = 3.2
    t9 = 3.8
    t10 = 4.0
    t11 = 4.6

    if t < t1:
        vx = 0
        vy = 0
        tau = t/t1
    elif t < t2:
        vx = 0
        vy = 0
        tau = 1
    elif t < t3:
        vx = cor_max*(t-t2)/(t3-t2)
        vy = 0
        tau = 1
    elif t < t4:
        vx = cor_max
        vy = 0
        tau = 1
    elif t < t5:
        vx = cor_max * (t5 - t)/ (t5 - t4)
        vy = 0
        tau = 1
    elif t < t6:
        vx = 0
        vy = 0
        tau = 1
    elif t < t7:
        vx = 0
        vy = cor_max*(t-t6)/(t7-t6)
        tau = 1
    elif t < t8:
        vx = 0
        vy = cor_max
        tau = 1
    elif t < t9:
        vx = 0
        vy = cor_max
        tau = (t9 - t)/(t9-t8)
    elif t < t10:
        vx = 0
        vy = cor_max
        tau = 0

    elif t < t11:
        vx = 0
        vy = cor_max * (t11 - t) / (t11 - t10)
        tau = 0
    else:
        vx = 0
        vy = 0
        tau = 0

    vel = {'x': vx, 'y': vy, 'tau': tau}
    return vel