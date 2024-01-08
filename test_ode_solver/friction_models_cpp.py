import frictionModelsCPP.build.FrictionModelCPPClass as cpp
import frictionModelsCPP.build.ReducedFrictionModelCPPClass as red_cpp
import surfaces.surfaces as surf
from frictionModels.frictionModel import ReducedFrictionModel


def properties_to_list(prop):
    list_ = []
    for index, key in enumerate(prop):
        if key == "grid_shape":
            list_.append(prop[key][0])
            list_.append(prop[key][1])
        else:
            list_.append(prop[key])
    return list_

class ModelRed():
    def __init__(self, shape, p):
        self.model_fric = red_cpp.ReducedFrictionModel()
        self.model_fric.init(properties_to_list(p), shape, 1)

        self.y_init = [0.0, 0.0, 0.0]

    def f(self, t, y, vel, force):
        dz = self.model_fric.step_ode(y.tolist(), vel)
        f = self.model_fric.get_force_at_center()
        f_cop = self.model_fric.get_force_at_cop()
        force.fx = f[0]
        force.fy = f[1]
        force.tau = f[2]
        force.fx_cop = f_cop[0]
        force.fy_cop = f_cop[1]
        force.tau_cop = f_cop[2]
        return dz

    def update_surface(self, p, shape_name, update=1):
        # np.array 2d cell x cell
        p_surf = p.flatten().tolist()
        self.model_fric.update_surface(p_surf, shape_name, 1, update)

class ModelDist():
    def __init__(self, shape, p):
        self.model_fric = cpp.DistributedFrictionModel()
        self.model_fric.init(properties_to_list(p), shape, 1)
        self.y_init = [0.0] * p['grid_shape'][0] * p['grid_shape'][1] * 2

    def f(self, t, y, vel, force):
        dz = self.model_fric.step_ode(y.tolist(), vel)
        f = self.model_fric.get_force_at_center()
        f_cop = self.model_fric.get_force_at_cop()
        force.fx = f[0]
        force.fy = f[1]
        force.tau = f[2]
        force.fx_cop = f_cop[0]
        force.fy_cop = f_cop[1]
        force.tau_cop = f_cop[2]
        return dz

    def update_surface(self, p, shape_name):
        # np.array 2d cell x cell
        p_surf = p.flatten().tolist()
        self.model_fric.update_surface(p_surf, shape_name, 1)

class Force(object):
    def __init__(self):
        self.fx = 0
        self.fy = 0
        self.tau = 0
        self.fx_cop = 0
        self.fy_cop = 0
        self.tau_cop = 0

# This is a python version for the ellipse approximation without the pre-calculated limit surface


shape_set = {'Square': surf.p_square, 'Circle': surf.p_circle, 'Line': surf.p_line, 'LineGrad': surf.p_line_grad,
             'NonConvex1': surf.non_convex_1, 'NonConvex2':surf.non_convex_2, 'LineGrad2': surf.p_line_grad,
             'LineGrad3': surf.p_line_grad3, 'LineGrad4': surf.p_line_grad4}

class ModelEllipse(object):
    def __init__(self, shape, properties):
        self.properties = properties
        self.model_fric = ReducedFrictionModel(properties=properties, ls_active=False)
        shape_ = surf.PObject(properties['grid_size'], properties['grid_shape'], shape_set[shape])
        self.model_fric.update_p_x_y(shape_)
        #self.model_fric.update_pre_compute()
        self.y_init = [0.0, 0.0, 0.0]

    def f(self, t, y, vel, force):
        dy, _ = self.model_fric.ode_step(t, y, vel)

        f = self.model_fric.force_at_cop
        force.fx_cop = f['x']
        force.fy_cop = f['y']
        force.tau_cop = f['tau']
        return dy

    def update_surface(self, p_surf, name):
        shape_ = surf.PObject(self.properties['grid_size'], self.properties['grid_shape'], p_surf)
        self.model_fric.update_p_x_y(shape_)
        self.model_fric.update_pre_compute()

