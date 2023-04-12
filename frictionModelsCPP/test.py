import frictionModelsCPP.build.FrictionModelCPPClass as cpp
import frictionModelsCPP.build.ReducedFrictionModelCPPClass as red_cpp

from tqdm import tqdm
import sys
sys.path.append('..')  # add the parent directory to the Python path
import surfaces.surfaces as surf
from frictionModels.frictionModel import FullFrictionModel
fic = cpp.FullFrictionModel()
red_fic = red_cpp.ReducedFrictionModel()

properties = {'grid_shape': (20, 20),  # number of grid elements in x any
              'grid_size': 1e-3,  # the physical size of each grid element
              'mu_c': 1,
              'mu_s': 1.3,
              'v_s': 1e-3,
              'alpha': 2,
              's0': 1e5,
              's1': 2e1,
              's2': 0.4,
              'dt': 1e-4,
              'z_ba_ratio': 0.9,
              'stability': True,
              'elasto_plastic': True,
              'steady_state': True}

def properties_to_list(prop):
    list_ = []
    for index, key in enumerate(prop):
        if key == "grid_shape":
            list_.append(prop[key][0])
            list_.append(prop[key][1])
        else:
            list_.append(prop[key])
    return list_

shape_name = "Square"
fn = 1.1
fic.init(properties_to_list(properties), shape_name, fn)
red_fic.init(properties_to_list(properties), shape_name, fn)
print("cpp package")

for i in tqdm(range(50)):
    a = fic.step([0.1, 0.1, 2.0])

print(fic.step([0.1, 0.1, 2.0]))

shape = surf.PObject(properties['grid_size'], properties['grid_shape'], surf.p_square)

planar_lugre = FullFrictionModel(properties=properties)
planar_lugre.update_p_x_y(shape)
planar_lugre.set_fn(1.1)
print("python package")
for i in tqdm(range(50)):
    a = planar_lugre.step({'x':0.1, 'y':0.1, 'tau':2.0})

print(planar_lugre.step({'x':0.1, 'y':0.1, 'tau':2.0}))
