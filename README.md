# Planar LuGre friction modeling 

This repository is the code developed for the paper "Planar Friction Modelling with LuGre Dynamics and Limit Surfaces", 
which has been accepted to IEEE Transactions on Robotics (T-RO). The pre-print version of the paper is currently available in arXiv at:
http://arxiv.org/abs/2308.01123
. If you use this work, please cite the paper (proper citations will be added once the paper is published). 

## Dependencies 
### Python
```
python 3.10
Numpy 
tqdm
Matplotlib
seaborn
dash and plotly (optional, for plotly vizualisation)
```

### C++
Optional, if you want to recompile c++ code.
```
pybind11
```

## Execute

To run the friction models run the main.py file. In the main.py file you can
to uncomment the script or test you want to run. 

## Structure
### Friction models
The friction models are implemented both in python and as a python package in c++. The python version is found under 
frictionModels and the cpp version is found under frictionModelsCPP.

### Tests
The tests used for the paper are found under tests. They are executed by running the main.py file. 

In terminal 
```
python3 main.py
```

### Visualisation 
By uncommenting the plotly_viz.py script in main.py the visualization script will run in a browser and has slider for
changing parameters. 

### surfaces
In the surfaces folder the contact surfaces are defined. 




