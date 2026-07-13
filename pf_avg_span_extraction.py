# Direct Recorded Script
# PowerVIZ 6-2022-R4 ( 6.2.3 )
# Date: Wed Jan 21 14:30:10 2026

import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-t', type=int, help='Simulations Time Step')

args = parser.parse_args()

# ----------- Inputs to modify as needed ----------- #

chord_points = 5
hub_tip = 0.12
hub_radius = 0.016
le_loc = 0.012
te_loc = -0.012
variable = 'Skin Friction'
variable_n = 'skin-friction'
cysc = 'LRF'
aligned = 'Z-aligned'
data_path = '/scratch/renj3003/rotor-alone/6e-5_6000rpm/data/skin-friction/'

# ----------- Parameters Initialization ------------ #

chord_loc = np.linspace(le_loc,te_loc,chord_points)
#origin_loc = ((hub_tip - hub_radius)/2) + 0.016 
origin_loc = 0.075
#span = hub_tip - hub_radius
span = 0.118
height = 0.005
height_loc = np.linspace(0.0045,0.0003,chord_points)

# --------------- Variable extraction -------------- #

project1=app.currentProject
project1.timeStep=args.t
timeAnimation1=project1.timeAnimation
timeAnimation1.timestep=args.t
surfaceGraph3=project1.new(type="SurfaceGraph")
coordSystem3=project1.get(name=cysc, type="CoordSystem")
surfaceGraph3.coordinateSystem=coordSystem3
scalarPropertySet4=project1.get(name=variable, type="ScalarPropertySet")
surfaceGraph3.scalarPropertySet=scalarPropertySet4
surfaceGraph3.orientationMode=aligned
surfaceGraph3.rotate90Degrees()
surfaceGraph3.rotate90Degrees()
surfaceGraph3.rotate90Degrees()
surfaceGraph3.size=( (span, height, 1.5e-05), "m", cysc)
surfaceGraph3.upDownClassificationEnabled=False

for point_number, (c,h) in enumerate(zip(chord_loc,height_loc)):
	surfaceGraph3.position=( (origin_loc, h, c), "m", cysc)
	xYGraph5=surfaceGraph3.calculate()
	xYGraph5.exportToCSV(filename=data_path + "LRF-{}_s{:03d}.csv".format(variable_n,point_number), onlyVisible=True)
