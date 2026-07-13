import numpy as np
from converters.forces_strip import ForcesCSVConverter
from converters.span_2_radius import SpanConverter
from bladeprocessor.blades_postproc import BladePostProcessor

# ------------- Convertors ------------- #

#forces = ForcesCSVConverter(
#    file_path='/scratch/renj3003/rotor-alone/6e-5_6000rpm/data/forces_strip',
#    file_pattern='Force-Graph-*.csv',
#    dt=0.000056
#)

#forces.convert('forces_strips.h5')

timesteps = np.arange(0,46)

for ts in timesteps:
	span_to_radius = SpanConverter(
    	input_path=f'/scratch/renj3003/rotor-alone/6e-5_6000rpm-transitional/data/friction_lines/x-force/t{ts:02d}',
    	output_path=f'/scratch/renj3003/rotor-alone/6e-5_6000rpm-transitional/data/friction_lines/x-force/t{ts:02d}/{ts:02d}t_x-force_radius.h5',
    	variable_col='Value (Up)[ForcePerArea:newton/m^2]',
    	span_col='Position (Up)[Length:m]',
    	chord_length=0.025,
    	resolution=0.025/100,
        coordinate_system='cartesian',
    	surface_split=False
	)

	span_to_radius.convert()

# ------------- Post processing ------------- #

#blade_cp = BladePostProcessor('/scratch/renj3003/rotor-alone/15e-6_6000rpm/data/cp/pstat_radius.h5', rpm = 6000, pref = 101325, rho_ref = 1.204)

#blade_cp.plot_radii(var_name = 'pressure', idx_list = [100, 200, 300, 400], mode = 'cp')

