from converters.forces_strip import ForcesCSVConverter
from converters.span_2_radius import SpanToRadiusConverter
from bladeprocessor.blades_postproc import BladePostProcessor

# ------------- Convertors ------------- #

#forces = ForcesCSVConverter(
#    file_path='/scratch/renj3003/rotor-alone/6e-5_6000rpm/data/forces_strip',
#    file_pattern='Force-Graph-*.csv',
#    dt=0.000056
#)

#forces.convert('forces_strips.h5')


#span_to_radius = SpanToRadiusConverter(
#    input_path='/scratch/renj3003/edat/cases/baseline/data/yplus',
#    output_path='/scratch/renj3003/edat/cases/baseline/data/yplus/yplus_radius.h5',
#    variable_col='Value (Up)[Dimensionless:dimensionless]',
#    span_col='Position (Up)[Length:m]',
#    chord_length=0.047,
#    radius_resolution=0.047/100
#)

#span_to_radius.convert()

# ------------- Post processing ------------- #

blade_cp = BladePostProcessor('/storage/renj3003/Rotor_alone/UdeS_Case/6e-5_6000rpm/data/cp/pstatic_radius.h5', rpm = 6000, pref = 101325, rho_ref = 1.204)

blade_cp.plot_radii(var_name = 'pressure', idx_list = [100, 200, 300, 400], mode = 'cp')

