from converters.forces_strip import ForcesCSVConverter
from converters.span_2_radius import SpanToRadiusConverter

forces = ForcesCSVConverter(
    file_path='/scratch/renj3003/rotor-alone/6e-5_6000rpm/data/forces_strip',
    file_pattern='Force-Graph-*.csv',
    dt=0.000056
)

forces.convert('forces_strips.h5')

#span_to_radius = SpanToRadiusConverter(
#    input_path='/scratch/renj3003/edat/cases/baseline/data/yplus',
#    output_path='/scratch/renj3003/edat/cases/baseline/data/yplus/yplus_radius.h5',
#    variable_col='Value (Up)[Dimensionless:dimensionless]',
#    span_col='Position (Up)[Length:m]',
#    chord_length=0.047,
#    radius_resolution=0.047/100
#)

#span_to_radius.convert()
