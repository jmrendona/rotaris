from converters.forces_strip import ForcesCSVConverter
from converters.span_2_radius import SpanToRadiusConverter

forces = ForcesCSVConverter(
    file_path='path/to/forces/files',
    file_pattern='Forces-Strip-*.csv',
    dt=0.001
)

span_to_radius = SpanToRadiusConverter(
    input_path='path/to/spanwise/files',
    output_path='path/to/radius/files',
    variable_col='Variable[Force:N]',
    span_col='SpanwiseCoordinate[Length:m]',
    radius_resolution=1
)