from converters.forces_strip import ForcesCSVConverter

forces = ForcesCSVConverter(
    file_path='path/to/forces/files',
    file_pattern='Forces-Strip-*.csv',
    dt=0.001
)