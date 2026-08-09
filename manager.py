import numpy as np
from converters.forces_strip import ForcesCSVConverter
from converters.span_2_radius import SpanConverter
from bladeprocessor.blades_postproc import BladePostProcessor
from bladeprocessor.surface_field import SurfaceField, SurfaceFieldComparator
from bladeprocessor.friction_lines import FrictionLines

# ------------- Convertors ------------- #

#forces = ForcesCSVConverter(
#    file_path='/scratch/renj3003/rotor-alone/6e-5_6000rpm/data/forces_strip',
#    file_pattern='Force-Graph-*.csv',
#    dt=0.000056
#)

#forces.convert('forces_strips.h5')

# timesteps = np.arange(0,46)

# for ts in timesteps:
# 	span_to_radius = SpanConverter(
#     	input_path=f'/scratch/renj3003/rotor-alone/6e-5_6000rpm-transitional/data/friction_lines/x-force/t{ts:02d}',
#     	output_path=f'/scratch/renj3003/rotor-alone/6e-5_6000rpm-transitional/data/friction_lines/x-force/t{ts:02d}/{ts:02d}t_x-force_radius.h5',
#     	variable_col='Value (Up)[ForcePerArea:newton/m^2]',
#     	span_col='Position (Up)[Length:m]',
#     	chord_length=0.025,
#     	resolution=0.025/100,
#         coordinate_system='cartesian',
#     	surface_split=False
# 	)

# 	span_to_radius.convert()

# ------------- Post processing ------------- #

#blade_cp = BladePostProcessor('/scratch/renj3003/rotor-alone/15e-6_6000rpm/data/cp/pstat_radius.h5', rpm = 6000, pref = 101325, rho_ref = 1.204)

#blade_cp.plot_radii(var_name = 'pressure', idx_list = [100, 200, 300, 400], mode = 'cp')

case_2025 = SurfaceField(
	"/storage/renj3003/rotor-alone/UdeS_Case/6e-5_6000rpm/data/cp/pstatic_cartesian.h5",
	var_name = 'pstatic',
	r_tip = 0.125,
	c_ref = 0.025
)

case_2025T = SurfaceField(
	"/storage/renj3003/rotor-alone/UdeS_Case/6e-5_6000rpm-transitional/data/cp/pstatic_cartesian.h5",
	var_name = 'pstatic',
	r_tip = 0.125
)

case_2026 = SurfaceField(
	"/storage/renj3003/rotor-alone/UdeS_Case/15e-6_6000rpm/data/cp/pstatic_cartesian.h5",
	var_name = 'pstatic',
	r_tip = 0.125
)

case_2025.plot_contour(normalize=False, cbar_label='Static Pressure [Pa]', levels=np.linspace(98000, 101800, 100),savepath='/storage/renj3003/rotor-alone/UdeS_Case/6e-5_6000rpm/images/pstatic/avg_pstatic.png')

comparator = SurfaceFieldComparator({'2025': case_2025, '2025-T': case_2025T, '2026': case_2026})

comparator.plot_cases(cbar_label='Static Pressure [Pa]', levels=np.linspace(98000, 101800, 100), savepath='/storage/renj3003/rotor-alone/UdeS_Case/Comparison/images/cp/avg_pstatic_comparison.png')

# ------------- Wall shear / friction lines ------------- #
#
# Input: a SNCReader.to_h5(..., surface_split=True) file - the forces
# branch, NEVER the pf2ens/pressure one (see README.md, "Splitting into
# upper/lower surface"). rho_ref/rpm set the LOCAL Cf normalization
# (q_ref = 0.5*rho_ref*(omega*r)^2, matching BladePostProcessor.compute_cf()
# above - see README.md's "Equations" section for the full derivation).

#fl = FrictionLines(
#    '/storage/renj3003/rotor-alone/6e-5_6000rpm/data/forces/forces_rotor.h5',
#    r_tip=0.125,
#    rho_ref=1.22523,
#    rpm=6000,
#)

# Dimensional wall shear vector (tau = F - (F.n)n), no rho_ref/rpm needed:
#tau = fl.wall_shear(surface='Upper', frame=None)  # frame=None -> average over every frame in the file

# Cf magnitude and signed chordwise/spanwise components, one frame or the average:
#cf_mag = fl.cf(surface='Upper', frame=None, component=None)
#cf_chordwise_frame0 = fl.cf(surface='Upper', frame=0, component='chordwise')

# Cf vs local x/c at several radii, one plot per call - instantaneous and average:
#fl.plot_cf_radii(
#    radii=[0.045, 0.072, 0.100, 0.117, 0.122],
#    surface='Upper', frame=None, component=None,
#    savepath='/storage/renj3003/rotor-alone/6e-5_6000rpm/images/cf/cf_radii_avg.png',
#)
#fl.plot_cf_radii(
#    radii=[0.045, 0.072, 0.100, 0.117, 0.122],
#    surface='Upper', frame=0, component='chordwise',
#    savepath='/storage/renj3003/rotor-alone/6e-5_6000rpm/images/cf/cf_radii_chordwise_frame0.png',
#)

# Friction lines (Upper+Lower stacked by default) - span_min isolates one
# blade half on a two-bladed rotor centered at span=0 (see the method's
# docstring - there's no reliable automatic hub cutoff, pass what's right
# for this case's mesh):
#fl.friction_lines(
#    frame=None, span_min=0.03,
#    savepath='/storage/renj3003/rotor-alone/6e-5_6000rpm/images/cf/friction_lines_avg.png',
#)