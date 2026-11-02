import os
import re
import glob

import numpy as np

from case_config import load_case_config
from bladeprocessor.tip_vortex_tracking import TipVortexPhaseAverage
from bladeprocessor.meridional_field import meridional_cylindrical_components, plot_meridional_field

cfg = load_case_config()
master_path = cfg.master_path
r_tip = cfg.r_tip
rpm = cfg.rpm
normalize = cfg.normalize

# Dimensionless tip speed, used as normalize_by below only if this
# case's own yaml sets normalize: true - same convention pressure_manager.py/
# skin_friction_manager.py use for Cp/Cf (raw physical units when
# normalize is false, matching cases/*.yaml's own header comment).
u_tip = rpm * 2 * np.pi / 60 * r_tip

os.makedirs(os.path.join(master_path, 'images', 'tip_vortex'), exist_ok=True)
os.makedirs(os.path.join(master_path, 'images', 'meridional_field'), exist_ok=True)


# ------------- Tip-vortex tracking: phase-locked plane averaging (see README.md) ------------- #
#
# Extraction is a cluster job (needs pf2ens - see run_conversion.sh), not
# run here - ONE-SIDED inplane_range (not the two-sided default) so each
# plane has a single search zone, spanning the FULL 360deg (one-sided
# planes don't get the opposite azimuth for free the way a two-sided
# diametral plane does), and --angle-step matched EXACTLY to this case's
# own per-frame rotation angle (~2.0011 deg/frame for 6e-5_6000rpm - see
# HANDOFF.md's reference-frame investigation - not a re-derived or
# rounded value) so the relabeling below is exact, not approximate:
#
#   sbatch run_conversion.sh fnc-meridional-sweep SMR-VR8.fnc \
#       /storage/renj3003/rotor-alone/6e-5_6000rpm/data/fnc/tip_vortex_planes/ \
#       --angle-start 0 --angle-end 358 --angle-step 2.0011 \
#       --inplane-range 0 0.13 --variables vx,vy,vz \
#       --first 0 --last 199 --nc-stats nc_stats.txt
#
# (--nc-stats matters here beyond timing metadata: it's what lets
# TipVortexPhaseAverage auto-detect the rotation direction below, from
# the resulting files' own Metadata/lrf_position_rad.)

#plane_paths = sorted(glob.glob(
#    '/storage/renj3003/rotor-alone/6e-5_6000rpm/data/fnc/tip_vortex_planes/plane_*deg.h5'
#))
#tva = TipVortexPhaseAverage(plane_paths, spacing_deg=2.0011)

# cylindrical=True (default) converts (vx, vy, vz) into (v_r, v_theta, v_z) -
# omega_rad_s is deliberately NOT passed here (see README.md, "Two things
# flagged as open" - whether pf2ens's velocity is absolute or relative to
# the LRF is not yet verified, unlike Surface_X/Y/Z-Force):
#result = tva.compute(['vx', 'vy', 'vz'], cylindrical=True)

# One age label's averaged field, on whatever radial range the planes
# actually cover (one-sided here - NOT mirrored into a full diameter,
# see README.md):
#tva.plot_age_label(
#    result, label=5, variable='v_r',
#    savepath=os.path.join(master_path, 'images/tip_vortex/age5_vr.png'),
#)

# Every age label in one pass, e.g. for an animation across "wake age":
#for label in range(tva.n_planes):
#    tva.plot_age_label(
#        result, label=label, variable='v_r',
#        savepath=os.path.join(master_path, f'images/tip_vortex/age{label}_vr.png'),
#    )


# ------------- Meridional-plane (PowerVIZ Matrix export) field plots (see README.md) ------------- #

# Separate from the h5/pf2ens tip-vortex planes above - these are
# PowerVIZ's own "Matrix" .txt exports (txt/<component>_<angle>deg.txt,
# one regular (ny, nx) grid per file, NOT the scattered x,y,z,value
# format - see meridional_field.py's module docstring), one fixed-
# azimuth meridional cut per angle actually exported for this case.

u_tip_norm = u_tip if normalize else None


angle_files = sorted(
   glob.glob(os.path.join(master_path, 'txt/x_vel_*deg.txt')),
   key=lambda p: float(re.search(r'x_vel_(\d+)deg\.txt', p).group(1)),
)
for x_file in angle_files:
   angle_deg = float(re.search(r'x_vel_(\d+)deg\.txt', x_file).group(1))
   y_file = x_file.replace('x_vel_', 'y_vel_')
   z_file = x_file.replace('x_vel_', 'z_vel_')
   X, Y, v_r, v_theta, v_z = meridional_cylindrical_components(
       x_file, y_file, z_file, angle_deg=-angle_deg, rotation_axis='y',
   )
   # (filename prefix, field, LaTeX symbol) - same color scale (vmin/vmax)
   # across all three so magnitudes are directly comparable between panels.
   for prefix, field, sym in (('Ua', v_z, r'U_a'), ('Ur', v_r, r'U_r'), ('Ut', v_theta, r'U_\theta')):
       plot_meridional_field(
           X, Y, field,
           variable_label=rf'Mean ${sym}$ [m/s]' if not normalize else rf'Mean ${sym}/U_{{tip}}$',
           r_tip=r_tip, normalize_by=u_tip_norm, vmin=-15, vmax=15,
           savepath=os.path.join(master_path, f'images/meridional_field/{prefix}_{angle_deg:g}deg.png'),
       )
