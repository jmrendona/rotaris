import os
import glob

from case_config import load_case_config
from bladeprocessor.tip_vortex_tracking import TipVortexPhaseAverage

cfg = load_case_config()
master_path = cfg.master_path

os.makedirs(os.path.join(master_path, 'images', 'tip_vortex'), exist_ok=True)


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
