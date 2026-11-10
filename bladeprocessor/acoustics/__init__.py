'''
Far-field acoustics post-processing: far-field PSDs (direct LBM probes,
FW-H, iLES, experiment), acoustic-intensity/outgoing-fraction diagnostics,
Riemann-invariant wave split, and directivity. Ported from the
Acoustics_Handoff deliverable (rotaris-docs/Acoustics_Handoff) - see
HANDOFF_Acoustics.md there for the full theory and notation, and
acoustic_manager.py at the repo root for the runnable entry points.

These four submodules keep the same cross-import relationships they had
as standalone scripts (psd_three_angles imports direct_intensity;
intensity_maps and directivity import both), just made package-relative:

    direct_intensity   - mic geometry, Welch spectra, band fraction/outgoing PSD
    psd_three_angles   - shared settings (BASE_DIR, CASES, P_REF, ...), far-field
                          PSD comparison at 0/+50/-50 deg
    intensity_maps      - intensity maps, fraction spectra, near-mic test,
                          acoustic-validity checks, Riemann wave split
    directivity          - BPF/2BPF tone and band-OASPL directivity
'''
