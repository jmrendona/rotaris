"""Acoustic intensity tools for the direct (LBM probe) signals.

The probe files store pressure and the three velocity components, so the net
flow of acoustic energy (active intensity) can be computed at each mic:

    I(f) = Re{ S_pu(f) }          cross-spectral density of p' and u' (vector)

Two quantities are derived from it:

    fraction       F = rho c * I_r / S_pp      I_r = I . r_hat, r_hat = rotor -> mic
    outgoing PSD   F * S_pp

For a single sound wave travelling straight out from the rotor, p' = rho c u_r'
and F = 1: all the pressure seen by the mic is energy flowing outward. Sound
arriving from many directions (reflections, a diffuse field) carries energy
in all directions; the flows cancel, F tends to 0 while the pressure stays
large. If the field is an outgoing wave plus such a diffuse part, F is the
share of S_pp that belongs to the outgoing wave. 10*log10(1/F) is the
pressure-intensity index of sound-intensity measurement (Fahy, Sound
Intensity; ISO 9614).

The direct records are short (a few Welch segments), so F per frequency bin is
very noisy. F is therefore estimated per fractional-octave band (ratio of band
sums) and applied to the narrowband PSD. Bands where F <= 0 cannot be
corrected and are left out.

Coordinates: rotor centre at the origin, y along the rotor axis (thrust),
mics at 1.62 m. The direct-file rings were checked against the measured BPF
intensity direction (a clean outgoing wave at BPF).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as sg

R_MIC = 1.62                       # m
GAMMA = 1.4
NEAR_MIC = (0.0, -0.25, 0.25)      # m, near_mic.txt
FRACTION_BANDS_PER_OCTAVE = 3


def mic_position(path):
    """Mic coordinates (m) from the folder and file name of a direct probe."""
    path = Path(path)
    if path.name == 'near_mic.txt':
        return np.array(NEAR_MIC)
    phi, theta = (np.radians(float(v)) for v in
                  re.fullmatch(r'mic_(\d+)_(\d+)\.txt', path.name).groups())
    if path.parent.name == '2024':
        # 2024 mic_90_theta: ring in the rotor plane, theta from +z towards +x
        return R_MIC * np.array([np.sin(theta), 0.0, np.cos(theta)])
    # 2025 / 2025-T: meridian of the y-z plane rotated by phi about z
    # (all rings meet at theta = 90, the point on +z)
    return R_MIC * np.array([np.cos(theta) * np.sin(phi),
                             np.cos(theta) * np.cos(phi),
                             np.sin(theta)])


def unit(v):
    return v / np.linalg.norm(v)


def read_probe(path, time_min):
    """Pressure and velocity fluctuations after time_min, plus fs, untrimmed length and rho c."""
    data = pd.read_csv(path, sep=r'\s+')
    time = data['#time'].values
    keep = time > time_min
    p = data['static_pressure'].values[keep]
    u = np.stack([data[c].values[keep] for c in ('x_velocity', 'y_velocity', 'z_velocity')])
    rho = data['density'].values[keep].mean()
    rho_c = rho * np.sqrt(GAMMA * p.mean() / rho)
    return (p - p.mean(), u - u.mean(axis=1, keepdims=True),
            1.0 / (time[1] - time[0]), time.size, rho_c)


def spectra(p, u, fs, nperseg, nfft=None):
    """Welch S_pp and active intensity spectrum I (3 x nf)."""
    kw = dict(fs=fs, window='hann', nperseg=nperseg, nfft=nfft, scaling='density')
    f, s_pp = sg.welch(p, **kw)
    intensity = np.stack([np.real(sg.csd(p, u_i, **kw)[1]) for u_i in u])
    return f, s_pp, intensity


def probe_spectra(path, time_min, n_chunk, nfft_fn):
    """Spectra of a direct probe with the same segments as the direct PSD.

    Returns f, S_pp, I (3 x nf), r_hat, rho c.
    """
    p, u, fs, length, rho_c = read_probe(path, time_min)
    nperseg = int(length / n_chunk)
    f, s_pp, intensity = spectra(p, u, fs, nperseg, nfft_fn(nperseg))
    return f, s_pp, intensity, unit(mic_position(path)), rho_c


def band_edges(f_min=100.0, f_max=20000.0, per_octave=3):
    """Fractional-octave band centres, lower and upper edges (base 2)."""
    n = np.arange(np.floor(per_octave * np.log2(f_min / 1000)),
                  np.ceil(per_octave * np.log2(f_max / 1000)) + 1)
    centres = 1000 * 2.0 ** (n / per_octave)
    half = 2.0 ** (1 / (2 * per_octave))
    return centres, centres / half, centres * half


def band_sums(f, values, lows, highs):
    """Sum values (... x nf) over each band; NaN for bands without bins."""
    out = []
    for lo, hi in zip(lows, highs):
        m = (f >= lo) & (f < hi)
        out.append(values[..., m].sum(axis=-1) if m.any()
                   else np.full(values.shape[:-1], np.nan))
    return np.stack(out, axis=-1)


def band_fraction(f, s_pp, intensity, r_hat, rho_c, per_octave=FRACTION_BANDS_PER_OCTAVE):
    """Band centres, edges and fraction F = rho c * sum(I_r) / sum(S_pp) per band."""
    centres, lows, highs = band_edges(per_octave=per_octave)
    fraction = rho_c * band_sums(f, r_hat @ intensity, lows, highs) / band_sums(f, s_pp, lows, highs)
    return centres, lows, highs, fraction


def outgoing_psd(f, s_pp, intensity, r_hat, rho_c, per_octave=FRACTION_BANDS_PER_OCTAVE):
    """Narrowband S_pp scaled by the band fraction; NaN where F <= 0 or undefined."""
    _, lows, highs, fraction = band_fraction(f, s_pp, intensity, r_hat, rho_c, per_octave)
    factor = np.full(f.shape, np.nan)
    for lo, hi, value in zip(lows, highs, fraction):
        factor[(f >= lo) & (f < hi)] = value
    return np.where(factor > 0, s_pp * factor, np.nan)
