'''
Cross-SIMULATION comparison plots for StripForces results - overlay
several cases (e.g. different mesh resolutions/models, such as this
project's own '2025', '2025-T', '2026', '2026-DNS' thesis comparison) on
ONE set of axes, instead of one panel per case.

Distinct from StripForces.plot_time_trace()/plot_vs_angle(), which
overlay multiple STRIPS of a SINGLE case (one panel per case would be
needed to compare several cases with those), and distinct from
bladeprocessor.surface_field.SurfaceFieldComparator, which compares 2D
contour FIELDS across cases, not strip time-series/phase-locked data.

Motivating case (see manager_comparison.py): a thesis results chapter
comparing 4 simulation variants of the same isolated hovering rotor,
where a naive N_cases x N_components grid (one panel per case, one line
per strip within each) either buries the actual comparison of interest
(how does a given force component differ BETWEEN cases) inside a dozen
small, separately-legended panels, or, if strip count and case count are
both kept, becomes unreadable at the physical size a thesis page allows.

Design: color encodes CASE (cmap-sampled, same convention as
StripForces.plot_time_trace()/plot_vs_angle() uses for strip - see
_strip_colors(), reused here for case instead); LINESTYLE encodes STRIP
(cycling through a fixed list of dash patterns - see _LINESTYLES).
Distinguishing cases is the actual point of every plot in this module,
so case gets color, the stronger of the two channels for telling several
things apart on noisy, heavily-overlapping traces - confirmed, on a real
case, that case told apart only by linestyle was hard to read even with
_LINESTYLES' own large, noise-robust dash patterns. Strip-to-strip
magnitude, the OTHER thing every plot here shows, is instead carried by
each line's position (radius on the polar panels, vertical offset on the
Cartesian ones), not by its color or style, so it reads clearly either
way. Together these let every (case, strip) combination be told apart on
ONE panel per force component (3 panels total, not one per case), with a
SINGLE shared legend explaining both encodings for the whole figure -
built once, not repeated (or, worse, omitted) on every component's own
panel.

Print-size handling: figures are generated directly at their FINAL
physical size (inches, `total_width_in`/`total_height_in`), with LOCALLY
overridden (via `plt.rc_context`, not the project-wide `plt.rcParams`)
tick/label/legend font sizes - not the much larger font sizes
StripForces' own single-panel methods use, which would be badly
oversized once three panels share the same physical row.

The EXACT condition for a chosen `tick_fontsize`/`label_fontsize`/
`legend_fontsize` (points) to come out at that same size on the printed
page: the figure must be included with
`\\includegraphics[width=\\linewidth]{...}` (or whatever fixed width is
used) AND `total_width_in` must equal that same width converted to
inches (LaTeX `\\linewidth`/`\\textwidth` are usually reported in `pt`;
divide by 72.27 to get inches). If those two widths match, no further
scaling happens at inclusion time, and a `tick_fontsize=14` really is 14
pt on the page - the current default here, chosen so labels read at
comfortably above the thesis body text's own likely 11-12 pt even after
accounting for print rounding, per this project's own printed-figure
requirement. If they DON'T match (e.g. `total_width_in` is left at this
module's 6.3 in default but the actual document's `\\linewidth` is
different), `\\includegraphics` silently rescales the whole figure,
including its text, by that same mismatched ratio - check the actual
`\\linewidth` value in the thesis document and update `total_width_in`
to match before treating the on-page result as final. Saving as a
vector format (`savepath` ending in `.pdf`, not `.png`) is recommended
for this reason: vector text stays exactly the specified point size and
perfectly crisp at whatever the final rendered size ends up being,
whereas a raster (PNG) locks in a fixed pixel grid at the DPI it was
exported at and can look soft if LaTeX ends up scaling it at all.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Solid, then three custom dash patterns with deliberately LARGE segment
# lengths - in that order, cycling if there are ever more than 4 cases.
# Matplotlib's built-in '--'/':'/'-.' patterns use small, fine segments
# calibrated for a big, single-panel, low-density figure; at this
# module's compressed multi-panel print size, and especially against a
# NOISY, busy, heavily-overlapping raw time trace (as opposed to a
# smooth phase-locked curve), those fine patterns visually blur into
# what looks like a solid line - confirmed by direct visual inspection
# during development (fig56_compare_test.png in the test images folder
# showed exactly this: linestyle was legible on the smooth polar/phase-
# locked comparison but not on the noisy raw time trace, with the
# default matplotlib dash patterns). These larger, explicit (on, off[,
# on, off...]) tuples (in points) stay visually distinct even under
# heavy noise and line overlap. A 5th/6th entry could be added here if
# needed, but 4 already covers this project's own 2025/2025-T/2026/
# 2026-DNS comparison exactly.
_LINESTYLES = ['-', (0, (6, 2)), (0, (1, 1.6)), (0, (5, 1.5, 1.2, 1.5))]

_COMPONENT_LABELS = {
    'axial': r'$F_{axial}$ [N]',
    'radial': r'$F_{radial}$ [N]',
    'tangential': r'$F_{tangential}$ [N]',
}


def _strip_colors(items, cmap):

    '''
    Discrete colors for `items` (a strip list, or - for
    plot_time_trace_compare()'s own color=case encoding - a case list),
    evenly sampled from `cmap` - same convention as StripForces' own
    plots. Name kept from its original strip-only use; genuinely generic
    (only `len(items)` matters), so reused as-is rather than duplicated.
    '''

    return plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(items)))


def _build_shared_legend_case_color(legend_ax, strips, case_labels, case_colors, radii=None, fontsize=8):

    '''
    ONE combined legend for the whole multi-panel figure, drawn INSIDE
    its own dedicated axes cell (the fourth cell of the 2x2 grid built by
    plot_time_trace_compare()/plot_vs_angle_compare(), the three data
    panels occupying the other three) rather than as a figure-level
    floating legend anchored by `bbox_to_anchor`. This is a deliberate
    choice, not just a style preference: a figure-spanning legend row
    below three narrow panels, combined with print-sized (14pt+) fonts,
    was found during development to interact badly with
    `tight_layout()`/`bbox_inches='tight'`, producing large, asymmetric
    blank margins - confirmed by direct visual inspection
    (fig56_compare_test.png in the test images folder). A legend confined
    to a normal axes cell is laid out by matplotlib like any other
    subplot content, avoiding that failure mode entirely.

    One colored line per CASE (the color dimension in both of this
    module's plot functions), followed by one black line per selected
    strip (the linestyle dimension) - so a reader can decode both
    encodings from a single legend rather than needing them explained
    twice or guessed at.

    radii : array-like, optional
        Physical radius [m] of each selected strip (e.g.
        result['radius'][list(strips)]) - if given, appended to each
        strip's legend label as r [m] for a physically meaningful
        reference; omitted if not available (e.g. mismatched strip
        layouts between cases - see plot_time_trace_compare()'s own
        docstring on this assumption).
    '''

    handles, labels = [], []

    for j, case_label in enumerate(case_labels):
        handles.append(Line2D([0], [0], color=case_colors[j % len(case_colors)], linewidth=2.2))
        labels.append(str(case_label))

    for i, strip_i in enumerate(strips):
        handles.append(Line2D([0], [0], color='black', linestyle=_LINESTYLES[i % len(_LINESTYLES)], linewidth=1.6))
        if radii is not None:
            labels.append(f'Strip {strip_i + 1} ($r$={radii[i]:.3f} m)')
        else:
            labels.append(f'Strip {strip_i + 1}')

    legend_ax.axis('off')
    legend_ax.legend(handles, labels, loc='center', frameon=False, fontsize=fontsize,
                      handlelength=2.0, labelspacing=0.5, borderpad=0.5, handletextpad=0.6)


def _check_consistent_strip_layout(results: dict, strips):

    '''
    Warns (does not raise) if the selected strips' physical radii differ
    non-trivially between cases - _build_shared_legend_case_color()'s radii come
    from only ONE case (the last one processed), which silently assumes
    every case shares the same span_min/span_max/n_span_bins layout (as
    this project's own 4-case thesis comparison does, all built with the
    same 10-equally-spaced-strips convention - see class docstring).
    '''

    radii_by_case = {}
    for label, result in results.items():
        radius = result['radius']
        sel = np.asarray(strips)
        if np.any(sel >= len(radius)):
            raise ValueError(f"strips {strips} out of range for case '{label}' ({len(radius)} strips available).")
        radii_by_case[label] = radius[sel]

    reference = next(iter(radii_by_case.values()))
    for label, radii in radii_by_case.items():
        if not np.allclose(radii, reference, rtol=0.05, equal_nan=True):
            import warnings
            warnings.warn(
                f"Case '{label}' selected strips sit at different physical radii than the reference case "
                f"({radii} vs {reference}) - the shared legend's radius labels will only reflect the LAST "
                "case processed. Consider using bladeprocessor.strip_forces.StripForces.nearest_strip() per "
                "case instead of a shared raw index list if cases don't share the same span layout."
            )
            break


def plot_time_trace_compare(results: dict, dt: float = None, strips=(0, 4, 9),
                             component_list=('axial', 'tangential', 'radial'), cmap: str = 'cividis',
                             total_width_in: float = 6.3, total_height_in: float = 4.6,
                             tick_fontsize: int = 14, label_fontsize: int = 15, legend_fontsize: int = 14,
                             savepath: str = None, dpi: int = 600):

    '''
    Raw per-frame force time trace, compared across several SIMULATIONS
    at once - the cross-case counterpart to StripForces.plot_time_trace()
    (which compares strips WITHIN one case). One panel per component (3
    by default, matching StripForces' own axial/radial/tangential split),
    each panel overlaying every case's selected strips.

    Deliberately the OPPOSITE of the module's usual color=strip/
    linestyle=case convention (see module docstring, and
    plot_vs_angle_compare() which still uses it): here, color = CASE
    (still `cmap`-sampled, same as everywhere else in this module - see
    _strip_colors(); distinguishing the four cases is this figure's
    actual point) and linestyle = strip (reusing the same large,
    noise-robust dash patterns, _LINESTYLES). The raw traces this
    function plots are noisy and heavily overlapping - confirmed, on a
    real case, that 4 cases told apart only by linestyle under those
    conditions is hard to read even with the large custom dash patterns,
    whereas the strip-to-strip MAGNITUDE spread (the reason every strip
    still shares one y-axis per panel, not faceted into separate
    subplots) remains clear either way, since it's carried by each
    line's vertical position, not by its color or style. All
    cases/strips share ONE legend for the whole figure.

    Parameters
    ----------
    results : dict[str, dict]
        {case_label: compute()'s return value} - one entry per
        simulation to compare, e.g. {'2025': result_2025, '2025-T':
        result_2025t, '2026': result_2026, '2026-DNS': result_2026dns}.
        Every result must share the same strip INDEXING (this project's
        own 4-case comparison does, all built with the same 10-equally-
        spaced-strips convention) - _check_consistent_strip_layout()
        warns, but does not raise, if that assumption looks violated.
    dt : float, optional
        Physical timestep [s] - if omitted, the x-axis is frame index
        (dimensionless). Assumed the SAME dt for every case; pass
        per-case dt separately (call this function once per case's own
        dt, with `ax=` reused) if that's not true here.
    strips : sequence of int
        Which strip indices to plot from EVERY case - 3 by default (this
        project's own choice: enough to show the spanwise trend without
        the visual overload of all 10).
    component_list : sequence of str
        Which of 'axial', 'radial', 'tangential' to plot - EXACTLY 3,
        no more, no fewer: they fill the top-left, top-right, and
        bottom-left cells of a 2x2 grid, with the bottom-right cell
        reserved for the shared legend (see module/_build_shared_legend_case_color()
        docstrings for why a grid cell, not a figure-spanning legend row,
        is used). Passing a different number raises ValueError.
    cmap : str
        Colormap each CASE's own color is sampled from (evenly spaced,
        one sample per case - see _strip_colors()) - 'cividis' by
        default, same colormap/convention used for strip colors
        elsewhere in this module and in StripForces' own plots, kept
        here even though color now encodes case rather than strip, for
        visual consistency across this project's figures.
    total_width_in, total_height_in : float
        The figure's FINAL physical size in inches - see module
        docstring's "Print-size handling" note for the EXACT condition
        under which `tick_fontsize`/`label_fontsize` come out at that
        same point size on the printed page. Default 6.3 x 4.6 in
        matches a common single-column thesis \\linewidth at a roughly
        square-ish 2x2 grid; set `total_width_in` to your own document's
        actual \\linewidth (in inches: pt / 72.27).
    tick_fontsize, label_fontsize, legend_fontsize : int
        Font sizes in points, applied via a LOCAL plt.rc_context (does
        not affect this project's other, larger-format single-panel
        plots' own global rcParams). Default 14/15/14 pt, chosen so
        labels read comfortably above typical (11-12 pt) thesis body
        text once printed at `total_width_in` - see the module
        docstring's print-size note for the exact condition this relies
        on.

    Returns
    -------
    (fig, axes) - axes is the (2, 2) array from plt.subplots(); axes[1,1]
        is the legend cell (axis turned off), not a data panel.
    '''

    if len(results) == 0:
        raise ValueError("results is empty - need at least one case to plot.")
    if len(component_list) != 3:
        raise ValueError(
            f"component_list must have EXACTLY 3 entries to fill the 2x2 grid's three data cells "
            f"(bottom-right is reserved for the legend) - got {len(component_list)}: {component_list}."
        )

    case_labels = list(results.keys())
    strips = list(strips)
    _check_consistent_strip_layout(results, strips)
    case_colors = _strip_colors(case_labels, cmap)

    with plt.rc_context({
        'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern'],
        'axes.labelsize': label_fontsize, 'xtick.labelsize': tick_fontsize,
        'ytick.labelsize': tick_fontsize, 'legend.fontsize': legend_fontsize,
    }):

        fig, axes = plt.subplots(2, 2, figsize=(total_width_in, total_height_in))
        data_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]

        for ax, component in zip(data_axes, component_list):
            for j, case_label in enumerate(case_labels):
                arr = results[case_label][component]  # (n_frames, n_span_bins)
                t = np.arange(arr.shape[0]) * dt if dt is not None else np.arange(arr.shape[0])
                for i, strip_i in enumerate(strips):
                    ax.plot(t, arr[:, strip_i], color=case_colors[j],
                            linestyle=_LINESTYLES[i % len(_LINESTYLES)], linewidth=1.3, alpha=0.9)

            ax.set_xlabel('Time [s]' if dt is not None else 'Frame index')
            ax.set_ylabel(_COMPONENT_LABELS.get(component, component))
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.tick_params(axis='both', labelsize=tick_fontsize)

        last_case = results[case_labels[-1]]
        radii_for_legend = last_case['radius'][strips] if 'radius' in last_case else None
        _build_shared_legend_case_color(axes[1, 1], strips, case_labels, case_colors,
                                         radii=radii_for_legend, fontsize=legend_fontsize)

        fig.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=dpi)

    return fig, axes


def plot_vs_angle_compare(phase_locked: dict, strips=(0, 4, 9),
                           component_list=('axial', 'tangential', 'radial'), cmap: str = 'cividis',
                           total_width_in: float = 6.3, total_height_in: float = 4.8,
                           tick_fontsize: int = 10, label_fontsize: int = 15, legend_fontsize: int = 14,
                           title_pad: float = 28,
                           savepath: str = None, dpi: int = 600):

    '''
    Phase-locked (revolution-folded) force vs. rotor azimuth, compared
    across several SIMULATIONS at once - the cross-case counterpart to
    StripForces.plot_vs_angle() (which compares strips WITHIN one case).
    2x2 grid of POLAR axes (0 deg at 3 o'clock/east, increasing
    counterclockwise, same convention as StripForces.plot_vs_angle());
    the bottom-right cell stays a normal (non-polar) axes, holding the
    shared legend.

    Same color = CASE (cmap-sampled) / linestyle = strip design as
    plot_time_trace_compare() - see that function's own docstring for
    why (distinguishing cases is each figure's actual point; the
    module's older color=strip/linestyle=case convention is no longer
    used by either function in this module).

    Two print-size issues specific to a compact polar subplot, fixed
    here but not relevant to plot_time_trace_compare()'s Cartesian
    panels: (1) the default tick_fontsize (14, fine on a Cartesian axis)
    crowded BOTH the angular (0/45/90.../deg) and radial (force value)
    tick labels against each other and against the grid circles on a
    real case - lowered to 10 by default here. (2) matplotlib draws the
    polar 90-degree label just above the plot circle, which collided
    directly with this panel's own title at the previous, smaller `pad`
    - confirmed on a real case (title_pad=16 was not enough); raised to
    28 by default, exposed as its own parameter since the right value
    depends on tick_fontsize/label_fontsize and the figure's own
    physical size.

    Parameters
    ----------
    phase_locked : dict[str, dict]
        {case_label: phase_lock()'s return value} - one entry per
        simulation to compare. Same shared-strip-layout assumption as
        plot_time_trace_compare()'s `results`.
    component_list : sequence of str
        EXACTLY 3 entries - see plot_time_trace_compare()'s own
        docstring for why.
    title_pad : float
        Extra space, in points, between each polar panel's own top edge
        and its title - see above for why this needed raising past
        matplotlib's usual default.
    strips, cmap, total_width_in, total_height_in, tick_fontsize,
    label_fontsize, legend_fontsize : see plot_time_trace_compare() -
        identical meaning here (tick_fontsize's default differs - see
        above).

    Returns
    -------
    (fig, axes) - axes is the (2, 2) array from plt.subplots(); axes[1,1]
        is the (non-polar) legend cell, not a data panel.
    '''

    if len(phase_locked) == 0:
        raise ValueError("phase_locked is empty - need at least one case to plot.")
    if len(component_list) != 3:
        raise ValueError(
            f"component_list must have EXACTLY 3 entries to fill the 2x2 grid's three polar data cells "
            f"(bottom-right is reserved for the legend) - got {len(component_list)}: {component_list}."
        )

    case_labels = list(phase_locked.keys())
    strips = list(strips)
    _check_consistent_strip_layout(phase_locked, strips)
    case_colors = _strip_colors(case_labels, cmap)

    with plt.rc_context({
        'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern'],
        'axes.labelsize': label_fontsize, 'xtick.labelsize': tick_fontsize,
        'ytick.labelsize': tick_fontsize, 'legend.fontsize': legend_fontsize,
    }):

        fig = plt.figure(figsize=(total_width_in, total_height_in))
        data_axes = [
            fig.add_subplot(2, 2, 1, projection='polar'),
            fig.add_subplot(2, 2, 2, projection='polar'),
            fig.add_subplot(2, 2, 3, projection='polar'),
        ]
        legend_ax = fig.add_subplot(2, 2, 4)  # deliberately NOT polar - see module docstring

        for ax, component in zip(data_axes, component_list):
            for j, case_label in enumerate(case_labels):
                pl = phase_locked[case_label]
                theta = np.deg2rad(pl['azimuth_deg'])
                theta_closed = np.append(theta, theta[0])  # close the loop, no visible gap at 0/360 deg
                vals = pl[component]
                for i, strip_i in enumerate(strips):
                    y = vals[:, strip_i]
                    y_closed = np.append(y, y[0])
                    ax.plot(theta_closed, y_closed, color=case_colors[j],
                            linestyle=_LINESTYLES[i % len(_LINESTYLES)], linewidth=1.3, alpha=0.9)

            ax.set_theta_zero_location('E')
            ax.set_theta_direction(1)
            ax.set_rlabel_position(22.5)  # off the 45deg gridline, where it would otherwise overlap
            ax.set_title(_COMPONENT_LABELS.get(component, component), fontsize=label_fontsize, pad=title_pad)
            ax.tick_params(axis='both', labelsize=tick_fontsize)

        last_case = phase_locked[case_labels[-1]]
        radii_for_legend = last_case['radius'][strips] if 'radius' in last_case else None
        _build_shared_legend_case_color(legend_ax, strips, case_labels, case_colors,
                                         radii=radii_for_legend, fontsize=legend_fontsize)

        axes = np.array([[data_axes[0], data_axes[1]], [data_axes[2], legend_ax]], dtype=object)

        fig.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=dpi)

    return fig, axes


def _check_consistent_radii(curves_by_case: dict, radii_list):

    '''
    Warns (does not raise) if any case is missing one of `radii_list`'s own target radii - every
    case is expected to have been built by calling SurfaceVariable.at_radii()/
    FrictionLines.cf_at_radii() with the SAME `radii` argument (see plot_radii_compare()'s own
    docstring), so the dict keys should match exactly (they're the literal requested r_target
    values, not a re-detected band center) - a mismatch here means a case was built with a
    different `radii` list, not a small floating-point drift.
    '''

    for label, case_curves in curves_by_case.items():
        missing = [r for r in radii_list if r not in case_curves]
        if missing:
            import warnings
            warnings.warn(
                f"Case '{label}' has no curve for radius target(s) {missing} (built with a "
                "different `radii` list than the other cases?) - those will be skipped for this "
                "case only."
            )


def plot_radii_compare(curves_by_case: dict, ylabel: str, cmap: str = 'cividis',
                        marker_only_cases=(), total_width_in: float = 6.3, total_height_in: float = 4.6,
                        tick_fontsize: int = 14, label_fontsize: int = 15, legend_fontsize: int = 11,
                        savepath: str = None, dpi: int = 600):

    '''
    Cp or ONE Cf component vs. local x/c at a handful of fixed radii, overlaid across several
    SIMULATIONS on one panel - the cross-case counterpart to
    SurfaceVariable.plot_at_radii()/plot_cp_radii() and FrictionLines.plot_cf_radii() (which
    compare RADII within one case). Unlike this module's own usual color=CASE convention
    (plot_time_trace_compare()/plot_vs_angle_compare() - see module docstring for that rationale),
    here COLOR encodes RADIUS and LINESTYLE encodes case: every radius-to-radius comparison is a
    group of same-colored lines across cases, so reading "how do the cases differ at THIS radius"
    only needs following one color, not cross-referencing a linestyle legend as well - confirmed
    more readable in practice than the opposite mapping for this specific figure (unlike
    plot_time_trace_compare()'s own noisy raw traces, these are smooth per-radius curves, so
    radius is the natural grouping to scan visually, not case).

    Call this ONCE PER PHYSICAL QUANTITY (Cp, Cf magnitude, Cf chordwise, Cf spanwise, ...) - it
    does not know or assume which one it's plotting (same spirit as SurfaceVariable.plot_at_radii()'s
    own get_values/ylabel split); build 4 separate `curves_by_case` dicts upstream (one per
    quantity, see manager_comparison.py's own main_radii_compare()) and call this 4 times.

    Parameters
    ----------
    curves_by_case : dict[str, dict[float, dict[str, tuple[np.ndarray, np.ndarray]]]]
        {case_label: {radius: {surface: (xc, values)}}} - i.e., for EVERY case to compare,
        either SurfaceVariable.at_radii(radii, ..., surface=('Upper', 'Lower'))'s own return
        value AS-IS (Cp needs BOTH surfaces - a suction-side-only branch is not a physically
        complete Cp-vs-x/c picture, confirmed directly: an Upper-only cross-case Cp plot looked
        nothing like the matching single-case plot_cp_radii() reference, which always shows
        both) or FrictionLines.cf_at_radii(radii, surface='Upper', ...)'s own (xc, values) wrapped
        in a single-key {'Upper': (xc, values)} dict (Cf only ever needs Upper here - that one is
        already single-surface). Every case MUST have been built with the SAME `radii` target
        list (see _check_consistent_radii(), which warns rather than raises if a case is missing
        one).
    ylabel : str
        e.g. r'$C_p$ [-]', r'$|C_f|$ [-]', r'$C_f$ (chordwise) [-]' - this function does not know
        or assume which physical quantity is being compared.
    marker_only_cases : iterable of str, optional
        Case labels (a subset of curves_by_case's own keys) to render as DISCRETE MARKERS
        (color=radius still, but no connecting line and no linestyle slot used) instead of a
        line - for reference/validation data that isn't this project's own LBM simulations (e.g.
        a digitized iLES/experimental Cp curve) and so shouldn't visually compete with the
        smooth simulation lines for one of the 4 linestyle slots (and whose own point ORDER,
        for digitized data, may not be reliably sortable/connectable as a single clean curve
        anyway) - standard convention: lines for simulations, symbols for reference data.
    cmap, total_width_in, total_height_in, tick_fontsize, label_fontsize : see
        plot_time_trace_compare() - identical meaning here.
    legend_fontsize : int
        Smaller default (11) than the 2x2-grid functions' own 14 pt - this is a single-panel
        figure with the legend drawn INSIDE the data axes (not its own dedicated grid cell, which
        only the 2x2 multi-panel functions need - see module docstring's print-size note on why
        THAT combination specifically needed a dedicated cell), so a smaller legend keeps it from
        crowding the curves themselves.

    Returns
    -------
    (fig, ax)
    '''

    if len(curves_by_case) == 0:
        raise ValueError("curves_by_case is empty - need at least one case to plot.")

    case_labels = list(curves_by_case.keys())
    radii_list = sorted(next(iter(curves_by_case.values())).keys())
    _check_consistent_radii(curves_by_case, radii_list)
    radius_colors = _strip_colors(radii_list, cmap)

    with plt.rc_context({
        'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern'],
        'axes.labelsize': label_fontsize, 'xtick.labelsize': tick_fontsize,
        'ytick.labelsize': tick_fontsize, 'legend.fontsize': legend_fontsize,
    }):

        fig, ax = plt.subplots(figsize=(total_width_in, total_height_in))

        line_case_labels = [c for c in case_labels if c not in marker_only_cases]
        marker_case_labels = [c for c in case_labels if c in marker_only_cases]

        for case_label in case_labels:
            case_curves = curves_by_case[case_label]
            is_marker = case_label in marker_only_cases
            j = line_case_labels.index(case_label) if not is_marker else None
            for i, r_target in enumerate(radii_list):
                if r_target not in case_curves:
                    continue
                # Every surface at this radius (1 for Cf - Upper only; 2 for Cp - Upper AND
                # Lower, see this function's own docstring) shares the SAME color (radius) and
                # linestyle (case) - same "both surfaces, same color per radius" convention
                # SurfaceVariable.plot_at_radii() itself uses for the single-case reference plot.
                for xc, values in case_curves[r_target].values():
                    if is_marker:
                        ax.plot(xc, values, color=radius_colors[i], linestyle='none', marker='o',
                                markersize=4, markerfacecolor='none', markeredgewidth=1.2, alpha=0.9)
                    else:
                        ax.plot(xc, values, color=radius_colors[i],
                                linestyle=_LINESTYLES[j % len(_LINESTYLES)], linewidth=1.6, alpha=0.9)

        ax.set_xlabel(r'$x/c$ [-]')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(axis='both', labelsize=tick_fontsize)

        radius_handles = [Line2D([0], [0], color=radius_colors[i], linewidth=2.2) for i in range(len(radii_list))]
        line_case_handles = [Line2D([0], [0], color='black', linestyle=_LINESTYLES[j % len(_LINESTYLES)], linewidth=1.6)
                              for j in range(len(line_case_labels))]
        marker_case_handles = [Line2D([0], [0], color='black', linestyle='none', marker='o', markersize=5,
                                      markerfacecolor='none', markeredgewidth=1.2)
                                for _ in marker_case_labels]
        ax.legend(radius_handles + line_case_handles + marker_case_handles,
                  [f'$r$={r:.3f} m' for r in radii_list] + line_case_labels + marker_case_labels,
                  fontsize=legend_fontsize, ncol=2, loc='best', framealpha=0.85)

        fig.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=dpi)

    return fig, ax
