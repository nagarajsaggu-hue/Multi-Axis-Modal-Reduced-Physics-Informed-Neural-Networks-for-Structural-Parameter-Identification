"""Phase-2 visualisation for PINN SDOF system identification.

Four figures are produced:

1. timeseries
   ANSYS reference-coordinate displacement against the PINN prediction.

2. convergence
   Identified stiffness and effective damping versus training iteration.

3. spectrum
   Spectrum of the Phase-2 reference-coordinate displacement, showing the
   excitation frequency and the explicitly selected ANSYS reference mode.

4. comparison
   Stiffness comparison between PINN, transient least squares and ANSYS modal.

Phase-2 reference coordinates:

    X -> S_C07 / Node 3461 / Mode 1
    Y -> S_C07 / Node 3461 / Mode 2
    Z -> S_C07 / Node 3461 / Mode 4

No Node3575 logic and no modal-selection heuristic are used here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =====================================================================
# PATH HELPERS
# =====================================================================

def _fig_dir(cfg):
    """Return/create the figure directory for the active direction."""

    directory = (
        Path(cfg["output"]["dir"])
        /
        "figures"
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


def _reference_label(cfg):
    """Human-readable Phase-2 reference-coordinate label."""

    reference = cfg["reference"]

    return (
        f"{reference['sensor']} / "
        f"Node {reference['node_id']} / "
        f"Mode {reference['mode']}"
    )


# =====================================================================
# 1. TIME-SERIES FIT
# =====================================================================

def timeseries(cfg, tag, window=None):
    """Plot ANSYS reference displacement, PINN prediction and error."""

    path = (
        Path(cfg["output"]["dir"])
        /
        f"prediction_{tag}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {path}"
        )

    df = pd.read_csv(path)

    required = (
        "Time (s)",
        "u_measured (m)",
        "u_predicted (m)",
        "error (m)",
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Prediction file is missing columns: {missing}"
        )

    t = df["Time (s)"].to_numpy(dtype=float)
    measured = df["u_measured (m)"].to_numpy(dtype=float)
    predicted = df["u_predicted (m)"].to_numpy(dtype=float)
    error = df["error (m)"].to_numpy(dtype=float)

    if not all(
        np.all(np.isfinite(values))
        for values in (
            t,
            measured,
            predicted,
            error,
        )
    ):
        raise ValueError(
            "Prediction file contains NaN/Inf."
        )

    selection = np.ones(
        len(t),
        dtype=bool,
    )

    if window is not None:

        if len(window) != 2:
            raise ValueError(
                "window must contain [t_min, t_max]."
            )

        t_min = float(window[0])
        t_max = float(window[1])

        selection = (
            (t >= t_min)
            &
            (t <= t_max)
        )

        if not np.any(selection):
            raise ValueError(
                f"No samples inside plotting window {window}."
            )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    rms_measured = float(
        np.sqrt(
            np.mean(
                measured ** 2
            )
        )
    )

    relative_rmse = (
        rmse / rms_measured
        if rms_measured > 0.0
        else float("nan")
    )

    direction = cfg["active_direction"]
    reference_label = _reference_label(cfg)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 6),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                2,
                1,
            ]
        },
    )

    axes[0].plot(
        t[selection],
        measured[selection],
        linewidth=1.3,
        label="ANSYS",
    )

    axes[0].plot(
        t[selection],
        predicted[selection],
        linewidth=1.0,
        linestyle="--",
        label="PINN",
    )

    axes[0].set_ylabel(
        "displacement (m)"
    )

    axes[0].set_title(
        (
            f"Phase-2 {direction} response — "
            f"{reference_label}\n"
            f"RMSE = {rmse:.3e} m "
            f"({relative_rmse * 100.0:.3f}% of RMS)"
        )
    )

    axes[0].legend(
        loc="upper right"
    )

    axes[0].grid(
        alpha=0.3
    )

    axes[1].plot(
        t[selection],
        error[selection],
        linewidth=0.9,
    )

    axes[1].axhline(
        0.0,
        linewidth=0.7,
    )

    axes[1].set_ylabel(
        "PINN - ANSYS (m)"
    )

    axes[1].set_xlabel(
        "time (s)"
    )

    axes[1].grid(
        alpha=0.3
    )

    fig.tight_layout()

    output = (
        _fig_dir(cfg)
        /
        f"timeseries_{tag}.png"
    )

    fig.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output


# =====================================================================
# 2. PARAMETER CONVERGENCE
# =====================================================================

def convergence(cfg, tag, reference=None):
    """Plot PINN K and effective C versus training iteration.

    The authoritative ANSYS reference is used for stiffness.

    No physical ANSYS C line is imposed because the FE model has zero
    physical damping while the PINN C is interpreted as an effective
    reduced-model / numerical damping parameter.

    The optional `reference` argument is retained only for compatibility
    with the existing plotting wrapper.
    """

    path = (
        Path(cfg["output"]["dir"])
        /
        f"convergence_{tag}.csv"
    )

    if not path.exists():
        return None

    df = pd.read_csv(path)

    required = (
        "iteration",
        "K_N_per_m",
        "C_N_s_per_m",
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Convergence file is missing columns: {missing}"
        )

    iteration = df["iteration"].to_numpy(dtype=float)
    stiffness = df["K_N_per_m"].to_numpy(dtype=float)
    damping = df["C_N_s_per_m"].to_numpy(dtype=float)

    ansys_stiffness = float(
        cfg["reference"]["equivalent_stiffness_N_per_m"]
    )

    final_iteration = int(
        cfg["training"]["iterations"]
    )

    direction = cfg["active_direction"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
    )

    # -----------------------------------------------------------------
    # Stiffness
    # -----------------------------------------------------------------

    axes[0].plot(
        iteration,
        stiffness,
        linewidth=1.3,
        label="PINN",
    )

    axes[0].axhline(
        ansys_stiffness,
        linestyle="--",
        linewidth=1.1,
        label="ANSYS modal reference",
    )

    axes[0].axvline(
        final_iteration,
        linestyle=":",
        linewidth=1.0,
        label=f"validated stop = {final_iteration}",
    )

    axes[0].set_xlabel(
        "iteration"
    )

    axes[0].set_ylabel(
        "K (N/m)"
    )

    axes[0].set_title(
        f"{direction}: stiffness convergence"
    )

    axes[0].legend()

    axes[0].grid(
        alpha=0.3
    )

    # -----------------------------------------------------------------
    # Effective damping
    # -----------------------------------------------------------------

    axes[1].plot(
        iteration,
        damping,
        linewidth=1.3,
        label="PINN effective C",
    )

    axes[1].axvline(
        final_iteration,
        linestyle=":",
        linewidth=1.0,
        label=f"validated stop = {final_iteration}",
    )

    axes[1].set_xlabel(
        "iteration"
    )

    axes[1].set_ylabel(
        "C (N·s/m)"
    )

    axes[1].set_title(
        f"{direction}: effective damping convergence"
    )

    axes[1].legend()

    axes[1].grid(
        alpha=0.3
    )

    fig.suptitle(
        (
            f"Phase-2 parameter convergence — "
            f"{_reference_label(cfg)}"
        )
    )

    fig.tight_layout()

    output = (
        _fig_dir(cfg)
        /
        f"convergence_{tag}.png"
    )

    fig.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output


# =====================================================================
# 3. RESPONSE SPECTRUM
# =====================================================================

def spectrum(cfg, data, tag):
    """Plot reference-coordinate displacement spectrum.

    Only the explicitly selected Phase-2 ANSYS modal reference is shown.
    """

    displacement = np.asarray(
        data.u,
        dtype=float,
    ).reshape(-1)

    n = len(displacement)

    if n < 2:
        raise ValueError(
            "Too few samples for spectrum."
        )

    if not np.all(
        np.isfinite(displacement)
    ):
        raise ValueError(
            "Displacement contains NaN/Inf."
        )

    frequencies = np.fft.rfftfreq(
        n,
        float(data.dt),
    )

    window = np.hanning(n)

    magnitude = np.abs(
        np.fft.rfft(
            displacement
            *
            window
        )
    )

    peak = float(
        magnitude.max()
    )

    if peak > 0.0:
        magnitude = magnitude / peak

    drive_frequency = float(
        cfg["physics"]["force_frequency"]
    )

    reference_frequency = float(
        cfg["reference"]["frequency_hz"]
    )

    fourier_max = float(
        cfg[
            "network"
        ][
            "fourier_features"
        ][
            "f_max"
        ]
    )

    upper_frequency = min(
        float(frequencies.max()),
        max(
            60.0,
            fourier_max + 5.0,
            reference_frequency + 5.0,
        ),
    )

    fig, axis = plt.subplots(
        figsize=(10, 4.5)
    )

    axis.semilogy(
        frequencies,
        np.maximum(
            magnitude,
            1.0e-8,
        ),
        linewidth=1.1,
        label="ANSYS reference-coordinate response",
    )

    axis.axvline(
        drive_frequency,
        linestyle="--",
        linewidth=1.1,
        label=f"excitation = {drive_frequency:.3f} Hz",
    )

    axis.axvline(
        reference_frequency,
        linestyle=":",
        linewidth=1.2,
        label=(
            f"ANSYS Mode {cfg['reference']['mode']} "
            f"= {reference_frequency:.3f} Hz"
        ),
    )

    axis.set_xlim(
        0.0,
        upper_frequency,
    )

    axis.set_xlabel(
        "frequency (Hz)"
    )

    axis.set_ylabel(
        "normalised amplitude"
    )

    axis.set_title(
        (
            f"Phase-2 {cfg['active_direction']} displacement spectrum — "
            f"{_reference_label(cfg)}"
        )
    )

    axis.legend()

    axis.grid(
        alpha=0.3
    )

    fig.tight_layout()

    output = (
        _fig_dir(cfg)
        /
        f"spectrum_{tag}.png"
    )

    fig.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output


# =====================================================================
# 4. STIFFNESS COMPARISON
# =====================================================================

def comparison_bars(cfg, comparison, tag):
    """Plot stiffness from PINN, LS diagnostic and ANSYS modal reference."""

    rows = [
        row
        for row in comparison["rows"]
        if row.get("K_N_per_m") is not None
    ]

    if len(rows) < 2:
        return None

    names = [
        row["method"]
        for row in rows
    ]

    values = np.asarray(
        [
            float(
                row["K_N_per_m"]
            )
            for row in rows
        ],
        dtype=float,
    )

    ansys_stiffness = float(
        cfg["reference"]["equivalent_stiffness_N_per_m"]
    )

    percent_difference = (
        100.0
        *
        (
            values
            -
            ansys_stiffness
        )
        /
        ansys_stiffness
    )

    fig, axis = plt.subplots(
        figsize=(8, 4.8)
    )

    bars = axis.bar(
        names,
        values,
    )

    # -------------------------------------------------------------
    # IMPORTANT:
    # Labels use screen-point offsets instead of stiffness-value
    # offsets. This prevents bbox_inches="tight" from creating a
    # very large blank region above the plot.
    # -------------------------------------------------------------

    for bar, value, difference in zip(
        bars,
        values,
        percent_difference,
    ):

        axis.annotate(
            (
                f"{value:.4e}\n"
                f"{difference:+.3f}% vs ANSYS"
            ),
            xy=(
                bar.get_x()
                +
                bar.get_width()
                /
                2.0,
                value,
            ),
            xytext=(
                0,
                6,
            ),
            textcoords="offset points",
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=9,
        )

    axis.set_ylabel(
        "K (N/m)"
    )

    axis.set_title(
        (
            f"Phase-2 {cfg['active_direction']} stiffness comparison — "
            f"{_reference_label(cfg)}"
        )
    )

    axis.grid(
        alpha=0.3,
        axis="y",
    )

    lower = float(
        values.min()
    )

    upper = float(
        values.max()
    )

    span = max(
        upper
        -
        lower,
        0.001
        *
        ansys_stiffness,
    )

    axis.set_ylim(
        lower
        -
        0.30
        *
        span,
        upper
        +
        0.55
        *
        span,
    )

    fig.tight_layout()

    output = (
        _fig_dir(cfg)
        /
        f"comparison_{tag}.png"
    )

    fig.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output