"""Phase-2 SDOF physics and parameter utilities.

PHYSICAL EQUATION
-----------------
    M u'' + C u' + K u = F_eq(t)

where:

    M       known equivalent/reference-coordinate modal mass
    K       identified equivalent stiffness
    C       identified effective damping
    F_eq(t) approved reference-coordinate equivalent force history

The Phase-2 forcing is NOT regenerated as an ideal sine wave.

Instead, F_eq(t) is read from the approved Phase2_*_PINN_Input.csv file
by dataio.py and linearly interpolated at the arbitrary collocation times
used by DeepXDE.

SCALED EQUATION
---------------
With:

    u = U0 * u_hat
    F = F0 * F_hat

division by F0 gives:

    m_hat u_hat'' + c_hat u_hat' + k_hat u_hat = F_hat(t)

where:

    m_hat = M * U0 / F0
    c_hat = C * U0 / F0
    k_hat = K * U0 / F0

Time remains in physical seconds. Therefore all automatic derivatives are
with respect to real time.

DAMPING INTERPRETATION
----------------------
ANSYS physical damping was zero. The transient solution used numerical
damping. Therefore a non-zero identified C is interpreted as an effective
reduced-model / numerical damping quantity, not automatically as material
or structural physical damping.
"""

from __future__ import annotations

import math

import numpy as np


# =====================================================================
# SCALING
# =====================================================================

def scaled_mass(cfg) -> float:
    """Return known equivalent mass in scaled equation units."""

    return (
        float(cfg["physics"]["mass"])
        * float(cfg["scaling"]["displacement"])
        / float(cfg["scaling"]["force"])
    )


def to_scaled(
    k_physical,
    c_physical,
    cfg,
):
    """Convert physical K, C to scaled k_hat, c_hat."""

    scale = (
        float(cfg["scaling"]["displacement"])
        /
        float(cfg["scaling"]["force"])
    )

    return (
        float(k_physical) * scale,
        float(c_physical) * scale,
    )


def to_physical(
    k_hat,
    c_hat,
    cfg,
):
    """Convert scaled k_hat, c_hat to physical K, C."""

    scale = (
        float(cfg["scaling"]["force"])
        /
        float(cfg["scaling"]["displacement"])
    )

    return (
        float(k_hat) * scale,
        float(c_hat) * scale,
    )


# =====================================================================
# DERIVED STRUCTURAL PARAMETERS
# =====================================================================

def derived(
    k_physical,
    c_physical,
    mass,
):
    """Compute natural frequency and damping ratio."""

    k_physical = float(
        k_physical
    )

    c_physical = float(
        c_physical
    )

    mass = float(
        mass
    )

    if (
        k_physical <= 0
        or
        mass <= 0
    ):

        return {
            "omega_n_rad_s":
                float("nan"),

            "f_n_hz":
                float("nan"),

            "zeta":
                float("nan"),
        }

    omega_n = math.sqrt(
        k_physical
        /
        mass
    )

    zeta = (
        c_physical
        /
        (
            2.0
            *
            math.sqrt(
                k_physical
                *
                mass
            )
        )
    )

    return {
        "omega_n_rad_s":
            omega_n,

        "f_n_hz":
            omega_n
            /
            (
                2.0
                *
                math.pi
            ),

        "zeta":
            zeta,
    }


# =====================================================================
# FORCE — NUMPY REFERENCE VERSION
# =====================================================================

def force_physical(
    t,
    data,
):
    """Interpolate approved equivalent force in physical Newtons.

    This NumPy implementation is intended for diagnostics, evaluation and
    plotting.

    The PINN residual uses the PyTorch implementation below.
    """

    t_query = np.asarray(
        t,
        dtype=np.float64,
    )

    t_data = np.asarray(
        data.t,
        dtype=np.float64,
    ).reshape(
        -1
    )

    f_data = np.asarray(
        data.f,
        dtype=np.float64,
    ).reshape(
        -1
    )

    if t_data.size < 2:

        raise ValueError(
            "At least two force samples are required."
        )

    if np.any(
        np.diff(
            t_data
        )
        <= 0
    ):

        raise ValueError(
            "Force time history must be strictly increasing."
        )

    tolerance = max(
        1.0e-12,
        1.0e-9
        *
        float(
            data.dt
        ),
    )

    if (
        np.min(
            t_query
        )
        <
        t_data[0]
        -
        tolerance
        or
        np.max(
            t_query
        )
        >
        t_data[-1]
        +
        tolerance
    ):

        raise ValueError(
            "Requested force interpolation time lies outside "
            f"[{t_data[0]}, {t_data[-1]}] s."
        )

    return np.interp(
        t_query,
        t_data,
        f_data,
    )


def force_scaled(
    t,
    data,
):
    """Interpolate approved dimensionless equivalent force."""

    physical = force_physical(
        t,
        data,
    )

    return (
        physical
        /
        float(
            np.max(
                np.abs(
                    data.f
                )
            )
        )
        if False
        else physical
    )


# =====================================================================
# PYTORCH FORCE INTERPOLATOR
# =====================================================================

def make_force_interpolator(
    data,
    cfg,
):
    """Build differentiable piecewise-linear F_hat(t) for PyTorch.

    DeepXDE evaluates the PDE residual at arbitrary collocation points,
    not only at the original CSV timestamps.

    Therefore this function linearly interpolates the approved equivalent
    force history at those points.

    The table itself is constant. The interpolation is piecewise linear in
    time and participates correctly in the PyTorch computation graph.

    No sine-wave approximation is introduced.
    """

    import torch

    t_numpy = np.asarray(
        data.t,
        dtype=np.float64,
    ).reshape(
        -1
    )

    f_hat_numpy = np.asarray(
        data.f_hat,
        dtype=np.float64,
    ).reshape(
        -1
    )

    if t_numpy.size < 2:

        raise ValueError(
            "Force interpolation needs at least two samples."
        )

    if t_numpy.size != f_hat_numpy.size:

        raise ValueError(
            "Force and time histories have different lengths."
        )

    dt = np.diff(
        t_numpy
    )

    if np.any(
        dt <= 0
    ):

        raise ValueError(
            "Force interpolation time vector must be strictly increasing."
        )

    if not np.all(
        np.isfinite(
            t_numpy
        )
    ):

        raise ValueError(
            "Force interpolation time contains NaN/Inf."
        )

    if not np.all(
        np.isfinite(
            f_hat_numpy
        )
    ):

        raise ValueError(
            "Scaled force history contains NaN/Inf."
        )

    expected_source = cfg[
        "physics"
    ].get(
        "force_source"
    )

    if expected_source != "equivalent_csv":

        raise ValueError(
            "Phase-2 residual requires "
            "physics.force_source='equivalent_csv'."
        )

    # Cache constant tables independently for each dtype/device pair.
    cache = {}

    def interpolator(
        t,
    ):
        """Return F_hat at tensor t with same shape/dtype/device."""

        if not torch.is_tensor(
            t
        ):

            raise TypeError(
                "PINN force interpolator expects a PyTorch tensor."
            )

        key = (
            str(
                t.device
            ),
            t.dtype,
        )

        if key not in cache:

            t_table = torch.as_tensor(
                t_numpy,
                dtype=t.dtype,
                device=t.device,
            )

            f_table = torch.as_tensor(
                f_hat_numpy,
                dtype=t.dtype,
                device=t.device,
            )

            cache[
                key
            ] = (
                t_table,
                f_table,
            )

        else:

            (
                t_table,
                f_table,
            ) = cache[
                key
            ]

        original_shape = (
            t.shape
        )

        x = t.reshape(
            -1
        )

        # DeepXDE geometry is constructed from the same data limits, so
        # points should already lie in range. Clamping only protects tiny
        # floating-point endpoint excursions.
        x = torch.clamp(
            x,
            min=t_table[0],
            max=t_table[-1],
        )

        # Find the right table interval for every query point.
        right = torch.searchsorted(
            t_table,
            x,
            right=True,
        )

        right = torch.clamp(
            right,
            min=1,
            max=t_table.numel() - 1,
        )

        left = (
            right
            -
            1
        )

        t0 = t_table[
            left
        ]

        t1 = t_table[
            right
        ]

        f0 = f_table[
            left
        ]

        f1 = f_table[
            right
        ]

        weight = (
            x
            -
            t0
        ) / (
            t1
            -
            t0
        )

        force = (
            f0
            +
            weight
            *
            (
                f1
                -
                f0
            )
        )

        return force.reshape(
            original_shape
        )

    return interpolator


# =====================================================================
# POSITIVITY PARAMETERISATION
# =====================================================================

def inverse_softplus(
    y,
):
    """Return x for which softplus(x)=y."""

    y = float(
        y
    )

    if y <= 0:

        raise ValueError(
            "softplus output must be positive, "
            f"got {y}"
        )

    if y > 30:

        return y

    return math.log(
        math.expm1(
            y
        )
    )


def read_parameter(
    raw,
    positive=True,
):
    """Read trained DeepXDE/PyTorch scalar parameter."""

    import torch

    value = raw.detach()

    if positive:

        value = torch.nn.functional.softplus(
            value
        )

    return float(
        value.cpu().numpy()
    )


# =====================================================================
# PINN PDE RESIDUAL
# =====================================================================

def make_residual(
    cfg,
    data,
    k_raw,
    c_raw,
    positive=True,
):
    """Build the DeepXDE Phase-2 SDOF residual.

    Scaled equation:

        m_hat*u_hat_tt
        + c_hat*u_hat_t
        + k_hat*u_hat
        - F_hat_eq(t)
        = 0

    The equivalent force comes from the frozen Phase-2 CSV history.
    """

    import torch
    import deepxde as dde

    m_hat = scaled_mass(
        cfg
    )

    force_at = make_force_interpolator(
        data,
        cfg,
    )

    if positive:

        def k_of(
            value,
        ):

            return torch.nn.functional.softplus(
                value
            )

        def c_of(
            value,
        ):

            return torch.nn.functional.softplus(
                value
            )

    else:

        def k_of(
            value,
        ):

            return value

        def c_of(
            value,
        ):

            return value

    def residual(
        t,
        u,
    ):

        du_dt = dde.grad.jacobian(
            u,
            t,
        )

        d2u_dt2 = dde.grad.hessian(
            u,
            t,
        )

        f_hat = force_at(
            t
        )

        return (
            m_hat
            *
            d2u_dt2

            +
            c_of(
                c_raw
            )
            *
            du_dt

            +
            k_of(
                k_raw
            )
            *
            u

            -
            f_hat
        )

    return residual


# =====================================================================
# CLASSICAL LEAST-SQUARES DIAGNOSTIC
# =====================================================================

def least_squares_reference(
    data,
    cfg,
):
    """Diagnostic SDOF fit using the approved equivalent force.

    This remains a secondary diagnostic.

    The principal validation for Phase 2 is PINN versus the explicitly
    selected ANSYS modal reference.

    Spectral derivatives are retained here for continuity with the Phase-1
    implementation. They can be sensitive to finite-record/end effects,
    especially for the higher-frequency Z case.
    """

    t = np.asarray(
        data.t,
        dtype=np.float64,
    ).reshape(
        -1
    )

    u = np.asarray(
        data.u,
        dtype=np.float64,
    ).reshape(
        -1
    )

    f = np.asarray(
        data.f,
        dtype=np.float64,
    ).reshape(
        -1
    )

    n = len(
        t
    )

    dt = float(
        data.dt
    )

    frequencies = np.fft.rfftfreq(
        n,
        dt,
    )

    jw = (
        2j
        *
        np.pi
        *
        frequencies
    )

    spectrum = np.fft.rfft(
        u
    )

    velocity = np.fft.irfft(
        spectrum
        *
        jw,
        n=n,
    )

    acceleration = np.fft.irfft(
        spectrum
        *
        jw ** 2,
        n=n,
    )

    # Spectral differentiation assumes periodic continuation.
    # Trim 5% from each edge to limit wrap-around contamination.
    edge = max(
        10,
        n // 20,
    )

    selected = slice(
        edge,
        -edge,
    )

    mass = float(
        cfg[
            "physics"
        ][
            "mass"
        ]
    )

    matrix = np.column_stack(
        [
            u[
                selected
            ],
            velocity[
                selected
            ],
        ]
    )

    rhs = (
        f[
            selected
        ]
        -
        mass
        *
        acceleration[
            selected
        ]
    )

    solution, *_ = np.linalg.lstsq(
        matrix,
        rhs,
        rcond=None,
    )

    stiffness = float(
        solution[
            0
        ]
    )

    damping = float(
        solution[
            1
        ]
    )

    residual = (
        rhs
        -
        matrix
        @
        solution
    )

    ss_res = float(
        residual
        @
        residual
    )

    ss_tot = float(
        np.sum(
            (
                rhs
                -
                rhs.mean()
            )
            ** 2
        )
    )

    r_squared = (
        1.0
        -
        ss_res
        /
        ss_tot

        if ss_tot > 0

        else float(
            "nan"
        )
    )

    dof = max(
        1,
        len(
            rhs
        )
        -
        2,
    )

    gram = (
        matrix.T
        @
        matrix
    )

    try:

        covariance = (
            ss_res
            /
            dof
        ) * np.linalg.inv(
            gram
        )

        standard_error = np.sqrt(
            np.diag(
                covariance
            )
        )

        k_stderr = float(
            standard_error[
                0
            ]
        )

        c_stderr = float(
            standard_error[
                1
            ]
        )

    except np.linalg.LinAlgError:

        k_stderr = float(
            "nan"
        )

        c_stderr = float(
            "nan"
        )

    result = {
        "K":
            stiffness,

        "C":
            damping,

        "r_squared":
            r_squared,

        "K_stderr":
            k_stderr,

        "C_stderr":
            c_stderr,

        "mass_used":
            mass,
    }

    result.update(
        derived(
            stiffness,
            damping,
            mass,
        )
    )

    return result


# =====================================================================
# PHYSICS SMOKE TEST
# =====================================================================

def _run_smoke_test():
    """CPU-only verification of force interpolation and scaling."""

    from .config import load_config
    from .dataio import load

    cfg = load_config()

    data = load(
        cfg
    )

    direction = cfg[
        "active_direction"
    ]

    print(
        "=" * 88
    )

    print(
        f"PHASE-2 PHYSICS CHECK — {direction}"
    )

    print(
        "=" * 88
    )

    # ---------------------------------------------------------------
    # Physical/scaled constants
    # ---------------------------------------------------------------

    mass = float(
        cfg[
            "physics"
        ][
            "mass"
        ]
    )

    reference_stiffness = float(
        cfg[
            "physics"
        ][
            "reference_stiffness_N_per_m"
        ]
    )

    k_hat_reference, _ = to_scaled(
        reference_stiffness,
        0.0,
        cfg,
    )

    m_hat = scaled_mass(
        cfg
    )

    print(
        f"equivalent mass       : "
        f"{mass:.12f} kg"
    )

    print(
        f"scaled mass m_hat     : "
        f"{m_hat:.12e}"
    )

    print(
        f"reference K           : "
        f"{reference_stiffness:.12e} N/m"
    )

    print(
        f"reference k_hat       : "
        f"{k_hat_reference:.12e}"
    )

    # ---------------------------------------------------------------
    # NumPy interpolation at every original CSV timestamp.
    # This MUST reproduce the original equivalent-force vector exactly
    # to floating-point precision.
    # ---------------------------------------------------------------

    reconstructed = force_physical(
        data.t.reshape(
            -1
        ),
        data,
    )

    original = data.f.reshape(
        -1
    )

    maximum_error = float(
        np.max(
            np.abs(
                reconstructed
                -
                original
            )
        )
    )

    print(
        f"CSV-node interpolation: "
        f"max error {maximum_error:.6e} N"
    )

    if maximum_error > 1.0e-9:

        raise RuntimeError(
            "Physical-force interpolation failed "
            "at original CSV timestamps."
        )

    # ---------------------------------------------------------------
    # Midpoint interpolation cross-check.
    # ---------------------------------------------------------------

    midpoint_times = (
        0.5
        *
        (
            data.t[:-1, 0]
            +
            data.t[1:, 0]
        )
    )

    midpoint_numpy = np.interp(
        midpoint_times,
        data.t[:, 0],
        data.f[:, 0],
    )

    midpoint_function = force_physical(
        midpoint_times,
        data,
    )

    midpoint_error = float(
        np.max(
            np.abs(
                midpoint_function
                -
                midpoint_numpy
            )
        )
    )

    print(
        f"midpoint interpolation: "
        f"max error {midpoint_error:.6e} N"
    )

    if midpoint_error > 1.0e-9:

        raise RuntimeError(
            "Midpoint force interpolation failed."
        )

    # ---------------------------------------------------------------
    # PyTorch interpolation check.
    # ---------------------------------------------------------------

    import torch

    force_at = make_force_interpolator(
        data,
        cfg,
    )

    test_indices = np.linspace(
        0,
        data.n - 1,
        101,
        dtype=int,
    )

    t_test = torch.tensor(
        data.t[
            test_indices
        ],
        dtype=torch.float32,
    )

    torch_force = (
        force_at(
            t_test
        )
        .detach()
        .cpu()
        .numpy()
        .reshape(
            -1
        )
    )

    expected_scaled_force = (
        data.f_hat[
            test_indices
        ]
        .reshape(
            -1
        )
    )

    torch_error = float(
        np.max(
            np.abs(
                torch_force
                -
                expected_scaled_force
            )
        )
    )

    print(
        f"PyTorch interpolation : "
        f"max scaled error {torch_error:.6e}"
    )

    # float32 interpolation tolerance.
    if torch_error > 5.0e-5:

        raise RuntimeError(
            "PyTorch equivalent-force interpolation "
            "failed smoke test."
        )

    # ---------------------------------------------------------------
    # Scaling round trip.
    # ---------------------------------------------------------------

    k_roundtrip, _ = to_physical(
        k_hat_reference,
        0.0,
        cfg,
    )

    relative_roundtrip_error = abs(
        k_roundtrip
        -
        reference_stiffness
    ) / reference_stiffness

    print(
        f"K scaling round-trip  : "
        f"relative error "
        f"{relative_roundtrip_error:.6e}"
    )

    if relative_roundtrip_error > 1.0e-12:

        raise RuntimeError(
            "Stiffness scaling round-trip failed."
        )

    print(
        f"force source          : "
        f"{cfg['physics']['force_source']}"
    )

    print(
        "physics equation      : "
        "m_hat*u_tt + c_hat*u_t + "
        "k_hat*u - F_hat_eq(t) = 0"
    )

    print(
        "PHYSICS INTERFACE     : PASS"
    )

    print(
        "=" * 88
    )


if __name__ == "__main__":

    _run_smoke_test()