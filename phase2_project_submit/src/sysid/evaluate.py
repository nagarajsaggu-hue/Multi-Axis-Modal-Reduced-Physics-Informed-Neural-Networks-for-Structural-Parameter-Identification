"""Phase-2 evaluation for PINN SDOF system identification.

This module compares the final PINN result against:

1. The authoritative ANSYS modal reference stored in config.yaml.
2. A transient least-squares diagnostic using the approved equivalent force,
   displacement, ANSYS velocity and ANSYS acceleration.

Important
---------
The ANSYS modal reference is NOT selected heuristically.

For Phase 2 the reference is explicit:

    X -> S_C07 / Node 3461 / Mode 1
    Y -> S_C07 / Node 3461 / Mode 2
    Z -> S_C07 / Node 3461 / Mode 4

The least-squares result is diagnostic only because it uses ANSYS-exported
velocity and acceleration directly. The PINN itself is displacement-driven and
obtains derivatives through automatic differentiation.

The reduced equation used consistently throughout evaluation is

    m* u_ddot + C u_dot + K u = F_eq(t)

where F_eq(t) is the approved reference-coordinate equivalent-force history.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


# =====================================================================
# SMALL NUMERICAL HELPERS
# =====================================================================

def _percent_difference(
    value,
    reference,
):
    """Signed percentage difference relative to reference."""

    if reference is None:
        return None

    reference = float(
        reference
    )

    if reference == 0.0:
        return None

    return (
        100.0
        *
        (
            float(
                value
            )
            -
            reference
        )
        /
        reference
    )


def _ratio(
    value,
    reference,
):
    """Return value/reference where meaningful."""

    if reference is None:
        return None

    reference = float(
        reference
    )

    if reference == 0.0:
        return None

    return (
        float(
            value
        )
        /
        reference
    )


def _derived(
    stiffness,
    damping,
    mass,
):
    """Compute natural frequency and damping ratio."""

    stiffness = float(
        stiffness
    )

    damping = float(
        damping
    )

    mass = float(
        mass
    )

    if (
        stiffness <= 0.0
        or
        mass <= 0.0
    ):

        return {
            "omega_n_rad_s":
                float(
                    "nan"
                ),

            "f_n_hz":
                float(
                    "nan"
                ),

            "zeta":
                float(
                    "nan"
                ),
        }

    omega_n = math.sqrt(
        stiffness
        /
        mass
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
            damping
            /
            (
                2.0
                *
                math.sqrt(
                    stiffness
                    *
                    mass
                )
            ),
    }


# =====================================================================
# STRICT PHASE-2 RESULT VALIDATION
# =====================================================================

def _validate_identified_result(
    cfg,
    identified,
    tag,
):
    """Ensure the result being evaluated belongs to the active final case."""

    direction = cfg[
        "active_direction"
    ]

    if identified.get(
        "phase"
    ) != "phase2":

        raise ValueError(
            f"identified_{tag}.json is not a Phase-2 result."
        )

    result_direction = identified.get(
        "direction"
    )

    if result_direction != direction:

        raise ValueError(
            f"Result direction {result_direction!r} does not match "
            f"active_direction {direction!r}."
        )

    expected_iterations = int(
        cfg[
            "training"
        ][
            "iterations"
        ]
    )

    actual_iterations = int(
        identified.get(
            "run",
            {},
        ).get(
            "iterations",
            -1,
        )
    )

    if actual_iterations != expected_iterations:

        raise ValueError(
            f"Result '{tag}' used {actual_iterations} iterations, "
            f"but the FINAL validated {direction} configuration requires "
            f"{expected_iterations}. "
            "Evaluate the validated production result instead."
        )

    cfg_reference = cfg[
        "reference"
    ]

    result_reference = identified.get(
        "reference",
        {},
    )

    if result_reference:

        if (
            result_reference.get(
                "sensor"
            )
            !=
            cfg_reference[
                "sensor"
            ]
        ):

            raise ValueError(
                "Result reference sensor does not match config."
            )

        if (
            int(
                result_reference.get(
                    "node_id",
                    -1,
                )
            )
            !=
            int(
                cfg_reference[
                    "node_id"
                ]
            )
        ):

            raise ValueError(
                "Result reference node does not match config."
            )

        if (
            int(
                result_reference.get(
                    "mode",
                    -1,
                )
            )
            !=
            int(
                cfg_reference[
                    "mode"
                ]
            )
        ):

            raise ValueError(
                "Result reference mode does not match config."
            )

    fixed_inputs = identified.get(
        "fixed_inputs",
        {},
    )

    if fixed_inputs:

        if (
            fixed_inputs.get(
                "force_source"
            )
            !=
            "equivalent_csv"
        ):

            raise ValueError(
                "Evaluated Phase-2 result did not use equivalent_csv force."
            )

        result_mass = fixed_inputs.get(
            "mass_kg"
        )

        if result_mass is not None:

            expected_mass = float(
                cfg[
                    "physics"
                ][
                    "mass"
                ]
            )

            if not np.isclose(
                float(
                    result_mass
                ),
                expected_mass,
                rtol=1.0e-10,
                atol=1.0e-8,
            ):

                raise ValueError(
                    "Result equivalent mass does not match active config."
                )


# =====================================================================
# AUTHORITATIVE ANSYS MODAL REFERENCE
# =====================================================================

def _ansys_reference(
    cfg,
):
    """Build the explicit Phase-2 ANSYS reference row."""

    reference = cfg[
        "reference"
    ]

    return {
        "method":
            "ANSYS modal",

        "K_N_per_m":
            float(
                reference[
                    "equivalent_stiffness_N_per_m"
                ]
            ),

        "C_N_s_per_m":
            None,

        "f_n_hz":
            float(
                reference[
                    "frequency_hz"
                ]
            ),

        "zeta":
            None,

        "mode":
            int(
                reference[
                    "mode"
                ]
            ),

        "mass_kg":
            float(
                reference[
                    "equivalent_mass_kg"
                ]
            ),

        "phi_reference":
            float(
                reference[
                    "phi_reference"
                ]
            ),

        "sensor":
            reference[
                "sensor"
            ],

        "node_id":
            int(
                reference[
                    "node_id"
                ]
            ),

        "note":
            (
                f"{reference['sensor']} / "
                f"Node {reference['node_id']} / "
                f"Mode {reference['mode']} "
                "(authoritative Phase-2 reference)"
            ),
    }


# =====================================================================
# TRANSIENT LEAST-SQUARES DIAGNOSTIC
# =====================================================================

def _least_squares_diagnostic(
    cfg,
    data,
):
    """Estimate K and C from the transient ANSYS histories.

    Uses

        F_eq - m*a = K*u + C*v

    with the approved equivalent-force history.

    This remains a diagnostic rather than the main identification result
    because v and a are ANSYS-exported response quantities, whereas the PINN
    obtains derivatives from its displacement network using AD.
    """

    mass = float(
        cfg[
            "physics"
        ][
            "mass"
        ]
    )

    u = np.asarray(
        data.u,
        dtype=np.float64,
    ).reshape(
        -1
    )

    v = np.asarray(
        data.v,
        dtype=np.float64,
    ).reshape(
        -1
    )

    a = np.asarray(
        data.a,
        dtype=np.float64,
    ).reshape(
        -1
    )

    force = np.asarray(
        data.f,
        dtype=np.float64,
    ).reshape(
        -1
    )

    n = len(
        u
    )

    if not (
        len(
            v
        )
        ==
        len(
            a
        )
        ==
        len(
            force
        )
        ==
        n
    ):

        raise ValueError(
            "Least-squares diagnostic received inconsistent array lengths."
        )

    if n < 50:

        raise ValueError(
            "Too few samples for least-squares diagnostic."
        )

    if not all(
        np.all(
            np.isfinite(
                x
            )
        )
        for x
        in (
            u,
            v,
            a,
            force,
        )
    ):

        raise ValueError(
            "Least-squares diagnostic received NaN/Inf."
        )

    rhs = (
        force
        -
        mass
        *
        a
    )

    design = np.column_stack(
        (
            u,
            v,
        )
    )

    coefficients, residuals, rank, singular_values = np.linalg.lstsq(
        design,
        rhs,
        rcond=None,
    )

    stiffness = float(
        coefficients[
            0
        ]
    )

    damping = float(
        coefficients[
            1
        ]
    )

    reconstructed_force = (
        mass
        *
        a
        +
        stiffness
        *
        u
        +
        damping
        *
        v
    )

    force_error = (
        reconstructed_force
        -
        force
    )

    rmse_force = float(
        np.sqrt(
            np.mean(
                force_error ** 2
            )
        )
    )

    force_rms = float(
        np.sqrt(
            np.mean(
                force ** 2
            )
        )
    )

    denominator = float(
        np.sum(
            (
                force
                -
                force.mean()
            )
            ** 2
        )
    )

    r_squared = (
        1.0
        -
        float(
            np.sum(
                force_error ** 2
            )
        )
        /
        denominator

        if denominator > 0.0

        else float(
            "nan"
        )
    )

    derived = _derived(
        stiffness,
        damping,
        mass,
    )

    condition_number = (
        float(
            singular_values[
                0
            ]
            /
            singular_values[
                -1
            ]
        )

        if (
            len(
                singular_values
            )
            >=
            2
            and
            singular_values[
                -1
            ]
            >
            0.0
        )

        else float(
            "inf"
        )
    )

    return {
        "K":
            stiffness,

        "C":
            damping,

        "f_n_hz":
            derived[
                "f_n_hz"
            ],

        "zeta":
            derived[
                "zeta"
            ],

        "omega_n_rad_s":
            derived[
                "omega_n_rad_s"
            ],

        "r_squared":
            r_squared,

        "force_rmse_N":
            rmse_force,

        "force_relative_rmse":
            (
                rmse_force
                /
                force_rms

                if force_rms > 0.0

                else float(
                    "nan"
                )
            ),

        "rank":
            int(
                rank
            ),

        "condition_number":
            condition_number,

        "n_samples":
            int(
                n
            ),

        "force_source":
            "equivalent_csv",

        "force_column":
            data.columns[
                "equivalent_force"
            ],

        "note":
            (
                "Diagnostic only: uses ANSYS-exported velocity and "
                "acceleration with the approved equivalent-force history."
            ),
    }


# =====================================================================
# COMPARISON ASSEMBLY
# =====================================================================

def build_table(
    cfg,
    data,
    identified,
):
    """Assemble Phase-2 PINN / LS / ANSYS comparison."""

    mass = float(
        cfg[
            "physics"
        ][
            "mass"
        ]
    )

    stiffness_pinn = float(
        identified[
            "identified"
        ][
            "K_N_per_m"
        ]
    )

    damping_pinn = float(
        identified[
            "identified"
        ][
            "C_N_s_per_m"
        ]
    )

    derived_pinn = identified[
        "derived"
    ]

    pinn_row = {
        "method":
            "PINN",

        "K_N_per_m":
            stiffness_pinn,

        "C_N_s_per_m":
            damping_pinn,

        "f_n_hz":
            float(
                derived_pinn[
                    "f_n_hz"
                ]
            ),

        "zeta":
            float(
                derived_pinn[
                    "zeta"
                ]
            ),

        "note":
            (
                f"{identified['run']['iterations']} iterations, "
                "equivalent-force PINN"
            ),
    }

    ls = _least_squares_diagnostic(
        cfg,
        data,
    )

    ls_row = {
        "method":
            "Least squares",

        "K_N_per_m":
            ls[
                "K"
            ],

        "C_N_s_per_m":
            ls[
                "C"
            ],

        "f_n_hz":
            ls[
                "f_n_hz"
            ],

        "zeta":
            ls[
                "zeta"
            ],

        "note":
            (
                f"diagnostic, force R2 = "
                f"{ls['r_squared']:.6f}"
            ),
    }

    ansys_row = _ansys_reference(
        cfg
    )

    stiffness_ansys = float(
        ansys_row[
            "K_N_per_m"
        ]
    )

    frequency_ansys = float(
        ansys_row[
            "f_n_hz"
        ]
    )

    comparison = {
        "phase":
            "phase2",

        "direction":
            cfg[
                "active_direction"
            ],

        "reference_coordinate": {
            "sensor":
                cfg[
                    "reference"
                ][
                    "sensor"
                ],

            "node_id":
                int(
                    cfg[
                        "reference"
                    ][
                        "node_id"
                    ]
                ),

            "mode":
                int(
                    cfg[
                        "reference"
                    ][
                        "mode"
                    ]
                ),

            "mass_kg":
                mass,
        },

        "force": {
            "source":
                cfg[
                    "physics"
                ][
                    "force_source"
                ],

            "column":
                data.columns[
                    "equivalent_force"
                ],

            "peak_N":
                float(
                    np.max(
                        np.abs(
                            data.f
                        )
                    )
                ),
        },

        "rows": [
            pinn_row,
            ls_row,
            ansys_row,
        ],

        "fit":
            identified.get(
                "fit",
                {},
            ),

        "validation":
            identified.get(
                "validation",
                {},
            ),

        "least_squares_diagnostic":
            ls,

        "agreement": {
            "pinn_vs_ansys_modal": {
                "K_ratio":
                    _ratio(
                        stiffness_pinn,
                        stiffness_ansys,
                    ),

                "K_percent_diff":
                    _percent_difference(
                        stiffness_pinn,
                        stiffness_ansys,
                    ),

                "f_n_ratio":
                    _ratio(
                        derived_pinn[
                            "f_n_hz"
                        ],
                        frequency_ansys,
                    ),

                "f_n_percent_diff":
                    _percent_difference(
                        derived_pinn[
                            "f_n_hz"
                        ],
                        frequency_ansys,
                    ),
            },

            "pinn_vs_least_squares": {
                "K_ratio":
                    _ratio(
                        stiffness_pinn,
                        ls[
                            "K"
                        ],
                    ),

                "K_percent_diff":
                    _percent_difference(
                        stiffness_pinn,
                        ls[
                            "K"
                        ],
                    ),

                "C_ratio":
                    _ratio(
                        damping_pinn,
                        ls[
                            "C"
                        ],
                    ),

                "C_percent_diff":
                    _percent_difference(
                        damping_pinn,
                        ls[
                            "C"
                        ],
                    ),
            },
        },
    }

    return comparison


# =====================================================================
# HUMAN-READABLE REPORT
# =====================================================================

def format_table(
    comparison,
) -> str:
    """Create concise text report."""

    lines = []

    direction = comparison[
        "direction"
    ]

    reference = comparison[
        "reference_coordinate"
    ]

    lines.append(
        "="
        *
        96
    )

    lines.append(
        f"PHASE-2 EVALUATION — {direction}"
    )

    lines.append(
        "="
        *
        96
    )

    lines.append(
        (
            f"Reference coordinate : "
            f"{reference['sensor']} / "
            f"Node {reference['node_id']} / "
            f"Mode {reference['mode']}"
        )
    )

    lines.append(
        (
            f"Equivalent mass      : "
            f"{reference['mass_kg']:.9f} kg"
        )
    )

    lines.append(
        (
            f"Physics force        : "
            f"{comparison['force']['column']} "
            f"(peak {comparison['force']['peak_N']:.6f} N)"
        )
    )

    lines.append(
        ""
    )

    header = (
        f"{'method':<17}"
        f"{'K (N/m)':>16}"
        f"{'C (N.s/m)':>16}"
        f"{'f_n (Hz)':>13}"
        f"{'zeta (%)':>13}   "
        f"note"
    )

    lines.append(
        header
    )

    lines.append(
        "-"
        *
        len(
            header
        )
    )

    for row in comparison[
        "rows"
    ]:

        stiffness = (
            f"{row['K_N_per_m']:.6e}"

            if row[
                "K_N_per_m"
            ]
            is not None

            else "-"
        )

        damping = (
            f"{row['C_N_s_per_m']:.6e}"

            if row[
                "C_N_s_per_m"
            ]
            is not None

            else "-"
        )

        frequency = (
            f"{row['f_n_hz']:.6f}"

            if row[
                "f_n_hz"
            ]
            is not None

            else "-"
        )

        zeta = (
            f"{row['zeta'] * 100.0:.8f}"

            if row[
                "zeta"
            ]
            is not None

            else "-"
        )

        lines.append(
            f"{row['method']:<17}"
            f"{stiffness:>16}"
            f"{damping:>16}"
            f"{frequency:>13}"
            f"{zeta:>13}   "
            f"{row['note']}"
        )

    lines.append(
        ""
    )

    ansys_agreement = comparison[
        "agreement"
    ][
        "pinn_vs_ansys_modal"
    ]

    lines.append(
        (
            "PINN vs ANSYS modal:"
        )
    )

    lines.append(
        (
            f"  K difference       : "
            f"{ansys_agreement['K_percent_diff']:+.6f} %"
        )
    )

    lines.append(
        (
            f"  frequency difference: "
            f"{ansys_agreement['f_n_percent_diff']:+.6f} %"
        )
    )

    validation = comparison.get(
        "validation",
        {}
    )

    if validation:

        lines.append(
            ""
        )

        lines.append(
            "PINN holdout validation:"
        )

        lines.append(
            (
                f"  RMSE               : "
                f"{validation['rmse_holdout_m']:.6e} m"
            )
        )

        lines.append(
            (
                f"  relative RMSE      : "
                f"{validation['relative_rmse_holdout'] * 100.0:.6f} %"
            )
        )

        lines.append(
            (
                f"  R2                 : "
                f"{validation['r_squared_holdout']:.9f}"
            )
        )

    ls = comparison[
        "least_squares_diagnostic"
    ]

    lines.append(
        ""
    )

    lines.append(
        "Transient least-squares diagnostic:"
    )

    lines.append(
        (
            f"  force R2           : "
            f"{ls['r_squared']:.9f}"
        )
    )

    lines.append(
        (
            f"  relative force RMSE: "
            f"{ls['force_relative_rmse'] * 100.0:.6f} %"
        )
    )

    lines.append(
        (
            f"  design condition   : "
            f"{ls['condition_number']:.6e}"
        )
    )

    lines.append(
        (
            "  interpretation     : diagnostic only; ANSYS-exported "
            "velocity/acceleration are not PINN training targets."
        )
    )

    lines.append(
        "="
        *
        96
    )

    return "\n".join(
        lines
    )


# =====================================================================
# PUBLIC RUN ENTRY POINT
# =====================================================================

def run(
    cfg,
    data,
    tag="run",
):
    """Evaluate one final Phase-2 identified result."""

    out_dir = Path(
        cfg[
            "output"
        ][
            "dir"
        ]
    )

    identified_path = (
        out_dir
        /
        f"identified_{tag}.json"
    )

    if not identified_path.exists():

        raise FileNotFoundError(
            f"No identified result found at {identified_path}. "
            f"Check active_direction and tag."
        )

    with identified_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        identified = json.load(
            file
        )

    _validate_identified_result(
        cfg,
        identified,
        tag,
    )

    comparison = build_table(
        cfg,
        data,
        identified,
    )

    comparison_json = (
        out_dir
        /
        f"comparison_{tag}.json"
    )

    with comparison_json.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            comparison,
            file,
            indent=2,
        )

    text = format_table(
        comparison
    )

    comparison_txt = (
        out_dir
        /
        f"comparison_{tag}.txt"
    )

    with comparison_txt.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            text
            +
            "\n"
        )

    print(
        text
    )

    print()
    print(
        f"Wrote {comparison_json}"
    )

    print(
        f"Wrote {comparison_txt}"
    )

    return comparison