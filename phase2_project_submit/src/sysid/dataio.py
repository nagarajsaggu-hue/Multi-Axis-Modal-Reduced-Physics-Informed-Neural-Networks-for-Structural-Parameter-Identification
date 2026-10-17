"""Strict Phase-2 data loading and validation.

The Phase-2 CSVs contain:

- reference-coordinate displacement
    -> primary PINN observation

- ANSYS velocity and acceleration
    -> diagnostics only

- raw ANSYS face force
    -> provenance / QC only

- equivalent reference-coordinate force
    -> the ONLY valid PINN physics RHS

This module validates the approved dataset and QC files before training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .config import load_json


@dataclass
class Dataset:
    """Validated Phase-2 record for one active direction."""

    direction: str

    # ---------------------------------------------------------------
    # RESPONSE
    # ---------------------------------------------------------------

    t: np.ndarray

    u: np.ndarray
    u_hat: np.ndarray

    # ANSYS diagnostic arrays.
    # They are NOT supervised PINN targets.
    v: np.ndarray
    a: np.ndarray

    # ---------------------------------------------------------------
    # FORCE
    # ---------------------------------------------------------------

    # Approved equivalent/reference-coordinate force.
    f: np.ndarray
    f_hat: np.ndarray

    # Original ANSYS resultant face force.
    # Retained only for traceability / validation.
    raw_f: np.ndarray

    # ---------------------------------------------------------------
    # RECORD INFORMATION
    # ---------------------------------------------------------------

    dt: float

    meta: dict
    modal: dict
    reference: dict

    preprocessing_qc: dict
    force_qc: dict

    columns: Dict[str, str]

    @property
    def n(
        self,
    ) -> int:

        return len(
            self.t
        )


# =====================================================================
# JSON HELPERS
# =====================================================================

def _require_json(
    path,
    label,
):

    payload = load_json(
        path
    )

    if payload is None:

        raise FileNotFoundError(

            f"Required {label} JSON not found:\n"
            f"{path}"
        )

    return payload


# =====================================================================
# CSV COLUMN VALIDATION
# =====================================================================

def _require_columns(
    df: pd.DataFrame,
    required: Dict[str, str],
) -> None:

    missing = [

        column

        for column
        in required.values()

        if column not in df.columns
    ]

    if missing:

        raise KeyError(

            "Required CSV columns are missing:\n"

            +
            "\n".join(

                f"  - {name}"

                for name
                in missing
            )

            +
            "\n\nAvailable columns:\n"

            +
            "\n".join(

                f"  - {name}"

                for name
                in df.columns
            )
        )


def _as_float_array(
    df: pd.DataFrame,
    column: str,
) -> np.ndarray:

    try:

        values = pd.to_numeric(

            df[
                column
            ],

            errors="raise",

        ).to_numpy(
            dtype=np.float64
        )

    except Exception as exc:

        raise ValueError(

            f"Column {column!r} "
            "is not clean numeric data."
        ) from exc

    if values.ndim != 1:

        raise ValueError(

            f"Column {column!r} "
            "did not produce a 1-D array."
        )

    if not np.all(
        np.isfinite(
            values
        )
    ):

        bad = np.where(

            ~np.isfinite(
                values
            )

        )[0][:10]

        raise ValueError(

            f"Column {column!r} contains "
            f"NaN/Inf at rows {bad.tolist()}."
        )

    return values


# =====================================================================
# NUMERICAL CHECK
# =====================================================================

def _assert_close(
    name,
    actual,
    expected,
    *,
    rtol=1.0e-9,
    atol=1.0e-12,
):

    if not np.isclose(

        float(
            actual
        ),

        float(
            expected
        ),

        rtol=rtol,
        atol=atol,

    ):

        raise ValueError(

            f"{name} mismatch: "
            f"actual={actual!r}, "
            f"expected={expected!r}"
        )


# =====================================================================
# SYNTHESIZED MODAL REFERENCE
# =====================================================================

def _build_modal_reference(
    reference: dict,
) -> dict:
    """Create the selected modal structure expected by legacy consumers.

    This prevents evaluation/visualization code from having to infer the
    reference mode from local modal amplitudes.

    The authoritative reference is already known from Phase-2 preprocessing.
    """

    selected = {

        "mode":
            int(
                reference[
                    "mode"
                ]
            ),

        "frequency_hz":
            float(
                reference[
                    "frequency_hz"
                ]
            ),

        "generalized_mass_kg":
            float(
                reference[
                    "equivalent_mass_kg"
                ]
            ),

        "generalized_stiffness_N_per_m":
            float(
                reference[
                    "equivalent_stiffness_N_per_m"
                ]
            ),

        "phi_reference":
            float(
                reference[
                    "phi_reference"
                ]
            ),
    }

    return {

        "selected_reference_mode":
            selected,

        "modes": [
            selected
        ],
    }


# =====================================================================
# PREPROCESSING QC VALIDATION
# =====================================================================

def _validate_preprocessing_qc(
    cfg,
    preprocessing_qc,
):

    direction = cfg[
        "active_direction"
    ]

    reference = cfg[
        "reference"
    ]

    quality = cfg[
        "quality"
    ]

    # ---------------------------------------------------------------
    # GLOBAL QC
    # ---------------------------------------------------------------

    if int(
        preprocessing_qc[
            "n_samples"
        ]
    ) != int(
        quality[
            "expected_n_samples"
        ]
    ):

        raise ValueError(

            "Preprocessing QC sample count does not "
            "match the approved configuration."
        )

    _assert_close(

        "Preprocessing QC dt",

        preprocessing_qc[
            "dt_seconds"
        ],

        quality[
            "expected_dt_seconds"
        ],

        rtol=1.0e-9,

        atol=quality[
            "time_atol_seconds"
        ],
    )

    if (
        preprocessing_qc.get(
            "reference_sensor"
        )
        !=
        reference[
            "sensor"
        ]
    ):

        raise ValueError(

            "Preprocessing QC reference sensor does "
            "not match config reference sensor."
        )

    # ---------------------------------------------------------------
    # DIRECTION-SPECIFIC QC
    # ---------------------------------------------------------------

    block = preprocessing_qc[
        direction
    ]

    if int(
        block[
            "target_mode"
        ]
    ) != int(
        reference[
            "mode"
        ]
    ):

        raise ValueError(

            f"Preprocessing QC target mode for "
            f"{direction} does not match config."
        )

    if (
        block.get(
            "reference_sensor"
        )
        !=
        reference[
            "sensor"
        ]
    ):

        raise ValueError(

            f"Preprocessing QC reference sensor for "
            f"{direction} does not match config."
        )

    phi_key = (

        "phi4_reference"

        if direction == "Z"

        else "reference_phi"
    )

    checks = {

        "reference phi": (

            block[
                phi_key
            ],

            reference[
                "phi_reference"
            ],
        ),

        "reference frequency": (

            block[
                "frequency_hz"
            ],

            reference[
                "frequency_hz"
            ],
        ),

        "equivalent mass": (

            block[
                "equivalent_mass_kg"
            ],

            reference[
                "equivalent_mass_kg"
            ],
        ),

        "equivalent stiffness": (

            block[
                "equivalent_stiffness_N_per_m"
            ],

            reference[
                "equivalent_stiffness_N_per_m"
            ],
        ),
    }

    for label, (
        actual,
        expected,
    ) in checks.items():

        _assert_close(

            f"{direction} {label}",

            actual,
            expected,

            rtol=quality[
                "scalar_rtol"
            ],

            atol=1.0e-12,
        )


# =====================================================================
# FORCE-PROJECTION QC VALIDATION
# =====================================================================

def _validate_force_qc(
    cfg,
    force_qc,
    raw_force: np.ndarray,
    equivalent_force: np.ndarray,
):

    direction = cfg[
        "active_direction"
    ]

    reference = cfg[
        "reference"
    ]

    quality = cfg[
        "quality"
    ]

    if direction not in force_qc.get(
        "directions",
        {},
    ):

        raise KeyError(

            f"Force-projection QC has "
            f"no {direction} block."
        )

    block = force_qc[
        "directions"
    ][
        direction
    ]

    # ---------------------------------------------------------------
    # MODE
    # ---------------------------------------------------------------

    if int(
        block[
            "target_mode"
        ]
    ) != int(
        reference[
            "mode"
        ]
    ):

        raise ValueError(

            f"Force-projection QC target mode for "
            f"{direction} does not match config."
        )

    # ---------------------------------------------------------------
    # REFERENCE PHI
    # ---------------------------------------------------------------

    _assert_close(

        f"{direction} force-QC phi",

        block[
            "phi_reference"
        ],

        reference[
            "phi_reference"
        ],

        rtol=quality[
            "scalar_rtol"
        ],

        atol=1.0e-12,
    )

    # ---------------------------------------------------------------
    # COMPLETE FORCE-HISTORY CHECK
    #
    # For this fixed spatial load:
    #
    #     F_eq(t) = scale * F_raw(t)
    #
    # The approved scale comes from exact FE modal projection.
    # ---------------------------------------------------------------

    scale = float(

        block[
            "equivalent_force_per_raw_force"
        ]
    )

    predicted = (

        raw_force
        *
        scale
    )

    allowed = max(

        float(
            quality[
                "force_atol_N"
            ]
        ),

        float(
            quality[
                "scalar_rtol"
            ]
        )
        *
        max(

            1.0,

            float(
                np.max(
                    np.abs(
                        equivalent_force
                    )
                )
            ),
        ),
    )

    max_abs_error = float(

        np.max(

            np.abs(

                equivalent_force
                -
                predicted
            )
        )
    )

    if max_abs_error > allowed:

        raise ValueError(

            f"{direction} equivalent-force history "
            "does not match the approved "
            "force-projection scale.\n"

            f"max error = {max_abs_error:.6e} N\n"
            f"allowed   = {allowed:.6e} N"
        )

    # ---------------------------------------------------------------
    # PEAK CHECKS
    # ---------------------------------------------------------------

    _assert_close(

        f"{direction} raw force peak",

        np.max(
            np.abs(
                raw_force
            )
        ),

        block[
            "raw_force_peak_N"
        ],

        rtol=quality[
            "scalar_rtol"
        ],

        atol=quality[
            "force_atol_N"
        ],
    )

    _assert_close(

        f"{direction} equivalent force peak",

        np.max(
            np.abs(
                equivalent_force
            )
        ),

        block[
            "equivalent_force_peak_N"
        ],

        rtol=quality[
            "scalar_rtol"
        ],

        atol=quality[
            "force_atol_N"
        ],
    )


# =====================================================================
# MAIN LOADER
# =====================================================================

def load(
    cfg,
) -> Dataset:
    """Load and strictly validate the active Phase-2 dataset."""

    direction = cfg[
        "active_direction"
    ]

    data_cfg = cfg[
        "data"
    ]

    quality = cfg[
        "quality"
    ]

    csv_path = data_cfg[
        "csv"
    ]

    df = pd.read_csv(
        csv_path
    )

    # ---------------------------------------------------------------
    # EXACT COLUMN CONTRACT
    # ---------------------------------------------------------------

    columns = {

        "time":
            data_cfg[
                "time_column"
            ],

        "observation":
            data_cfg[
                "observation_column"
            ],

        "velocity":
            data_cfg[
                "velocity_column"
            ],

        "acceleration":
            data_cfg[
                "acceleration_column"
            ],

        "raw_force":
            data_cfg[
                "raw_force_column"
            ],

        "equivalent_force":
            data_cfg[
                "force_column"
            ],
    }

    _require_columns(
        df,
        columns,
    )

    # ---------------------------------------------------------------
    # LOAD FULL APPROVED RECORD
    # ---------------------------------------------------------------

    t_full = _as_float_array(

        df,
        columns[
            "time"
        ],
    )

    u_full = _as_float_array(

        df,
        columns[
            "observation"
        ],
    )

    v_full = _as_float_array(

        df,
        columns[
            "velocity"
        ],
    )

    a_full = _as_float_array(

        df,
        columns[
            "acceleration"
        ],
    )

    raw_f_full = _as_float_array(

        df,
        columns[
            "raw_force"
        ],
    )

    f_full = _as_float_array(

        df,
        columns[
            "equivalent_force"
        ],
    )

    # ---------------------------------------------------------------
    # LENGTH
    # ---------------------------------------------------------------

    expected_n = int(

        quality[
            "expected_n_samples"
        ]
    )

    if len(
        t_full
    ) != expected_n:

        raise ValueError(

            f"{direction}: expected "
            f"{expected_n} samples, "
            f"found {len(t_full)}."
        )

    if not (

        len(t_full)
        ==
        len(u_full)
        ==
        len(v_full)
        ==
        len(a_full)
        ==
        len(raw_f_full)
        ==
        len(f_full)

    ):

        raise ValueError(

            "Phase-2 arrays do not have "
            "identical lengths."
        )

    # ---------------------------------------------------------------
    # TIME GRID
    # ---------------------------------------------------------------

    dt_all = np.diff(
        t_full
    )

    if np.any(
        dt_all <= 0
    ):

        raise ValueError(

            "Time must be strictly increasing."
        )

    dt = float(

        np.median(
            dt_all
        )
    )

    expected_dt = float(

        quality[
            "expected_dt_seconds"
        ]
    )

    time_atol = float(

        quality[
            "time_atol_seconds"
        ]
    )

    if not np.allclose(

        dt_all,
        expected_dt,

        rtol=1.0e-9,
        atol=time_atol,

    ):

        raise ValueError(

            f"{direction}: time grid is not "
            f"the approved uniform "
            f"{expected_dt:.12g} s grid.\n"

            f"min dt = {dt_all.min():.12g}\n"
            f"max dt = {dt_all.max():.12g}"
        )

    _assert_close(

        f"{direction} start time",

        t_full[
            0
        ],

        quality[
            "expected_t_start_seconds"
        ],

        rtol=0.0,
        atol=time_atol,
    )

    _assert_close(

        f"{direction} end time",

        t_full[
            -1
        ],

        quality[
            "expected_t_end_seconds"
        ],

        rtol=0.0,
        atol=time_atol,
    )

    # ---------------------------------------------------------------
    # REQUIRED SUPPORTING RECORDS
    # ---------------------------------------------------------------

    preprocessing_qc = _require_json(

        cfg[
            "provenance"
        ][
            "preprocessing_qc"
        ],

        "Phase2_Preprocessing_QC",
    )

    force_qc = _require_json(

        cfg[
            "provenance"
        ][
            "force_projection_qc"
        ],

        "Phase2_Force_Projection_QC",
    )

    meta = _require_json(

        data_cfg[
            "meta"
        ],

        f"{direction} multisensor metadata",
    )

    # ---------------------------------------------------------------
    # CROSS-CHECK CSV AGAINST PREPROCESSING + FORCE QC
    # ---------------------------------------------------------------

    _validate_preprocessing_qc(

        cfg,
        preprocessing_qc,
    )

    _validate_force_qc(

        cfg,
        force_qc,

        raw_f_full,
        f_full,
    )

    # ---------------------------------------------------------------
    # OPTIONAL TRAINING TIME WINDOW
    #
    # The complete original record is always validated FIRST.
    # ---------------------------------------------------------------

    t_min = data_cfg.get(
        "t_min"
    )

    t_max = data_cfg.get(
        "t_max"
    )

    t_min = (

        float(
            t_full[
                0
            ]
        )

        if t_min is None

        else float(
            t_min
        )
    )

    t_max = (

        float(
            t_full[
                -1
            ]
        )

        if t_max is None

        else float(
            t_max
        )
    )

    if (

        t_min
        <
        t_full[
            0
        ]
        -
        time_atol

        or

        t_max
        >
        t_full[
            -1
        ]
        +
        time_atol

    ):

        raise ValueError(

            f"Requested time window "
            f"[{t_min}, {t_max}] lies outside "
            f"the approved record "
            f"[{t_full[0]}, {t_full[-1]}]."
        )

    if t_max <= t_min:

        raise ValueError(

            "data.t_max must be greater "
            "than data.t_min."
        )

    keep = (

        (
            t_full
            >=
            t_min
            -
            time_atol
        )

        &

        (
            t_full
            <=
            t_max
            +
            time_atol
        )
    )

    if int(
        np.sum(
            keep
        )
    ) < 50:

        raise ValueError(

            "Only "
            f"{int(np.sum(keep))} "
            "samples remain after the selected "
            f"time window [{t_min}, {t_max}]."
        )

    # ---------------------------------------------------------------
    # WINDOWED ARRAYS
    # ---------------------------------------------------------------

    t = t_full[
        keep
    ]

    u = u_full[
        keep
    ]

    v = v_full[
        keep
    ]

    a = a_full[
        keep
    ]

    raw_f = raw_f_full[
        keep
    ]

    f = f_full[
        keep
    ]

    # ---------------------------------------------------------------
    # NON-DIMENSIONALISATION
    # ---------------------------------------------------------------

    displacement_scale = float(

        cfg[
            "scaling"
        ][
            "displacement"
        ]
    )

    force_scale = float(

        cfg[
            "scaling"
        ][
            "force"
        ]
    )

    if (

        displacement_scale <= 0

        or

        force_scale <= 0

    ):

        raise ValueError(

            "Scaling values must be positive."
        )

    u_hat = (

        u
        /
        displacement_scale
    )

    f_hat = (

        f
        /
        force_scale
    )

    # ---------------------------------------------------------------
    # EXPLICIT MODAL REFERENCE
    # ---------------------------------------------------------------

    modal = _build_modal_reference(

        cfg[
            "reference"
        ]
    )

    # ---------------------------------------------------------------
    # FINAL DATASET
    # ---------------------------------------------------------------

    return Dataset(

        direction=
            direction,

        t=
            t.reshape(
                -1,
                1,
            ),

        u=
            u.reshape(
                -1,
                1,
            ),

        u_hat=
            u_hat.reshape(
                -1,
                1,
            ),

        v=
            v.reshape(
                -1,
                1,
            ),

        a=
            a.reshape(
                -1,
                1,
            ),

        f=
            f.reshape(
                -1,
                1,
            ),

        f_hat=
            f_hat.reshape(
                -1,
                1,
            ),

        raw_f=
            raw_f.reshape(
                -1,
                1,
            ),

        dt=
            dt,

        meta=
            meta,

        modal=
            modal,

        reference=
            dict(
                cfg[
                    "reference"
                ]
            ),

        preprocessing_qc=
            preprocessing_qc,

        force_qc=
            force_qc,

        columns=
            columns,
    )


# =====================================================================
# AUDIT SUMMARY
# =====================================================================

def print_summary(
    data: Dataset,
    cfg: dict,
) -> None:
    """Print a compact audit summary of the loaded Phase-2 record."""

    reference = cfg[
        "reference"
    ]

    print(
        "=" * 88
    )

    print(
        f"PHASE-2 DATA CONTRACT CHECK — "
        f"{data.direction}"
    )

    print(
        "=" * 88
    )

    print(
        f"CSV                  : "
        f"{cfg['data']['csv']}"
    )

    print(
        f"samples              : "
        f"{data.n}"
    )

    print(

        f"time                 : "
        f"{data.t.min():.3f} to "
        f"{data.t.max():.3f} s"
    )

    print(
        f"dt                   : "
        f"{data.dt:.12g} s"
    )

    print(
        f"observation column   : "
        f"{data.columns['observation']}"
    )

    print(
        f"equivalent-force col : "
        f"{data.columns['equivalent_force']}"
    )

    print(
        f"raw-force col        : "
        f"{data.columns['raw_force']}"
    )

    print(

        f"displacement range   : "
        f"{data.u.min():.6e} to "
        f"{data.u.max():.6e} m"
    )

    print(

        f"scaled |u_hat| peak  : "
        f"{np.max(np.abs(data.u_hat)):.6f}"
    )

    print(

        f"raw force peak       : "
        f"{np.max(np.abs(data.raw_f)):.6f} N"
    )

    print(

        f"equiv. force peak    : "
        f"{np.max(np.abs(data.f)):.6f} N"
    )

    print(

        f"scaled |F_hat| peak  : "
        f"{np.max(np.abs(data.f_hat)):.6f}"
    )

    print(

        f"reference            : "
        f"{reference['sensor']} / "
        f"Node {reference['node_id']} / "
        f"Mode {reference['mode']}"
    )

    print(

        f"equivalent mass      : "
        f"{reference['equivalent_mass_kg']:.9f} kg"
    )

    print(

        f"reference frequency  : "
        f"{reference['frequency_hz']:.12f} Hz"
    )

    print(

        f"reference stiffness  : "
        f"{reference['equivalent_stiffness_N_per_m']:.9e} N/m"
    )

    print(
        "preprocessing QC     : PASS"
    )

    print(
        "force projection QC  : PASS"
    )

    print(
        "training RHS role    : EQUIVALENT FORCE ONLY"
    )

    print(
        "=" * 88
    )


# =====================================================================
# DIRECT SMOKE TEST
# =====================================================================

if __name__ == "__main__":

    from .config import load_config

    cfg = load_config()

    dataset = load(
        cfg
    )

    print_summary(
        dataset,
        cfg,
    )