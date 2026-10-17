"""Phase-2 PINN training for equivalent SDOF system identification.

The neural network maps physical time t to scaled reference-coordinate
displacement u_hat(t).

Trainable physical parameters:
    k_hat
    c_hat

Known structural input:
    m_hat

Physics equation:
    m_hat * u_hat_tt
    + c_hat * u_hat_t
    + k_hat * u_hat
    = F_hat_eq(t)

The equivalent-force history comes from the frozen Phase-2 input CSV and is
interpolated inside physics.py at DeepXDE collocation points.

Only displacement is used as a supervised observation. Velocity and
acceleration exported from ANSYS remain diagnostic quantities. The PINN obtains
u_t and u_tt through automatic differentiation.

The proven Phase-1 optimisation architecture is intentionally preserved during
the Phase-2 migration:
    - DeepXDE
    - PyTorch backend
    - Fourier features
    - Adam optimizer
    - softplus-positive K and C
    - distributed holdout validation
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from . import physics


# =====================================================================
# TRAINING PREFLIGHT
# =====================================================================

def _preflight(
    cfg,
    data,
):
    """Fail before network construction if the Phase-2 contract is inconsistent."""

    direction = cfg[
        "active_direction"
    ]

    if data.direction != direction:

        raise ValueError(
            f"Dataset direction {data.direction!r} "
            f"does not match config direction {direction!r}."
        )

    if cfg[
        "physics"
    ].get(
        "force_source"
    ) != "equivalent_csv":

        raise ValueError(
            "Phase-2 training requires "
            "physics.force_source='equivalent_csv'."
        )

    if data.n < 50:

        raise ValueError(
            "Too few samples for PINN training."
        )

    required_arrays = {
        "time": data.t,
        "displacement": data.u,
        "scaled displacement": data.u_hat,
        "equivalent force": data.f,
        "scaled equivalent force": data.f_hat,
    }

    for name, array in required_arrays.items():

        values = np.asarray(
            array
        )

        if values.shape != (
            data.n,
            1,
        ):

            raise ValueError(
                f"{name} has shape {values.shape}; "
                f"expected ({data.n}, 1)."
            )

        if not np.all(
            np.isfinite(
                values
            )
        ):

            raise ValueError(
                f"{name} contains NaN/Inf."
            )

    if not np.all(
        np.diff(
            data.t[:, 0]
        )
        >
        0
    ):

        raise ValueError(
            "Training time vector is not strictly increasing."
        )

    force_peak = float(
        np.max(
            np.abs(
                data.f_hat
            )
        )
    )

    if not np.isclose(
        force_peak,
        1.0,
        rtol=1.0e-6,
        atol=1.0e-8,
    ):

        raise ValueError(
            "Scaled equivalent-force peak should be approximately 1.0, "
            f"got {force_peak:.9g}."
        )

    if float(
        cfg[
            "physics"
        ][
            "mass"
        ]
    ) <= 0:

        raise ValueError(
            "Equivalent mass must be positive."
        )

    if float(
        cfg[
            "scaling"
        ][
            "displacement"
        ]
    ) <= 0:

        raise ValueError(
            "Displacement scale must be positive."
        )

    if float(
        cfg[
            "scaling"
        ][
            "force"
        ]
    ) <= 0:

        raise ValueError(
            "Force scale must be positive."
        )


# =====================================================================
# MAIN TRAINING
# =====================================================================

def run(
    cfg,
    data,
    tag="run",
    iterations=None,
    seed=0,
):
    """Train the Phase-2 PINN and return identified parameters."""

    import torch
    import deepxde as dde

    _preflight(
        cfg,
        data,
    )

    direction = cfg[
        "active_direction"
    ]

    dde.config.set_random_seed(
        seed
    )

    iterations = (
        int(
            iterations
        )
        if iterations is not None
        else int(
            cfg[
                "training"
            ][
                "iterations"
            ]
        )
    )

    if iterations <= 0:

        raise ValueError(
            "Training iterations must be positive."
        )

    out_dir = Path(
        cfg[
            "output"
        ][
            "dir"
        ]
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =================================================================
    # TRAINABLE PARAMETERS
    # =================================================================

    positive = bool(
        cfg[
            "training"
        ].get(
            "enforce_positive",
            True,
        )
    )

    k_init = float(
        cfg[
            "training"
        ][
            "k_hat_init"
        ]
    )

    c_init = float(
        cfg[
            "training"
        ][
            "c_hat_init"
        ]
    )

    if k_init <= 0:

        raise ValueError(
            "training.k_hat_init must be positive."
        )

    if c_init <= 0:

        raise ValueError(
            "training.c_hat_init must be positive."
        )

    if positive:

        k_raw = dde.Variable(
            physics.inverse_softplus(
                k_init
            )
        )

        c_raw = dde.Variable(
            physics.inverse_softplus(
                c_init
            )
        )

        print(
            "Parameters constrained positive "
            "(softplus)"
        )

    else:

        k_raw = dde.Variable(
            k_init
        )

        c_raw = dde.Variable(
            c_init
        )

    # =================================================================
    # PHASE-2 PHYSICS RESIDUAL
    #
    # IMPORTANT:
    # data is passed explicitly because the residual now uses the approved
    # equivalent-force history rather than generating an ideal sine wave.
    # =================================================================

    residual = physics.make_residual(
        cfg,
        data,
        k_raw,
        c_raw,
        positive=positive,
    )

    # =================================================================
    # TIME DOMAIN
    # =================================================================

    t_lo = float(
        data.t.min()
    )

    t_hi = float(
        data.t.max()
    )

    geom = dde.geometry.TimeDomain(
        t_lo,
        t_hi,
    )

    # =================================================================
    # HOLDOUT VALIDATION
    # =================================================================

    holdout_every = int(
        cfg[
            "training"
        ].get(
            "holdout_every",
            5,
        )
    )

    if holdout_every < 1:

        raise ValueError(
            "training.holdout_every must be >= 1."
        )

    if holdout_every > 1:

        indices = np.arange(
            data.n
        )

        test_mask = (
            indices
            %
            holdout_every
        ) == (
            holdout_every
            //
            2
        )

        train_mask = (
            ~test_mask
        )

    else:

        train_mask = np.ones(
            data.n,
            dtype=bool,
        )

        test_mask = np.zeros(
            data.n,
            dtype=bool,
        )

    t_train = data.t[
        train_mask
    ]

    u_train = data.u_hat[
        train_mask
    ]

    t_test = data.t[
        test_mask
    ]

    print(
        f"Training on {train_mask.sum()} points, "
        f"holding out {test_mask.sum()} "
        "for validation"
    )

    observation = dde.icbc.PointSetBC(
        t_train,
        u_train,
        component=0,
    )

    # =================================================================
    # PDE DATA
    # =================================================================

    num_domain = int(
        cfg[
            "training"
        ][
            "num_domain"
        ]
    )

    if num_domain <= 0:

        raise ValueError(
            "training.num_domain must be positive."
        )

    problem = dde.data.PDE(
        geom,
        residual,
        [
            observation
        ],
        num_domain=num_domain,
        num_boundary=2,
        anchors=t_train,
    )

    # =================================================================
    # NETWORK
    # =================================================================

    layers = list(
        cfg[
            "network"
        ][
            "layers"
        ]
    )

    if len(
        layers
    ) < 3:

        raise ValueError(
            "Network must contain input, hidden and output layers."
        )

    if int(
        layers[
            -1
        ]
    ) != 1:

        raise ValueError(
            "Phase-2 reduced SDOF network must have one output."
        )

    fourier = (
        cfg[
            "network"
        ].get(
            "fourier_features"
        )
        or
        {}
    )

    use_fourier = bool(
        fourier.get(
            "enabled",
            False,
        )
    )

    if use_fourier:

        f_min = float(
            fourier.get(
                "f_min",
                5.0,
            )
        )

        f_max = float(
            fourier.get(
                "f_max",
                40.0,
            )
        )

        n_fourier = int(
            fourier.get(
                "n",
                64,
            )
        )

        if n_fourier <= 0:

            raise ValueError(
                "Fourier feature count must be positive."
            )

        if f_max <= f_min:

            raise ValueError(
                "Fourier f_max must be greater than f_min."
            )

        frequencies = np.linspace(
            f_min,
            f_max,
            n_fourier,
        )

        frequency_tensor = torch.tensor(
            frequencies,
            dtype=torch.float32,
        ).reshape(
            1,
            -1,
        )

        def feature_transform(
            t,
        ):

            angular = (
                2.0
                *
                np.pi
                *
                t
                *
                frequency_tensor.to(
                    device=t.device,
                    dtype=t.dtype,
                )
            )

            return torch.cat(
                [
                    torch.sin(
                        angular
                    ),
                    torch.cos(
                        angular
                    ),
                ],
                dim=1,
            )

        layers[
            0
        ] = (
            2
            *
            n_fourier
        )

        print(
            f"Fourier features: "
            f"{n_fourier} frequencies, "
            f"{f_min:.1f}-{f_max:.1f} Hz, "
            f"input dim {layers[0]}"
        )

    else:

        midpoint = (
            0.5
            *
            (
                t_lo
                +
                t_hi
            )
        )

        half_range = max(
            0.5
            *
            (
                t_hi
                -
                t_lo
            ),
            1.0e-12,
        )

        def feature_transform(
            t,
        ):

            return (
                t
                -
                midpoint
            ) / half_range

        cycles = (
            float(
                cfg[
                    "physics"
                ][
                    "force_frequency"
                ]
            )
            *
            (
                t_hi
                -
                t_lo
            )
        )

        if cycles > 30:

            print(
                f"WARNING: approximately "
                f"{cycles:.0f} forcing cycles occur "
                "inside the selected window."
            )

            print(
                "Fourier features are recommended "
                "for this record."
            )

    net = dde.nn.FNN(
        layers,
        cfg[
            "network"
        ][
            "activation"
        ],
        cfg[
            "network"
        ][
            "initializer"
        ],
    )

    net.apply_feature_transform(
        feature_transform
    )

    # =================================================================
    # MODEL + OPTIMIZER
    #
    # Adam is deliberately preserved from the proven Phase-1 codebase.
    # =================================================================

    model = dde.Model(
        problem,
        net,
    )

    loss_weights = list(
        cfg[
            "training"
        ][
            "loss_weights"
        ]
    )

    if len(
        loss_weights
    ) != 2:

        raise ValueError(
            "Phase-2 training expects exactly two loss weights: "
            "[physics, displacement_data]."
        )

    model.compile(
        "adam",
        lr=float(
            cfg[
                "training"
            ][
                "learning_rate"
            ]
        ),
        loss_weights=loss_weights,
        external_trainable_variables=[
            k_raw,
            c_raw,
        ],
    )

    # =================================================================
    # CONVERGENCE TRACKING
    # =================================================================

    variable_file = (
        out_dir
        /
        f"convergence_{tag}.dat"
    )

    tracker = dde.callbacks.VariableValue(
        [
            k_raw,
            c_raw,
        ],
        period=100,
        filename=str(
            variable_file
        ),
        precision=8,
    )

    # =================================================================
    # TRAINING SUMMARY
    # =================================================================

    reference = cfg[
        "reference"
    ]

    reference_k_hat, _ = physics.to_scaled(
        reference[
            "equivalent_stiffness_N_per_m"
        ],
        0.0,
        cfg,
    )

    print()
    print(
        "=" * 88
    )
    print(
        f"PHASE-2 PINN TRAINING — {direction}"
    )
    print(
        "=" * 88
    )

    print(
        f"tag                  : {tag}"
    )

    print(
        f"reference            : "
        f"{reference['sensor']} / "
        f"Node {reference['node_id']} / "
        f"Mode {reference['mode']}"
    )

    print(
        f"samples              : {data.n}"
    )

    print(
        f"training samples     : "
        f"{int(train_mask.sum())}"
    )

    print(
        f"holdout samples      : "
        f"{int(test_mask.sum())}"
    )

    print(
        f"collocation points   : "
        f"{num_domain}"
    )

    print(
        f"time range           : "
        f"{t_lo:.6f} to {t_hi:.6f} s"
    )

    print(
        f"equivalent mass      : "
        f"{cfg['physics']['mass']:.9f} kg"
    )

    print(
        f"force source         : "
        f"{cfg['physics']['force_source']}"
    )

    print(
        f"force column         : "
        f"{data.columns['equivalent_force']}"
    )

    print(
        f"force scale          : "
        f"{cfg['scaling']['force']:.9f} N"
    )

    print(
        f"displacement scale   : "
        f"{cfg['scaling']['displacement']:.9e} m"
    )

    print(
        f"initial k_hat        : "
        f"{k_init:.9e}"
    )

    print(
        f"ANSYS reference k_hat: "
        f"{reference_k_hat:.9e}"
    )

    print(
        f"initial c_hat        : "
        f"{c_init:.9e}"
    )

    print(
        f"optimizer            : Adam"
    )

    print(
        f"learning rate        : "
        f"{cfg['training']['learning_rate']:.6e}"
    )

    print(
        f"iterations           : "
        f"{iterations}"
    )

    print(
        "=" * 88
    )
    print()

    # =================================================================
    # TRAIN
    # =================================================================

    start_time = time.time()

    model.train(
        iterations=iterations,
        callbacks=[
            tracker
        ],
        display_every=max(
            1,
            min(
                1000,
                iterations,
            ),
        ),
    )

    runtime = (
        time.time()
        -
        start_time
    )

    print(
        f"Training finished in "
        f"{runtime / 60.0:.2f} min"
    )

    # =================================================================
    # IDENTIFIED PARAMETERS
    # =================================================================

    k_value = physics.read_parameter(
        k_raw,
        positive=positive,
    )

    c_value = physics.read_parameter(
        c_raw,
        positive=positive,
    )

    (
        stiffness,
        damping,

    ) = physics.to_physical(
        k_value,
        c_value,
        cfg,
    )

    mass = float(
        cfg[
            "physics"
        ][
            "mass"
        ]
    )

    result = {
        "phase":
            "phase2",

        "direction":
            direction,

        "tag":
            tag,

        "identified": {
            "K_N_per_m":
                stiffness,

            "C_N_s_per_m":
                damping,
        },

        "fixed_inputs": {
            "mass_kg":
                mass,

            "force_source":
                cfg[
                    "physics"
                ][
                    "force_source"
                ],

            "force_column":
                data.columns[
                    "equivalent_force"
                ],

            "force_scale_N":
                float(
                    cfg[
                        "scaling"
                    ][
                        "force"
                    ]
                ),

            "equivalent_force_peak_N":
                float(
                    np.max(
                        np.abs(
                            data.f
                        )
                    )
                ),

            "force_frequency_hz":
                float(
                    cfg[
                        "physics"
                    ][
                        "force_frequency"
                    ]
                ),

            "displacement_scale_m":
                float(
                    cfg[
                        "scaling"
                    ][
                        "displacement"
                    ]
                ),
        },

        "reference": {
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

            "mode":
                int(
                    reference[
                        "mode"
                    ]
                ),

            "phi_reference":
                float(
                    reference[
                        "phi_reference"
                    ]
                ),

            "frequency_hz":
                float(
                    reference[
                        "frequency_hz"
                    ]
                ),

            "equivalent_mass_kg":
                float(
                    reference[
                        "equivalent_mass_kg"
                    ]
                ),

            "equivalent_stiffness_N_per_m":
                float(
                    reference[
                        "equivalent_stiffness_N_per_m"
                    ]
                ),
        },

        "derived":
            physics.derived(
                stiffness,
                damping,
                mass,
            ),

        "scaled": {
            "k_hat":
                k_value,

            "c_hat":
                c_value,

            "m_hat":
                physics.scaled_mass(
                    cfg
                ),

            "reference_k_hat":
                reference_k_hat,
        },

        "run": {
            "iterations":
                iterations,

            "runtime_seconds":
                runtime,

            "seed":
                seed,

            "n_samples":
                data.n,

            "n_train":
                int(
                    train_mask.sum()
                ),

            "n_holdout":
                int(
                    test_mask.sum()
                ),

            "num_domain":
                num_domain,

            "t_range": [
                t_lo,
                t_hi,
            ],

            "enforce_positive":
                positive,

            "holdout_every":
                holdout_every,

            "optimizer":
                "adam",

            "learning_rate":
                float(
                    cfg[
                        "training"
                    ][
                        "learning_rate"
                    ]
                ),
        },
    }

    # =================================================================
    # PREDICTION OVER ALL MEASURED TIMES
    # =================================================================

    u_hat_predicted = model.predict(
        data.t
    ).reshape(
        -1
    )

    u_predicted = (
        u_hat_predicted
        *
        float(
            cfg[
                "scaling"
            ][
                "displacement"
            ]
        )
    )

    u_measured = data.u.reshape(
        -1
    )

    error = (
        u_predicted
        -
        u_measured
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    denominator = float(
        np.sqrt(
            np.mean(
                u_measured ** 2
            )
        )
    )

    variance_denominator = float(
        np.sum(
            (
                u_measured
                -
                u_measured.mean()
            )
            ** 2
        )
    )

    result[
        "fit"
    ] = {
        "rmse_m":
            rmse,

        "relative_rmse":
            (
                rmse
                /
                denominator

                if denominator > 0

                else float(
                    "nan"
                )
            ),

        "max_abs_error_m":
            float(
                np.max(
                    np.abs(
                        error
                    )
                )
            ),

        "r_squared":
            (
                1.0
                -
                float(
                    np.sum(
                        error ** 2
                    )
                )
                /
                variance_denominator

                if variance_denominator > 0

                else float(
                    "nan"
                )
            ),

        "note":
            "computed over all samples, including the holdout locations",
    }

    # =================================================================
    # HONEST HOLDOUT VALIDATION
    # =================================================================

    if test_mask.sum() > 0:

        training_error = error[
            train_mask
        ]

        holdout_error = error[
            test_mask
        ]

        holdout_measurement = u_measured[
            test_mask
        ]

        holdout_rms_denominator = float(
            np.sqrt(
                np.mean(
                    holdout_measurement ** 2
                )
            )
        )

        holdout_rmse = float(
            np.sqrt(
                np.mean(
                    holdout_error ** 2
                )
            )
        )

        holdout_variance = float(
            np.sum(
                (
                    holdout_measurement
                    -
                    holdout_measurement.mean()
                )
                ** 2
            )
        )

        result[
            "validation"
        ] = {
            "n_train":
                int(
                    train_mask.sum()
                ),

            "n_holdout":
                int(
                    test_mask.sum()
                ),

            "rmse_train_m":
                float(
                    np.sqrt(
                        np.mean(
                            training_error ** 2
                        )
                    )
                ),

            "rmse_holdout_m":
                holdout_rmse,

            "relative_rmse_holdout":
                (
                    holdout_rmse
                    /
                    holdout_rms_denominator

                    if holdout_rms_denominator > 0

                    else float(
                        "nan"
                    )
                ),

            "r_squared_holdout":
                (
                    1.0
                    -
                    float(
                        np.sum(
                            holdout_error ** 2
                        )
                    )
                    /
                    holdout_variance

                    if holdout_variance > 0

                    else float(
                        "nan"
                    )
                ),

            "note":
                "held-out points were never used as displacement observations",
        }

    # =================================================================
    # SAVE PREDICTIONS
    # =================================================================

    import pandas as pd

    pd.DataFrame(
        {
            "Time (s)":
                data.t.ravel(),

            "u_measured (m)":
                u_measured,

            "u_predicted (m)":
                u_predicted,

            "error (m)":
                error,

            "Equivalent_Force (N)":
                data.f.ravel(),
        }

    ).to_csv(
        out_dir
        /
        f"prediction_{tag}.csv",

        index=False,
    )

    # =================================================================
    # SAVE CONVERGENCE
    # =================================================================

    _tidy_convergence(
        variable_file,
        out_dir
        /
        f"convergence_{tag}.csv",
        cfg,
        positive_log=positive,
    )

    # =================================================================
    # SAVE IDENTIFIED PARAMETERS
    # =================================================================

    identified_path = (
        out_dir
        /
        f"identified_{tag}.json"
    )

    with identified_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    # =================================================================
    # CONSOLE SUMMARY
    # =================================================================

    reference_k = float(
        reference[
            "equivalent_stiffness_N_per_m"
        ]
    )

    reference_frequency = float(
        reference[
            "frequency_hz"
        ]
    )

    k_percent_difference = (
        100.0
        *
        (
            stiffness
            -
            reference_k
        )
        /
        reference_k
    )

    frequency_percent_difference = (
        100.0
        *
        (
            result[
                "derived"
            ][
                "f_n_hz"
            ]
            -
            reference_frequency
        )
        /
        reference_frequency
    )

    print()
    print(
        "=" * 88
    )

    print(
        f"PHASE-2 TRAINING RESULT — "
        f"{direction}"
    )

    print(
        "=" * 88
    )

    print(
        f"Identified K          : "
        f"{stiffness:.6e} N/m"
    )

    print(
        f"ANSYS reference K     : "
        f"{reference_k:.6e} N/m"
    )

    print(
        f"K difference          : "
        f"{k_percent_difference:+.6f} %"
    )

    print(
        f"Identified C          : "
        f"{damping:.6e} N.s/m"
    )

    print(
        f"Derived f_n           : "
        f"{result['derived']['f_n_hz']:.6f} Hz"
    )

    print(
        f"ANSYS reference f_n   : "
        f"{reference_frequency:.6f} Hz"
    )

    print(
        f"frequency difference  : "
        f"{frequency_percent_difference:+.6f} %"
    )

    print(
        f"Derived zeta          : "
        f"{result['derived']['zeta'] * 100.0:.8f} %"
    )

    print(
        f"Fit RMSE              : "
        f"{rmse:.6e} m"
    )

    print(
        f"Relative fit RMSE     : "
        f"{result['fit']['relative_rmse'] * 100.0:.6f} %"
    )

    if "validation" in result:

        validation = result[
            "validation"
        ]

        print(
            f"Holdout RMSE          : "
            f"{validation['rmse_holdout_m']:.6e} m"
        )

        print(
            f"Relative holdout RMSE : "
            f"{validation['relative_rmse_holdout'] * 100.0:.6f} %"
        )

        print(
            f"Holdout R2            : "
            f"{validation['r_squared_holdout']:.9f}"
        )

    print(
        f"Runtime               : "
        f"{runtime / 60.0:.3f} min"
    )

    print(
        f"Saved                 : "
        f"{identified_path}"
    )

    print(
        "=" * 88
    )

    return result


# =====================================================================
# CONVERGENCE LOG CONVERSION
# =====================================================================

def _tidy_convergence(
    dat_path,
    csv_path,
    cfg,
    positive_log=True,
):
    """Convert DeepXDE parameter log into physical-unit CSV."""

    if not Path(
        dat_path
    ).exists():

        return

    rows = []

    with open(
        dat_path,
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:

                continue

            try:

                step, rest = line.split(
                    " ",
                    1,
                )

                values = json.loads(
                    rest.replace(
                        "'",
                        '"',
                    )
                )

                if positive_log:

                    values = [
                        (
                            math.log1p(
                                math.exp(
                                    value
                                )
                            )
                            if value < 30
                            else value
                        )

                        for value
                        in values
                    ]

                (
                    stiffness,
                    damping,

                ) = physics.to_physical(
                    values[
                        0
                    ],
                    values[
                        1
                    ],
                    cfg,
                )

                rows.append(
                    {
                        "iteration":
                            int(
                                step
                            ),

                        "k_hat":
                            values[
                                0
                            ],

                        "c_hat":
                            values[
                                1
                            ],

                        "K_N_per_m":
                            stiffness,

                        "C_N_s_per_m":
                            damping,
                    }
                )

            except (
                ValueError,
                json.JSONDecodeError,
                IndexError,
            ):

                continue

    if rows:

        import pandas as pd

        pd.DataFrame(
            rows
        ).to_csv(
            csv_path,
            index=False,
        )