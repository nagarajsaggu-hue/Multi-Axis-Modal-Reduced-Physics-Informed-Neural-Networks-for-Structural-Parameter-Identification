"""Configuration loading for Phase 2 PINN system identification.

One config.yaml contains all X/Y/Z cases. Exactly one active_direction line must
be uncommented. This module resolves the selected case into the flat dictionary
consumed by the rest of the codebase.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import yaml


_VALID_DIRECTIONS = {"X", "Y", "Z"}


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[2]


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a deep copy of base."""
    result = copy.deepcopy(base)

    for key, value in (override or {}).items():

        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):

            result[key] = _deep_merge(
                result[key],
                value,
            )

        else:

            result[key] = copy.deepcopy(
                value
            )

    return result


def _resolve_path(
    root: Path,
    value,
):
    """Resolve a path against the repository root."""

    if value in (
        None,
        "",
    ):

        return None

    path = Path(value)

    if path.is_absolute():

        return str(path)

    return str(
        root / path
    )


def _check_single_active_direction(
    raw_text: str,
) -> None:
    """Require exactly one uncommented active_direction declaration."""

    matches = re.findall(

        r"^[ \t]*active_direction[ \t]*:",

        raw_text,

        flags=re.MULTILINE,
    )

    if len(matches) != 1:

        raise ValueError(

            "config.yaml must contain exactly ONE uncommented "
            "'active_direction:' line; "
            f"found {len(matches)}."
        )


def _validate_config(
    cfg: dict,
) -> None:
    """Validate the resolved Phase-2 configuration before data loading."""

    direction = cfg[
        "active_direction"
    ]

    if direction not in _VALID_DIRECTIONS:

        raise ValueError(

            "active_direction must be one of "
            f"{_VALID_DIRECTIONS}, "
            f"got {direction!r}"
        )

    data = cfg[
        "data"
    ]

    required_data_keys = (

        "csv",
        "meta",
        "time_column",
        "observation_column",
        "velocity_column",
        "acceleration_column",
        "raw_force_column",
        "force_column",
    )

    missing = [

        key

        for key
        in required_data_keys

        if not data.get(
            key
        )
    ]

    if missing:

        raise KeyError(

            "Missing required data configuration keys: "
            f"{missing}"
        )

    if (
        data["force_column"]
        ==
        data["raw_force_column"]
    ):

        raise ValueError(

            "Equivalent force column and raw face "
            "force column cannot match."
        )

    if not str(
        data["force_column"]
    ).startswith(
        "Equivalent_Force_"
    ):

        raise ValueError(

            "Phase 2 physics must use an "
            "'Equivalent_Force_*' column, got "
            f"{data['force_column']!r}."
        )

    if not str(
        data["raw_force_column"]
    ).startswith(
        "Raw_Face_Force_"
    ):

        raise ValueError(

            "raw_force_column must explicitly identify "
            "the ANSYS raw face force."
        )

    reference = cfg[
        "reference"
    ]

    for key in (

        "sensor",
        "node_id",
        "mode",
        "phi_reference",
        "frequency_hz",
        "equivalent_mass_kg",
        "equivalent_stiffness_N_per_m",

    ):

        if reference.get(
            key
        ) is None:

            raise KeyError(
                f"Missing reference.{key}"
            )

    if (
        reference["sensor"]
        !=
        "S_C07"
        or
        int(
            reference["node_id"]
        )
        !=
        3461
    ):

        raise ValueError(

            "Phase-2 reference coordinate must be "
            "S_C07 / Node 3461."
        )

    physics = cfg[
        "physics"
    ]

    if physics.get(
        "force_source"
    ) != "equivalent_csv":

        raise ValueError(

            "Phase-2 physics.force_source must be "
            "'equivalent_csv'."
        )

    if float(
        physics["mass"]
    ) <= 0:

        raise ValueError(
            "physics.mass must be positive."
        )

    scaling = cfg[
        "scaling"
    ]

    if (
        float(
            scaling["displacement"]
        )
        <=
        0
        or
        float(
            scaling["force"]
        )
        <=
        0
    ):

        raise ValueError(

            "Scaling values must be positive."
        )

    training = cfg[
        "training"
    ]

    if int(
        training["iterations"]
    ) <= 0:

        raise ValueError(

            "training.iterations must be positive."
        )

    if float(
        training["learning_rate"]
    ) <= 0:

        raise ValueError(

            "training.learning_rate must be positive."
        )


def load_config(
    path=None,
) -> dict:
    """Load config.yaml and resolve the selected X/Y/Z case.

    The returned structure intentionally preserves the legacy top-level
    keys:

        data
        physics
        scaling
        network
        training
        output

    This keeps the downstream migration small and explicit.
    """

    root = project_root()

    config_path = (

        Path(path)

        if path

        else root / "config.yaml"
    )

    raw_text = config_path.read_text(
        encoding="utf-8"
    )

    _check_single_active_direction(
        raw_text
    )

    raw = yaml.safe_load(
        raw_text
    )

    if not isinstance(
        raw,
        dict,
    ):

        raise ValueError(

            "config.yaml did not parse to a mapping."
        )

    direction = str(

        raw.get(
            "active_direction",
            "",
        )

    ).upper()

    if direction not in _VALID_DIRECTIONS:

        raise ValueError(

            "active_direction must be X, Y or Z; "
            f"got {direction!r}"
        )

    cases = (
        raw.get(
            "cases"
        )
        or
        {}
    )

    if direction not in cases:

        raise KeyError(

            f"No cases.{direction} block exists "
            "in config.yaml."
        )

    case = copy.deepcopy(
        cases[
            direction
        ]
    )

    # -----------------------------------------------------------------
    # DATA PATHS
    # -----------------------------------------------------------------

    data = copy.deepcopy(
        case[
            "data"
        ]
    )

    data["csv"] = _resolve_path(

        root,
        data["csv"],
    )

    data["meta"] = _resolve_path(

        root,
        data["meta"],
    )

    # -----------------------------------------------------------------
    # PROVENANCE / QC PATHS
    # -----------------------------------------------------------------

    provenance = copy.deepcopy(

        raw.get(
            "provenance"
        )
        or
        {}
    )

    for key in (

        "preprocessing_qc",
        "force_projection_qc",
        "preprocessing_summary",

    ):

        if provenance.get(
            key
        ):

            provenance[
                key
            ] = _resolve_path(

                root,
                provenance[
                    key
                ],
            )

    # -----------------------------------------------------------------
    # REFERENCE COORDINATE
    # -----------------------------------------------------------------

    reference = copy.deepcopy(
        case[
            "reference"
        ]
    )

    # -----------------------------------------------------------------
    # PHYSICS
    # -----------------------------------------------------------------

    physics = copy.deepcopy(
        case[
            "physics"
        ]
    )

    # Single source of truth:
    # structural mass / mode / frequency / stiffness come from reference.

    physics["mass"] = float(

        reference[
            "equivalent_mass_kg"
        ]
    )

    physics[
        "reference_mode"
    ] = int(

        reference[
            "mode"
        ]
    )

    physics[
        "reference_frequency_hz"
    ] = float(

        reference[
            "frequency_hz"
        ]
    )

    physics[
        "reference_stiffness_N_per_m"
    ] = float(

        reference[
            "equivalent_stiffness_N_per_m"
        ]
    )

    # -----------------------------------------------------------------
    # SCALING
    # -----------------------------------------------------------------

    scaling = copy.deepcopy(
        case[
            "scaling"
        ]
    )

    # -----------------------------------------------------------------
    # NETWORK + DIRECTION-SPECIFIC OVERRIDES
    # -----------------------------------------------------------------

    network = _deep_merge(

        raw.get(
            "network"
        )
        or
        {},

        case.get(
            "network_overrides"
        )
        or
        {},
    )

    # -----------------------------------------------------------------
    # TRAINING + DIRECTION-SPECIFIC OVERRIDES
    # -----------------------------------------------------------------

    training = _deep_merge(

        raw.get(
            "training"
        )
        or
        {},

        case.get(
            "training_overrides"
        )
        or
        {},
    )

    # -----------------------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------------------

    output_root = _resolve_path(

        root,

        (
            raw.get(
                "output"
            )
            or
            {}
        )[
            "root"
        ],
    )

    output_dir = (

        Path(
            output_root
        )
        /
        direction
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # FINAL RESOLVED CONFIG
    # -----------------------------------------------------------------

    cfg = {

        "phase":
            raw.get(
                "phase",
                "phase2",
            ),

        "active_direction":
            direction,

        "data":
            data,

        "reference":
            reference,

        "physics":
            physics,

        "scaling":
            scaling,

        "network":
            network,

        "training":
            training,

        "quality":
            copy.deepcopy(
                raw.get(
                    "quality"
                )
                or
                {}
            ),

        "provenance":
            provenance,

        "output": {

            "root":
                output_root,

            "dir":
                str(
                    output_dir
                ),
        },

        "_root":
            str(
                root
            ),

        "_config_path":
            str(
                config_path.resolve()
            ),
    }

    _validate_config(
        cfg
    )

    return cfg


def load_json(
    path,
):
    """Load a JSON file.

    Returns None only when no path was supplied or the file does not exist.
    """

    if not path:

        return None

    path = Path(
        path
    )

    if not path.exists():

        return None

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )