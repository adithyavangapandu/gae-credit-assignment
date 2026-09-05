"""Pure NumPy advantage estimators and credit diagnostics."""

from gae_credit.estimators.gae import (
    GAEResult,
    compute_event_credit,
    compute_gae,
    compute_omitted_tail,
    compute_segment_credit,
    compute_sign_disagreement,
    compute_tail_fraction,
)

__all__ = [
    "GAEResult",
    "compute_event_credit",
    "compute_gae",
    "compute_omitted_tail",
    "compute_segment_credit",
    "compute_sign_disagreement",
    "compute_tail_fraction",
]
