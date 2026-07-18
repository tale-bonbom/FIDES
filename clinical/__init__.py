"""
Clinical measurement and statistics utilities for FIDES.

This package provides:
- measurements: EDV, ESV, EF calculation from segmentation volumes
- statistics: ICC, Bland-Altman, Wilcoxon signed-rank tests
- error_cancellation: Theorem 1 (Error Cancellation) verification
- tables: Result table generation (LaTeX/CSV)
"""

from clinical.measurements import (
    compute_edv,
    compute_esv,
    compute_ef,
    compute_clinical_metrics,
)
from clinical.statistics import (
    icc_2_1,
    bland_altman,
    wilcoxon_test,
)
