"""
Pipeline configuration definitions for all 38 FIDES configurations.

Each configuration encodes a specific combination of pipeline components,
corresponding to the TICP design principles.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class PipelineConfig:
    """Configuration for a FIDES pipeline."""

    name: str
    phase: str
    arp: bool = False
    psr: bool = False
    lbu: bool = False
    tta: Optional[str] = None
    postprocessing: Optional[str] = None
    normalization: Optional[str] = None
    intensity_tta: bool = False
    continuous_rotation: Optional[float] = None
    scale_augmentation: bool = False
    logits_smoothing: bool = False
    temperature_scaling: bool = False
    multi_scale_ensemble: bool = False

    def print_summary(self):
        print(f"  Configuration: {self.name}")
        print(f"  Phase: {self.phase}")
        print(f"  ARP (aspect ratio preservation): {self.arp}")
        print(f"  PSR (pixel spacing restoration): {self.psr}")
        print(f"  LBU (logits-bilinear-upsampling): {self.lbu}")
        print(f"  TTA: {self.tta}")
        if self.postprocessing:
            print(f"  Post-processing: {self.postprocessing}")
        if self.normalization:
            print(f"  Normalization: {self.normalization}")
        print()


# ============================================================
# Phase I: Geometric Correction (7 configs)
# ============================================================
PHASE_I = [
    PipelineConfig(name="corseg_original", phase="I"),
    PipelineConfig(name="arp_only", phase="I", arp=True),
    PipelineConfig(name="arp_psr", phase="I", arp=True, psr=True),
    PipelineConfig(name="arp_psr_lbu", phase="I", arp=True, psr=True, lbu=True),
    PipelineConfig(name="arp_psr_lbu_htta", phase="I", arp=True, psr=True, lbu=True, tta="h"),
    PipelineConfig(name="arp_psr_lbu_htta_morph", phase="I", arp=True, psr=True, lbu=True, tta="h", postprocessing="morph"),
    PipelineConfig(name="fides_optimal", phase="I", arp=True, psr=True, lbu=True, tta="hvr"),
]

# ============================================================
# Phase II: Post-Processing Disqualification (7 configs)
# ============================================================
PHASE_II = [
    PipelineConfig(name="ccf", phase="II", arp=True, psr=True, lbu=True, tta="hvr", postprocessing="ccf"),
    PipelineConfig(name="isf", phase="II", arp=True, psr=True, lbu=True, tta="hvr", postprocessing="isf"),
    PipelineConfig(name="amf", phase="II", arp=True, psr=True, lbu=True, tta="hvr", postprocessing="amf"),
    PipelineConfig(name="cbr", phase="II", arp=True, psr=True, lbu=True, tta="hvr", postprocessing="cbr"),
    PipelineConfig(name="roi_pruning", phase="II", arp=True, psr=True, lbu=True, tta="hvr", postprocessing="roi"),
    PipelineConfig(name="ccf_isf_combined", phase="II", arp=True, psr=True, lbu=True, tta="hvr", postprocessing="ccf_isf"),
    PipelineConfig(name="full_postproc", phase="II", arp=True, psr=True, lbu=True, tta="hvr", postprocessing="full"),
]

# ============================================================
# Phase III: Augmentation Boundary (8 configs)
# ============================================================
PHASE_III = [
    PipelineConfig(name="etta_h", phase="III", arp=True, psr=True, lbu=True, tta="h"),
    PipelineConfig(name="etta_hv", phase="III", arp=True, psr=True, lbu=True, tta="hv"),
    PipelineConfig(name="etta_hvr", phase="III", arp=True, psr=True, lbu=True, tta="hvr"),
    PipelineConfig(name="mse_2scale", phase="III", arp=True, psr=True, lbu=True, tta="hvr", multi_scale_ensemble=True),
    PipelineConfig(name="mse_3scale", phase="III", arp=True, psr=True, lbu=True, tta="hvr", multi_scale_ensemble=True),
    PipelineConfig(name="selective_ccf", phase="III", arp=True, psr=True, lbu=True, tta="hvr", postprocessing="selective_ccf"),
    PipelineConfig(name="logits_refine", phase="III", arp=True, psr=True, lbu=True, tta="hvr", logits_smoothing=True),
    PipelineConfig(name="etta_mse_combined", phase="III", arp=True, psr=True, lbu=True, tta="hvr", multi_scale_ensemble=True, postprocessing="selective_ccf"),
]

# ============================================================
# Phase IV: Input Perturbation (8 configs)
# ============================================================
PHASE_IV = [
    PipelineConfig(name="pct_norm", phase="IV", arp=True, psr=True, lbu=True, tta="hvr", normalization="pct"),
    PipelineConfig(name="clahe", phase="IV", arp=True, psr=True, lbu=True, tta="hvr", normalization="clahe"),
    PipelineConfig(name="itta", phase="IV", arp=True, psr=True, lbu=True, tta="hvr", intensity_tta=True),
    PipelineConfig(name="pct_clahe", phase="IV", arp=True, psr=True, lbu=True, tta="hvr", normalization="pct_clahe"),
    PipelineConfig(name="pct_itta", phase="IV", arp=True, psr=True, lbu=True, tta="hvr", normalization="pct", intensity_tta=True),
    PipelineConfig(name="clahe_itta", phase="IV", arp=True, psr=True, lbu=True, tta="hvr", normalization="clahe", intensity_tta=True),
    PipelineConfig(name="pct_clahe_itta", phase="IV", arp=True, psr=True, lbu=True, tta="hvr", normalization="pct_clahe", intensity_tta=True),
    PipelineConfig(name="no_norm", phase="IV", arp=True, psr=True, lbu=True, tta="hvr", normalization="none"),
]

# ============================================================
# Phase V: Inference Strategy Boundary (8 configs)
# ============================================================
PHASE_V = [
    PipelineConfig(name="rot_5", phase="V", arp=True, psr=True, lbu=True, tta="hvr", continuous_rotation=5.0),
    PipelineConfig(name="rot_10", phase="V", arp=True, psr=True, lbu=True, tta="hvr", continuous_rotation=10.0),
    PipelineConfig(name="rot_5_10", phase="V", arp=True, psr=True, lbu=True, tta="hvr", continuous_rotation=7.5),
    PipelineConfig(name="scale_aug", phase="V", arp=True, psr=True, lbu=True, tta="hvr", scale_augmentation=True),
    PipelineConfig(name="logits_smooth", phase="V", arp=True, psr=True, lbu=True, tta="hvr", logits_smoothing=True),
    PipelineConfig(name="temp_scale", phase="V", arp=True, psr=True, lbu=True, tta="hvr", temperature_scaling=True),
    PipelineConfig(name="rot_scale_combined", phase="V", arp=True, psr=True, lbu=True, tta="hvr", continuous_rotation=5.0, scale_augmentation=True),
    PipelineConfig(name="all_strategies", phase="V", arp=True, psr=True, lbu=True, tta="hvr", continuous_rotation=5.0, scale_augmentation=True, logits_smoothing=True, temperature_scaling=True),
]

# Clinical configuration (no eTTA-HVR, recommended for EF measurement)
FIDES_CLINICAL = PipelineConfig(
    name="fides_clinical",
    phase="clinical",
    arp=True, psr=True, lbu=True, tta=None,
)

ALL_CONFIGS = PHASE_I + PHASE_II + PHASE_III + PHASE_IV + PHASE_V + [FIDES_CLINICAL]
CONFIG_NAMES = [c.name for c in ALL_CONFIGS]
_CONFIG_MAP = {c.name: c for c in ALL_CONFIGS}


def get_config(name: str) -> PipelineConfig:
    """Get a pipeline configuration by name."""
    if name not in _CONFIG_MAP:
        raise ValueError(f"Unknown configuration: {name}. Available: {CONFIG_NAMES}")
    return _CONFIG_MAP[name]


def get_configs_by_phase(phase: str) -> List[PipelineConfig]:
    """Get all configurations for a given phase."""
    return [c for c in ALL_CONFIGS if c.phase == phase]
