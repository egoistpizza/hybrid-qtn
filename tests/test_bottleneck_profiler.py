import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest

from scripts.bottleneck_profiler import (
    axial_mps_bottleneck_profile,
    measured_axial_mps_bottleneck,
    measured_standard_bottleneck,
    reduction_percentage,
    standard_bottleneck_profile,
)


def test_standard_bottleneck_formula_matches_the_unet():
    assert standard_bottleneck_profile(512, 1024).total == measured_standard_bottleneck(512) == 14_159_872


@pytest.mark.parametrize("bond_dim", [8, 16, 32, 64])
def test_axial_bottleneck_formula_matches_the_hybrid_unet(bond_dim):
    profile = axial_mps_bottleneck_profile(512, 1024, bond_dim, spatial_size=32)
    assert profile.total == measured_axial_mps_bottleneck(bond_dim, image_size=512)


def test_reduction_at_bond_dimension_thirty_two():
    standard = standard_bottleneck_profile(512, 1024).total
    axial = axial_mps_bottleneck_profile(512, 1024, 32, spatial_size=32).total
    assert axial == 1_292_273
    assert reduction_percentage(standard, axial) == pytest.approx(90.874, abs=1e-3)
