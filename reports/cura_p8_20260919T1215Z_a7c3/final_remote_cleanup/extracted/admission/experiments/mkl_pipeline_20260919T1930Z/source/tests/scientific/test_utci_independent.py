"""Independent checks against the pinned official UTCI a0.002 source.

The official Fortran is parsed directly. Candidate helpers never produce the
expected polynomial values or coefficient inventory.
"""
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from importlib.resources import files
import re

import numpy as np
import pytest

from solweig_light.comfort.utci import utci_calculator
from solweig_light.comfort._utci_scalar import polynomial_scalar


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "reports/characterization/p8_utci_official_source"
SOURCE = OFFICIAL / "UTCI_a002.f90"
README = OFFICIAL / "ReadMe_UTCI_a002.txt"
VARIABLES = ("Ta", "va", "D_Tmrt", "Pa")


def _verified_official_text():
    manifest = json.loads((OFFICIAL / "source_manifest.json").read_text())
    assert manifest["url"] == "https://utci.org/resources/UTCI%20Program%20Code.zip"
    assert manifest["sha256"] == "83ea34dc2428093c8b0f4e299bfb0a24752ff49ec1e11741927a75dafae283f1"
    expected = manifest["text_members"]
    for member, path in (("UTCI Program Code/UTCI_a002.f90", SOURCE),
                         ("UTCI Program Code/ReadMe_UTCI_a002.txt", README)):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[member]["sha256"]
    return SOURCE.read_bytes().decode("latin-1")


def _official_polynomial_terms():
    text = _verified_official_text()
    start = text.index("UTCI_approx=Ta+&")
    stop = text.index("\n      return", start)
    lines = text[start:stop].splitlines()
    terms = [(Decimal(1), ("Ta",), {name: int(name == "Ta") for name in VARIABLES})]
    pattern = re.compile(r"^\s*\(\s*([+-]?\d+\.\d+D[+-]\d+)\s*\)\s*(.*?)\s*(?:\+\s*&)?\s*$")
    for line in lines[1:]:
        match = pattern.match(line)
        if not match:
            continue
        coefficient = Decimal(match.group(1).replace("D", "E"))
        factors = tuple(re.findall(r"D_Tmrt|Ta|va|Pa", match.group(2)))
        counts = Counter(factors)
        terms.append((coefficient, factors, {name: counts[name] for name in VARIABLES}))
    assert len(terms) == 211
    return terms


def _official_utci(ta, vapor_pressure_hpa, tmrt, wind):
    values = {"Ta": float(ta), "va": float(wind),
              "D_Tmrt": float(tmrt) - float(ta),
              "Pa": float(vapor_pressure_hpa) / 10.0}
    result = 0.0
    for coefficient, factors, _ in _official_polynomial_terms():
        term = float(coefficient)
        # Retain the written Fortran multiplication sequence; do not use the
        # candidate expression or an algebraically regrouped monomial.
        for factor in factors:
            term *= values[factor]
        result += term
    return result


def _official_saturation_hpa(ta):
    text = _verified_official_text()
    start = text.index("REAL :: g(0:7)")
    stop = text.index("/)", start)
    coefficients = [Decimal(token.replace("D", "E")) for token in
                    re.findall(r"[+-]?\d+(?:\.\d*)?(?:E[+-]?\d+)?", text[start:stop], re.I)]
    # The declaration contributes the bounds 0 and 7; the final eight values
    # are the published Hardy saturation-vapour-pressure coefficients.
    g = [float(value) for value in coefficients[-8:]]
    tk = float(ta) + 273.15
    exponent = g[7] * np.log(tk)
    for index in range(7):
        exponent += g[index] * tk ** (index - 2)
    return float(np.exp(exponent) * 0.01)


def test_candidate_polynomial_coefficients_and_exponents_match_official_source():
    official = _official_polynomial_terms()
    candidate = json.loads(files("solweig_light.comfort").joinpath("utci_coefficients.json").read_text())
    assert candidate["term_count"] == len(candidate["ordered_terms"]) == len(official) == 211
    for item, (coefficient, _, powers) in zip(candidate["ordered_terms"], official, strict=True):
        assert item["index"] >= 0
        assert Decimal(str(item["coefficient"])) == coefficient
        assert item["powers"] == powers


# Valid official-domain points include boundaries and mixed-sign/cancellation
# cases; they are a focused diagnostic set, not a full-domain error guarantee.
PRESSURE_CASES = (
    (-50.0, 0.0, -80.0, 0.5),
    (50.0, 50.0, 120.0, 17.0),
    (0.0, 5.0, 0.0, 0.5),
    (-20.0, 1.0, 50.0, 17.0),
    (32.0, 4.0, 2.0, 7.3),
)


@pytest.mark.parametrize("ta,ehpa,tmrt,wind", PRESSURE_CASES)
def test_pressure_polynomial_matches_independent_official_evaluator(ta, ehpa, tmrt, wind):
    assert -50 <= ta <= 50
    assert -30 <= tmrt - ta <= 70
    assert 0.5 <= wind <= 17
    assert 0 <= ehpa <= min(50, _official_saturation_hpa(ta))
    expected = _official_utci(ta, ehpa, tmrt, wind)
    actual = polynomial_scalar(*(np.float32(value) for value in
                                 (tmrt - ta, ta, wind, ehpa / 10.0)))
    # Runtime polynomial arithmetic is float32 and retains written operation
    # order. Keep the frozen 0.02 C gate; this focused set does not establish a
    # tighter maximum over the continuous validity domain.
    assert abs(float(actual) - expected) <= 0.02


RH_CASES = (
    (-40.0, 15.0, -50.0, 0.5),
    (0.0, 100.0, 30.0, 1.0),
    (25.0, 50.0, 70.0, 3.0),
    (45.0, 20.0, 80.0, 17.0),
)


@pytest.mark.parametrize("ta,rh,tmrt,wind", RH_CASES)
def test_rh_interface_against_official_saturation_and_polynomial(ta, rh, tmrt, wind):
    ehpa = _official_saturation_hpa(ta) * rh / 100.0
    assert 0 <= ehpa <= 50
    expected = _official_utci(ta, ehpa, tmrt, wind)
    actual = utci_calculator(*(np.array([value], dtype=np.float32)
                               for value in (ta, rh, tmrt, wind)))[0]
    # This includes differences in RH->vapour-pressure arithmetic and the
    # candidate float32 path. It therefore retains the frozen 0.02 C UTCI gate
    # and is deliberately distinct from direct-pressure polynomial parity.
    assert abs(float(actual) - expected) <= 0.02


def test_official_wind_domain_does_not_validate_model_floor():
    readme = README.read_bytes().decode("latin-1")
    assert "wind speed between 0.5 and 17 m/s" in readme
    pipeline = (ROOT / "src/solweig_light/pipeline.py").read_text()
    assert "np.float32(.15)" in pipeline
    # The inherited workflow floor is 0.15 m/s for WBGT compatibility, below
    # the official UTCI polynomial validity limit. No official-parity claim is
    # made for workflow pixels whose effective speed is below 0.5 m/s.
