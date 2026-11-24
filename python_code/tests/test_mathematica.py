"""Cross-check against the original Mathematica package MyQubits 0.0.4.

The same random inputs are fed to the Mathematica package (through
``wolframscript``) and to the Python port, and the results are compared.
The test is skipped if ``wolframscript`` is not available.  The package file
defaults to ``../MyQubits-0.0.4/Kernel/MyQubits.wl`` relative to this
directory's parent and can be set with the ``MYQUBITS_WL`` environment variable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from myqubits import (bases, channels, core, discord, entanglement, entropy, lindblad, measures,
                      qubits, random_states, subsystems)

WL = Path(os.environ.get("MYQUBITS_WL",
                         Path(__file__).resolve().parents[2] / "MyQubits-0.0.4" / "Kernel" / "MyQubits.wl"))

pytestmark = pytest.mark.skipif(shutil.which("wolframscript") is None or not WL.exists(),
                                reason="wolframscript or the Mathematica package is not available")

WOLFRAM = r"""
Get["@WL@"];
in = Import["@IN@", "RawJSON"];
c[name_] := in[name]["re"] + I in[name]["im"];
{rho2, rho4, sigma4, rho6, rho8, psi4, h2, l2, l2b, kraus, g3, m6, p6, p12} =
  c /@ {"rho2", "rho4", "sigma4", "rho6", "rho8", "psi4", "h2", "l2", "l2b", "kraus", "g3", "m6", "p6", "p12"};
p6 = Re[p6]; p12 = Re[p12];
{optAB, optBA} = Re[c /@ {"optAB", "optBA"}];
out = <|
  "DiscAtOptAB" -> vonNeumann[PartialTrace[rho4, {2, 2}, {2}]] - vonNeumann[rho4] + ClAB[rho4] @@ optAB,
  "DiscAtOptBA" -> vonNeumann[PartialTrace[rho4, {2, 2}, {1}]] - vonNeumann[rho4] + ClBA[rho4] @@ optBA,
  "vonNeumann" -> vonNeumann[rho4],
  "vonNeumannE" -> vonNeumann[rho4, E],
  "Renyi2" -> Renyi[rho4, 2],
  "RenyiHalf" -> Renyi[rho4, 0.5],
  "KullbackLeibler" -> KullbackLeibler[rho4, sigma4],
  "Fidelity" -> Fidelity[rho4, sigma4],
  "QuantumFidelity" -> QuantumFidelity[rho4, sigma4],
  "TraceNorm" -> TraceNorm[g3],
  "TraceDistance" -> TraceDistance[rho4, sigma4],
  "MatrixSqrt" -> MatrixSqrt[rho4],
  "MatrixAbs" -> MatrixAbs[g3],
  "Purity" -> Purity[rho4],
  "Concurrence" -> Concurrence[rho4],
  "ConcurrenceKet" -> Concurrence[psi4],
  "EntanglementOfFormation" -> EntanglementOfFormation[rho4],
  "Negativity" -> Negativity[rho4],
  "Negativity23" -> Negativity[rho6, {2, 3}],
  "MutualInformation" -> MutualInformation[rho4],
  "MutualInformation23" -> MutualInformation[rho6, {2, 3}],
  "TripartiteInformation" -> TripartiteInformation[rho8],
  "ClAB" -> ClAB[rho4][0.7, 1.9],
  "ClBA" -> ClBA[rho4][0.7, 1.9],
  "DiscZAB" -> DiscZAB[rho4],
  "DiscZBA" -> DiscZBA[rho4],
  "GeomDiscordAB" -> GeomDiscordAB[rho4],
  "GeomDiscordBA" -> GeomDiscordBA[rho4],
  "PartialTrace8_2" -> PartialTrace[rho8, {2, 2, 2}, {2}],
  "PartialTrace8_13" -> PartialTrace[rho8, {2, 2, 2}, {1, 3}],
  "PartialTrace6_1" -> PartialTrace[rho6, {2, 3}, 1],
  "PartialTrace6_2" -> PartialTrace[rho6, {2, 3}, 2],
  "PartialTrace4_1" -> PartialTrace[rho4, 1],
  "PartialTranspose6_2" -> PartialTranspose[rho6, {2, 3}, 2],
  "PartialTranspose8_13" -> PartialTranspose[rho8, {2, 2, 2}, {1, 3}],
  "Reshuffle" -> Reshuffle[rho4],
  "Reshuffle23" -> Reshuffle[m6, {{2, 3}, {3, 2}}],
  "Vec" -> Vec[g3], "Res" -> Res[g3],
  "Depo" -> Depo[rho2, 0.3],
  "AD" -> AD[rho2, 0.4],
  "Pauli" -> Pauli[rho2, {0.1, 0.2, 0.3}],
  "KrausToPhi" -> KrausTo\[Phi][kraus, rho2],
  "KrausToF" -> KrausToF[kraus],
  "KrausToS" -> KrausToS[kraus],
  "FToS" -> FToS[KrausToF[kraus]],
  "SToF" -> SToF[KrausToS[kraus]],
  "FToPhi" -> FTo\[Phi][KrausToF[kraus], rho2],
  "ME" -> ME[h2, {l2, l2b}][rho2],
  "Dc" -> \[ScriptCapitalD]c[l2][rho2],
  "DcascAB" -> \[ScriptCapitalD]cascAB[l2, l2b][rho2],
  "HToVec" -> HToVec[h2],
  "DToVec" -> \[ScriptCapitalD]ToVec[l2],
  "DcascABToVec" -> \[ScriptCapitalD]cascABToVec[l2, l2b],
  "METoVec" -> METoVec[h2, {l2, l2b}],
  "MatrixExpSolution" -> Unvec[MatrixExp[METoVec[h2, {l2, l2b}] 1.5] . Vec[rho2]],
  "MESolvet" -> MESolvet[ME[h2, {l2, l2b}], rho2, 0, 1.5][1.5],
  "GEN3" -> GEN[3],
  "Vec2" -> Vec2[rho4],
  "StateToBloch" -> StateToBloch[rho4],
  "BlochToState" -> BlochToState[{0.1, 0.2, 0.3}],
  "bv" -> bv[rho2],
  "Trid" -> Trid[4, 1., 2., 3.],
  "TridclosedVec" -> Tridclosed[4, {1., 2., 3., 4.}, {5., 6., 7., 8.}, {9., 10., 11., 12.}],
  "TridclosedBlock" -> Tridclosed[3, sx, sz, sy],
  "TridBlock" -> Trid[3, sx, sz, sy],
  "TightBinding" -> TightBinding[0.7, 4],
  "NQubitsSy2" -> (NQubits[3]; Sy[2]),
  "shannonEntropy" -> shannonEntropy[p6],
  "MutualInformationClassic" -> MutualInformationClassic[p6, {2, 3}],
  "conditionalEntropyXcY" -> conditionalEntropyXcY[p6, {2, 3}],
  "conditionalEntropyYcX" -> conditionalEntropyYcX[p6, {2, 3}],
  "marginal" -> marginalProbabilityDistribution[p12, {2, 3, 2}, 2]
|>;
toJ[x_] := With[{n = N[Normal[x]]}, <|"re" -> Re[n], "im" -> Im[n]|>];
Export["@OUT@", toJ /@ out, "RawJSON"];
"""


def _inputs():
    rng = np.random.default_rng(20241105)
    h = core.dag(g := random_states.ginibre(2, rng=rng)) + g
    return {
        "rho2": random_states.random_state(2, rng=rng),
        "rho4": random_states.random_state(4, rng=rng),
        "sigma4": random_states.random_state(4, "bures", rng=rng),
        "rho6": random_states.random_state(6, rng=rng),
        "rho8": random_states.random_state(8, rng=rng),
        "psi4": random_states.random_ket(4, rng=rng),
        "h2": h,
        "l2": random_states.ginibre(2, rng=rng) / 2,
        "l2b": random_states.ginibre(2, rng=rng) / 2,
        "kraus": random_states.random_kraus(2, 3, rng=rng),
        "g3": random_states.ginibre(3, rng=rng),
        "m6": random_states.ginibre(6, rng=rng),
        "p6": random_states.random_simplex(6, rng=rng),
        "p12": random_states.random_simplex(12, rng=rng),
    }


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mma")
    inp = _inputs()
    # optimal measurement directions found by Python, re-evaluated by Mathematica
    for key, measured in (("optAB", 0), ("optBA", 1)):
        inp[key] = np.array(discord.quantum_discord(inp["rho4"], measured, full_output=True)[1:])
    (tmp / "in.json").write_text(json.dumps(
        {k: {"re": np.real(v).tolist(), "im": np.imag(v).tolist()} for k, v in inp.items()}))
    script = (WOLFRAM.replace("@WL@", str(WL)).replace("@IN@", str(tmp / "in.json"))
              .replace("@OUT@", str(tmp / "out.json")))
    (tmp / "check.wl").write_text(script)
    proc = subprocess.run(["wolframscript", "-file", str(tmp / "check.wl")],
                          capture_output=True, text=True, timeout=900)
    if not (tmp / "out.json").exists():
        pytest.fail(f"wolframscript failed:\n{proc.stdout}\n{proc.stderr}")
    raw = json.loads((tmp / "out.json").read_text())
    return inp, {k: np.asarray(v["re"]) + 1j * np.asarray(v["im"]) for k, v in raw.items()}


def _dense(x):
    return x.toarray() if hasattr(x, "toarray") else x


PY = {
    "vonNeumann": lambda i: entropy.von_neumann(i["rho4"]),
    "vonNeumannE": lambda i: entropy.von_neumann(i["rho4"], np.e),
    "Renyi2": lambda i: entropy.renyi(i["rho4"], 2),
    "RenyiHalf": lambda i: entropy.renyi(i["rho4"], 0.5),
    "KullbackLeibler": lambda i: entropy.relative_entropy(i["rho4"], i["sigma4"]),
    "Fidelity": lambda i: measures.root_fidelity(i["rho4"], i["sigma4"]),
    "QuantumFidelity": lambda i: measures.fidelity(i["rho4"], i["sigma4"]),
    "TraceNorm": lambda i: measures.trace_norm(i["g3"]),
    "TraceDistance": lambda i: measures.trace_distance(i["rho4"], i["sigma4"]),
    "MatrixSqrt": lambda i: core.matrix_sqrt(i["rho4"]),
    "MatrixAbs": lambda i: core.matrix_abs(i["g3"]),
    "Purity": lambda i: measures.purity(i["rho4"]),
    "Concurrence": lambda i: entanglement.concurrence(i["rho4"]),
    "ConcurrenceKet": lambda i: entanglement.concurrence(i["psi4"]),
    "EntanglementOfFormation": lambda i: entanglement.entanglement_of_formation(i["rho4"]),
    "Negativity": lambda i: entanglement.negativity(i["rho4"]),
    "Negativity23": lambda i: entanglement.negativity(i["rho6"], (2, 3)),
    "MutualInformation": lambda i: entropy.mutual_information(i["rho4"]),
    "MutualInformation23": lambda i: entropy.mutual_information(i["rho6"], (2, 3)),
    "TripartiteInformation": lambda i: entropy.tripartite_information(i["rho8"]),
    "ClAB": lambda i: discord.conditional_entropy_after_measurement(i["rho4"], 0.7, 1.9, 0),
    "ClBA": lambda i: discord.conditional_entropy_after_measurement(i["rho4"], 0.7, 1.9, 1),
    "GeomDiscordAB": lambda i: discord.geometric_discord(i["rho4"], measured=1),
    "GeomDiscordBA": lambda i: discord.geometric_discord(i["rho4"], measured=0),
    "PartialTrace8_2": lambda i: subsystems.partial_trace(i["rho8"], (2, 2, 2), 1),
    "PartialTrace8_13": lambda i: subsystems.partial_trace(i["rho8"], (2, 2, 2), [0, 2]),
    "PartialTrace6_1": lambda i: subsystems.partial_trace(i["rho6"], (2, 3), 0),
    "PartialTrace6_2": lambda i: subsystems.partial_trace(i["rho6"], (2, 3), 1),
    "PartialTrace4_1": lambda i: subsystems.partial_trace(i["rho4"], None, 0),
    "PartialTranspose6_2": lambda i: subsystems.partial_transpose(i["rho6"], (2, 3), 1),
    "PartialTranspose8_13": lambda i: subsystems.partial_transpose(i["rho8"], (2, 2, 2), [0, 2]),
    "Reshuffle": lambda i: subsystems.reshuffle(i["rho4"]),
    "Reshuffle23": lambda i: subsystems.reshuffle(i["m6"], ((2, 3), (3, 2))),
    "Vec": lambda i: core.vec(i["g3"]),
    "Res": lambda i: core.res(i["g3"]),
    "Depo": lambda i: channels.depolarizing(i["rho2"], 0.3),
    "AD": lambda i: channels.amplitude_damping(i["rho2"], 0.4),
    "Pauli": lambda i: channels.pauli_channel(i["rho2"], (0.1, 0.2, 0.3)),
    "KrausToPhi": lambda i: channels.apply_kraus(i["kraus"], i["rho2"]),
    "KrausToF": lambda i: channels.kraus_to_ptm(i["kraus"]),
    "KrausToS": lambda i: channels.kraus_to_choi(i["kraus"]),
    "FToS": lambda i: channels.ptm_to_choi(channels.kraus_to_ptm(i["kraus"])),
    "SToF": lambda i: channels.choi_to_ptm(channels.kraus_to_choi(i["kraus"])),
    "FToPhi": lambda i: channels.apply_ptm(channels.kraus_to_ptm(i["kraus"]), i["rho2"]),
    "ME": lambda i: lindblad.lindbladian(i["h2"], [i["l2"], i["l2b"]])(i["rho2"]),
    "Dc": lambda i: lindblad.adjoint_dissipator(i["l2"], i["rho2"]),
    "DcascAB": lambda i: lindblad.cascade_dissipator(i["l2"], i["l2b"], i["rho2"]),
    "HToVec": lambda i: lindblad.hamiltonian_superop(i["h2"]),
    "DToVec": lambda i: lindblad.dissipator_superop(i["l2"]),
    "DcascABToVec": lambda i: lindblad.cascade_superop(i["l2"], i["l2b"]),
    "METoVec": lambda i: lindblad.liouvillian(i["h2"], [i["l2"], i["l2b"]]),
    "MatrixExpSolution": lambda i: lindblad.mesolve(i["h2"], [i["l2"], i["l2b"]], i["rho2"],
                                                    [0, 1.5]).states[-1],
    "MESolvet": lambda i: lindblad.mesolve(i["h2"], [i["l2"], i["l2b"]], i["rho2"], [0, 1.5],
                                           method="ode").states[-1],
    "GEN3": lambda i: bases.hermitian_basis(3),
    "Vec2": lambda i: bases.basis_coefficients(i["rho4"]),
    "StateToBloch": lambda i: bases.state_to_bloch(i["rho4"]),
    "BlochToState": lambda i: bases.bloch_to_state([0.1, 0.2, 0.3]),
    "bv": lambda i: qubits.bloch_vector(i["rho2"]),
    "Trid": lambda i: core.tridiagonal(4, 1.0, 2.0, 3.0),
    "TridclosedVec": lambda i: core.tridiagonal(4, [1, 2, 3, 4.], [5, 6, 7, 8.], [9, 10, 11, 12.],
                                                periodic=True),
    "TridclosedBlock": lambda i: core.tridiagonal(3, core.sx, core.sz, core.sy, periodic=True),
    "TridBlock": lambda i: core.tridiagonal(3, core.sx, core.sz, core.sy),
    "TightBinding": lambda i: qubits.tight_binding(0.7, 4),
    "NQubitsSy2": lambda i: qubits.QubitRegister(3).sy[1],
    "shannonEntropy": lambda i: entropy.shannon_entropy(i["p6"]),
    "MutualInformationClassic": lambda i: entropy.classical_mutual_information(i["p6"], (2, 3)),
    "conditionalEntropyXcY": lambda i: entropy.conditional_entropy(i["p6"], (2, 3), given=1),
    "conditionalEntropyYcX": lambda i: entropy.conditional_entropy(i["p6"], (2, 3), given=0),
    "marginal": lambda i: entropy.marginal(i["p12"], (2, 3, 2), 1),
}

TOL = {"MESolvet": 1e-6}


@pytest.mark.parametrize("name", sorted(PY))
def test_matches_mathematica(results, name):
    inp, mma = results
    py = np.asarray(_dense(PY[name](inp)))
    np.testing.assert_allclose(py, mma[name], rtol=0, atol=TOL.get(name, 1e-9))


@pytest.mark.parametrize("name, measured", [("AB", 0), ("BA", 1)])
def test_discord_matches_mathematica(results, name, measured):
    """Mathematica's ``DiscZ..`` searches a coarse grid and then a finer grid
    around the best coarse point, which can miss the minimum (e.g. close to
    the poles).  Python's optimum, plugged into Mathematica's own ``Cl..``
    function, must reproduce the Python discord and must never exceed
    Mathematica's ``DiscZ..`` value."""
    inp, mma = results
    py = discord.quantum_discord(inp["rho4"], measured)
    np.testing.assert_allclose(py, mma[f"DiscAtOpt{name}"].real, rtol=0, atol=1e-9)
    assert py <= mma[f"DiscZ{name}"].real + 1e-12
