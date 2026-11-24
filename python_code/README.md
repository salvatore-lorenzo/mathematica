# myqubits (Python)

Python port of the Mathematica package **MyQubits 0.0.4**. It covers quantum
information and open quantum systems: states and operators, partial traces,
entropies and correlation measures, entanglement and discord, quantum channels
(Kraus, superoperator, Choi, process matrix), Lindblad master equations and
Bloch-sphere plots.

The package is redesigned for NumPy/SciPy rather than translated line by line
(see [Design](#design)). It is checked against the original Mathematica
package: `tests/test_mathematica.py` compares more than 70 results.

## Installation

```bash
cd python_code
pip install -e .            # numpy + scipy
pip install -e ".[all]"     # + matplotlib (bloch), sympy (symbolic), pytest
pytest                      # Mathematica cross-check runs if wolframscript is installed
```

## Modules

Every module can be imported on its own. `core` holds the basic definitions
that the other modules share.

| module | content |
|---|---|
| `myqubits.core` | kets, Pauli/ladder operators, gates, Bell/GHZ/W states, `kron`, `dag`, `comm`, `ket2dm`, `direct_sum`, `tridiagonal`, `vec`/`unvec`/`res`/`unres`, superoperator helpers, matrix square root and absolute value, validity checks |
| `myqubits.subsystems` | `partial_trace`, `partial_transpose`, `permute_subsystems`, `reshuffle` |
| `myqubits.measures` | `fidelity`, `root_fidelity`, `trace_norm`, `trace_distance`, `purity` |
| `myqubits.entropy` | von Neumann, Rényi, relative entropy, Shannon, marginals, classical/quantum mutual information, conditional entropies, tripartite information |
| `myqubits.entanglement` | `concurrence`, `concurrence_x`, `entanglement_of_formation`, `negativity`, `log_negativity` |
| `myqubits.discord` | `quantum_discord`, `geometric_discord`, `conditional_entropy_after_measurement`, `two_qubit_correlations` |
| `myqubits.bases` | Gell-Mann matrices, orthonormal Hermitian basis, Pauli-string basis, matrix units, generalized Bloch vectors |
| `myqubits.channels` | standard channels, conversions between Kraus / superoperator / Choi / process matrix, the `Channel` class |
| `myqubits.lindblad` | dissipators, Lindbladian generator, Liouvillian superoperators, cascaded systems, `mesolve`, `propagate`, `solve_generator`, `steady_state` |
| `myqubits.qubits` | `QubitRegister`, local operators, XX (tight-binding) chain, qubit Bloch vector |
| `myqubits.random_states` | random simplex points, kets, Ginibre matrices, unitaries, density matrices (HS, Bures, induced), POVMs, Kraus maps |
| `myqubits.bloch` | `BlochSphere` plotting class (needs matplotlib) |
| `myqubits.symbolic` | SymPy tools: symbolic matrices, real-symbol conjugation, symbolic partial traces, closed-form Lindblad solutions (needs sympy) |

`import myqubits` exposes the contents of `core` at the top level and loads
the other modules lazily (`myqubits.entropy` works without an explicit import).

[FUNCTIONS.md](FUNCTIONS.md) lists every function, class and constant in each
file, with its signature and a one-line description.

## Quick start

```python
import numpy as np
from myqubits.core import ket2dm, phi_plus, sm, sz, kron
from myqubits.subsystems import partial_trace
from myqubits.entropy import von_neumann, mutual_information
from myqubits.entanglement import concurrence, negativity
from myqubits.discord import quantum_discord
from myqubits.lindblad import mesolve
from myqubits.channels import Channel, amplitude_damping_kraus
from myqubits.random_states import random_state

rho = ket2dm(phi_plus)
partial_trace(rho, [2, 2], 0)              # trace out qubit 0 -> I/2
von_neumann(partial_trace(phi_plus, [2, 2], 1))   # kets are fine: 1.0
mutual_information(rho), concurrence(rho), negativity(rho), quantum_discord(rho)

# Lindblad equation: exact (matrix exponential) or ODE
sol = mesolve(0.5 * sz, np.sqrt(0.1) * sm, random_state(2, rng=1), np.linspace(0, 20, 201))
sol.states.shape        # (201, 2, 2)
sol(3.7)                # state at any time
sol.expect(sz)          # <sz>(t)

# Channels in any representation
ch = Channel.from_kraus(amplitude_damping_kraus(0.3))
ch(random_state(2)); ch.choi(); ch.ptm(); (ch @ ch).kraus(); ch.is_cptp()
```

Symbolic computations (Mathematica's `MESolve`, `SymMatrix`, `cf`, `Conj`):

```python
import sympy as S
from myqubits import symbolic as sym
g, w = S.symbols("gamma omega", positive=True)
rho_t = sym.mesolve(w / 2 * sym.sz, [S.sqrt(g) * sym.sm], sym.sym_matrix("r", 2, real=False))
rho_t[1, 1]             # r_22*exp(-gamma*tau)
```

## Conventions

These are the same as in the Mathematica package, except for the numbering:

* **Indices start at 0.** Subsystems, qubits, Gell-Mann indices and the
  measured qubit of the discord functions are all 0-based. Mathematica's
  `PartialTrace[rho, {2,2,2}, {1,3}]` is `partial_trace(rho, (2,2,2), [0, 2])`.
  Negative indices count from the end.
* Kets are 1-D arrays and operators are 2-D arrays (dense, or `scipy.sparse`
  CSR arrays where sparsity matters). `ket0 = |0> = [1, 0]`,
  `ket1 = |1> = [0, 1]`, `sm = |0><1|`, `sp = |1><0|`.
* In `kron(A, B)`, `A` acts on subsystem 0.
* `vec` stacks columns and `res` stacks rows. Every superoperator acts on `vec`.
* Logarithms are base 2 unless `base` is given.
* Collections of operators (Kraus operators, POVMs, bases) are stacks of shape `(k, d, d)`.
* Many functions accept stacks of states `(..., d, d)`, for example a whole
  trajectory `sol.states`. This works for entropies, `purity`, `partial_trace`,
  `partial_transpose`, `negativity`, `bloch_vector`, channels and the
  Lindbladian generator.
* Random functions take `rng` (a seed or a `numpy.random.Generator`) and most take `size`.
* Module constants (`sx`, `phi_plus`, ...) are read-only arrays, so they cannot be modified by accident.

## Design

These changes go beyond a literal translation:

* **One general partial trace** built from reshapes and axis permutations. It
  replaces `PartialTrace`, `PartialTrace1/2`, `PartialTraceGeneral` and
  `ListReshape`, and handles kets directly (the reduced state comes from the
  amplitudes, without building `|psi><psi|`), sparse operators and batches.
  The `keep=` keyword selects the subsystems to keep instead.
* **Discord in closed form.** For a measurement direction `n`, the
  post-measurement states follow from the Fano form (`a`, `b`, `T`) of the
  state, so the conditional entropy is a vectorized expression. The whole
  grid is evaluated at once and then refined with Nelder-Mead. It takes about
  20 ms (Mathematica's `DiscZAB`: about 470 ms) and finds the true minimum (see below).
* **Channels through the superoperator.** Choi matrix = index permutation of
  the superoperator; process matrix = unitary change of basis. Conversions
  are a reshape or a matrix product instead of `d²` applications of the map.
  A `Channel` class composes channels (`@`, `**`) and checks CP/TP/unitality.
* **Concurrence** from the singular values of `sqrt(rho) sqrt(rho~)`, which
  avoids the non-Hermitian eigenproblem.
* **XX chain** built directly from bit flips (`O(n 2^n)`, no Kronecker
  products). It supports periodic boundaries and position-dependent couplings.
* **Solvers.** `mesolve` is exact: `expm`, or `expm_multiply` for sparse
  Liouvillians, which never forms the exponential. `method="ode"` integrates
  the matrix-form generator with the effective Hamiltonian precomputed.
  `solve_generator` takes any (time-dependent) generator. `steady_state` is new.
* **`NQubits` global definitions** are replaced by a `QubitRegister` object, so several registers can coexist.
* **Cached, read-only operator bases** (`hermitian_basis`, `gell_mann`), plus a new Pauli-string basis for multi-qubit Pauli transfer matrices.
* **Not ported, because Python already has them:** `einsum`/`isum` → `numpy.einsum`;
  `Dim`, `Len` → `.shape`, `len`; `mf` → `print`; `OO[n]` → `np.zeros`.

### Note on `DiscZAB` / `DiscZBA` in the Mathematica package

The Mathematica functions search a grid with step π/20 and then a 40× finer
grid around the best coarse point. This can miss the minimum. For the random
state in `tests/test_mathematica.py`, `DiscZAB` gives 0.061336. Mathematica's
own `ClAB`, evaluated at the direction found by the Python code
(θ ≈ 0.068, near the pole), gives a discord of 0.060938. The test checks
this exactly.

## Mathematica → Python

| Mathematica | Python |
|---|---|
| `m`, `p` | `core.ket0`, `core.ket1` |
| `sx sy sz sp sm id wh cnot` | `core.sx sy sz sp sm id2 wh cnot` (+ `hadamard`, `paulis`) |
| `Psip Psim Phip Phim W3 GHZ3` | `core.psi_plus psi_minus phi_plus phi_minus w3 ghz3` (+ `ghz_state(n)`, `w_state(n)`, `ket("010")`, `basis`) |
| `Id[n]`, `OO[n]` | `core.identity(n, sparse=)`, `np.zeros((n, n))` |
| `CircleTimes` (⊗), `CircleDot`, `CirclePlus`, `Todm` | `core.kron`, `core.outer`, `core.direct_sum`, `core.ket2dm` |
| `Comm`, `AComm`, `Conj` | `core.comm`, `core.acomm`, `np.conj` / `symbolic.conj_real` |
| `cf`, `SymMatrix`, `SymVector` | `symbolic.simplify_real`, `symbolic.sym_matrix`, `symbolic.sym_vector` |
| `Trid`, `Tridopen`, `Tridclosed` | `core.tridiagonal(n, up, diag, down, periodic=False/True)` |
| `MatrixSqrt`, `MatrixAbs` | `core.matrix_sqrt`, `core.matrix_abs` |
| `Fidelity`, `QuantumFidelity` | `measures.root_fidelity`, `measures.fidelity` |
| `TraceNorm`, `TraceDistance`, `Purity` | `measures.trace_norm`, `measures.trace_distance`, `measures.purity` |
| `NQubits[n]; Sx[i]` … `Sm[i]` | `reg = qubits.QubitRegister(n); reg.sx[i-1]` … `reg.sm[i-1]` |
| `TightBinding[J, n]` | `qubits.tight_binding(J, n)` |
| `bv` | `qubits.bloch_vector` |
| `RandomSimplex`, `RandomKet`, `GinibreMatrix` | `random_states.random_simplex`, `random_ket`, `ginibre` |
| `RandomUnitary`, `RandomSpecialUnitary`, `RandomOrthogonal` | `random_states.random_unitary`, `random_special_unitary`, `random_orthogonal` |
| `RandomState[d, "HS"/"Bures"/k]`, `RandomPOVM` | `random_states.random_state(d, "hs"/"bures"/k)`, `random_povm` |
| `RandomKrausOPS[n, k]` | `random_states.random_kraus(2**n, k)` |
| `Vec`, `Unvec`, `Res`, `Unres` | `core.vec`, `unvec`, `res`, `unres` |
| `Reshuffle` | `subsystems.reshuffle` |
| `PartialTrace`, `PartialTrace1/2`, `PartialTraceGeneral` | `subsystems.partial_trace(a, dims, sys)` (0-based `sys`) |
| `PartialTranspose` | `subsystems.partial_transpose(a, dims, sys)` |
| `Depo`, `AD`, `Pauli` | `channels.depolarizing`, `amplitude_damping`, `pauli_channel` |
| `vonNeumann`, `Renyi`, `KullbackLeibler` | `entropy.von_neumann`, `renyi`, `relative_entropy` |
| `MutualInformation`, `TripartiteInformation` | `entropy.mutual_information`, `tripartite_information` |
| `shannonEntropy`, `marginalProbabilityDistribution` | `entropy.shannon_entropy`, `entropy.marginal` (0-based variable) |
| `MutualInformationClassic` | `entropy.classical_mutual_information` |
| `conditionalEntropyXcY`, `conditionalEntropyYcX` | `entropy.conditional_entropy(p, sizes, given=1)`, `given=0` |
| `Concurrence`, `ConcurrenceX`, `EntanglementOfFormation`, `Negativity` | `entanglement.concurrence`, `concurrence_x`, `entanglement_of_formation`, `negativity` |
| `ClAB[ρ][θ, φ]`, `ClBA[ρ][θ, φ]` | `discord.conditional_entropy_after_measurement(ρ, θ, φ, measured=0)`, `measured=1` |
| `DiscZAB`, `DiscZBA` | `discord.quantum_discord(ρ, measured=0)`, `measured=1` |
| `GeomDiscordAB`, `GeomDiscordBA` | `discord.geometric_discord(ρ, measured=1)`, `measured=0` (the measured qubit is explicit) |
| `𝒟[L][ρ]`, `𝒟c[L][X]`, `𝒟cascAB[A, B][ρ]` | `lindblad.dissipator(L, ρ)`, `adjoint_dissipator(L, X)`, `cascade_dissipator(A, B, ρ)` |
| `ME[H, ops]` | `lindblad.lindbladian(H, ops)` (returns a function) |
| `HToVec`, `𝒟ToVec`, `𝒟cascABToVec`, `METoVec` | `lindblad.hamiltonian_superop`, `dissipator_superop`, `cascade_superop`, `liouvillian` |
| `MESolvet[ℒ, ρ0, ti, tf]` | `lindblad.solve_generator(ℒ, ρ0, [ti, tf])` or `mesolve(H, ops, ρ0, times, method="ode")` |
| `METoVecSolve` | `lindblad.mesolve(H, ops, ρ0, times)` (numeric) / `symbolic.mesolve` (closed form) |
| `MESolve[ℒ, ρ0]` | `symbolic.solve_linear(ℒ, ρ0)` |
| `Vec2`, `UnVec2` | `bases.basis_coefficients`, `bases.from_coefficients` |
| `Lambda1/2/3`, `GeneralizedPauliMatrices` | `bases.gell_mann_symmetric/antisymmetric/diagonal` (0-based), `bases.gell_mann` |
| `GEN`, `TAU` | `bases.hermitian_basis`, `bases.matrix_units` |
| `StateToBloch`, `BlochToState` | `bases.state_to_bloch`, `bases.bloch_to_state` |
| `KrausToϕ`, `KrausToF`, `KrausToS` | `channels.apply_kraus`, `kraus_to_ptm`, `kraus_to_choi` |
| `FToϕ`, `ϕToF`, `FToS`, `SToF` | `channels.apply_ptm`, `map_to_ptm`, `ptm_to_choi`, `choi_to_ptm` |
| `FToKraus`, `ϕToKraus`, `SToKraus` | `channels.ptm_to_kraus`, `map_to_kraus`, `choi_to_kraus` |
| option `"Basis" -> b` | keyword `basis=b` (stack `(d², d, d)`) |
| `einsum`, `isum` | `numpy.einsum`, e.g. `np.einsum("ij,jk->ik", A, B)` |
| `sfera`, `assi`, `point`, `arrow`, `line` | `bloch.BlochSphere()`, `.add_points`, `.add_vectors`, `.add_line` |
