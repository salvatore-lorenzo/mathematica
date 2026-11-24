# MyQubits (Python): functions by file

Public functions, classes and constants of each module in `myqubits/`, in source order, with the first line of each docstring. Use `help(myqubits.<module>.<name>)` for the full documentation and see [README.md](README.md) for the Mathematica → Python name mapping.

Indices of subsystems and qubits start at 0 (they started at 1 in Mathematica).

## Contents

- [`core.py`](#corepy) — Basic definitions shared by every other `myqubits` module.
- [`subsystems.py`](#subsystemspy) — Operations on composite systems: partial trace, partial transpose, subsystem permutation and reshuffling (realignment).
- [`measures.py`](#measurespy) — Distance and similarity measures between quantum states.
- [`entropy.py`](#entropypy) — Entropies and information measures, quantum and classical.
- [`entanglement.py`](#entanglementpy) — Entanglement measures: concurrence, entanglement of formation, negativity.
- [`discord.py`](#discordpy) — Quantum and geometric discord of two-qubit states.
- [`bases.py`](#basespy) — Operator bases: generalized Gell-Mann matrices, orthonormal Hermitian bases, Pauli-string bases, matrix units and generalized Bloch vectors.
- [`channels.py`](#channelspy) — Quantum channels and their representations.
- [`lindblad.py`](#lindbladpy) — Lindblad and cascaded master equations.
- [`qubits.py`](#qubitspy) — Multi-qubit registers, spin-chain Hamiltonians and single-qubit Bloch vectors.
- [`random_states.py`](#random_statespy) — Random states, unitaries, POVMs and channels.
- [`bloch.py`](#blochpy) — Bloch-sphere plots with matplotlib (optional dependency).
- [`symbolic.py`](#symbolicpy) — Symbolic computations with SymPy (optional dependency).
- [`__init__.py`](#__init__py)
- [Tests](#tests-tests)

## `core.py`

Basic definitions shared by every other `myqubits` module.

`from myqubits import core`  ·  50 public names

**Constants**

- `ATOL` — Absolute threshold below which eigenvalues and probabilities count as zero.
- `ket0` — `|0>` (`m` in Mathematica)
- `ket1` — `|1>` (`p` in Mathematica)
- `sx` — Pauli X
- `sy` — Pauli Y
- `sz` — Pauli Z
- `sp` — raising `|1><0| = (sx - i sy)/2`
- `sm` — lowering `|0><1| = (sx + i sy)/2`
- `id2` — 2x2 identity (`id`)
- `paulis` — stack `(sx, sy, sz)`, shape (3, 2, 2)
- `wh` — unnormalized Walsh-Hadamard
- `hadamard` — Hadamard gate `wh/sqrt(2)`
- `cnot` — CNOT, control = qubit 0
- `phi_plus` — `(|00> + |11>)/sqrt(2)`
- `phi_minus` — `(|00> - |11>)/sqrt(2)`
- `psi_plus` — `(|01> + |10>)/sqrt(2)`
- `psi_minus` — `(|01> - |10>)/sqrt(2)`
- `w3` — `(|001> + |010> + |100>)/sqrt(3)`
- `ghz3` — `(|000> + |111>)/sqrt(2)`

**Kets, gates and standard states**

- `basis(dims, index)` — Computational basis ket.
- `ket(label)` — Multi-qubit computational basis ket from a bit string, e.g. `ket("010")`.
- `ghz_state(n)` — `n`-qubit GHZ state `(|0...0> + |1...1>)/sqrt(2)`.
- `w_state(n)` — `n`-qubit W state, the uniform superposition of the single-excitation kets.
- `identity(n, sparse=False)` — `n x n` identity, dense or as a CSR sparse array.

**Basic algebra**

- `dag(a)` — Conjugate transpose; works on sparse arrays and on stacks `(..., m, n)`.
- `comm(a, b)` — Commutator `[A, B] = A B - B A`.
- `acomm(a, b)` — Anticommutator `{A, B} = A B + B A`.
- `kron(*ops)` — Kronecker (tensor) product of any number of kets or matrices.
- `tensor_power(op, n)` — `op (x) op (x) ... (x) op` with `n` factors.
- `outer(u, v)` — Outer product `u_i v_j` *without* conjugation (`CircleDot`).
- `ket2dm(psi)` — Projector `|psi><psi|` (not normalized); a 2-D input is a stack of kets.
- `as_dm(state)` — Density matrix of `state`: kets (1-D or column) are turned into projectors.
- `direct_sum(*ops)` — Block-diagonal direct sum `A (+) B (+) ...`; empty arrays are neutral.
- `tridiagonal(n, up, diag, down, *, periodic=False, sparse=True)` — (Block-)tridiagonal `n x n` matrix (`Trid`, `Tridopen`, `Tridclosed`).

**Matrix functions**

- `hermitian_function(a, f)` — `f(A) = V f(Λ) V†` for a Hermitian matrix (or stack) `A = V Λ V†`.
- `psd_sqrt(a)` — Square root of a positive semidefinite matrix; tiny negative eigenvalues are clipped.
- `matrix_sqrt(a)` — Principal square root (`MatrixSqrt`).
- `matrix_abs(a)` — `|A| = sqrt(A A†)` (`MatrixAbs`), computed from the SVD `A = U Σ V†` as `U Σ U†`.

**Checks**

- `is_hermitian(a, atol=1e-10)` — `A == A†` within `atol`.
- `is_unitary(u, atol=1e-10)` — `U† U == I` within `atol`.
- `is_psd(a, atol=1e-10)` — Hermitian with no eigenvalue below `-atol`.
- `is_density_matrix(rho, atol=1e-10)` — Positive semidefinite with unit trace.

**Vectorization and superoperators (column stacking)**

- `vec(a)` — Column-stacking vectorization; a stack `(..., m, n)` gives `(..., m n)`.
- `unvec(v, rows=None)` — Inverse of `vec`: fill columns of length `rows` (default: square matrix).
- `res(a)` — Row-stacking vectorization (`Flatten`); a stack `(..., m, n)` gives `(..., m n)`.
- `unres(v, cols=None)` — Inverse of `res`: fill rows of length `cols` (default: square matrix).
- `spre(a)` — Superoperator of `X -> A X`: `kron(I, A)`.
- `spost(b)` — Superoperator of `X -> X B`: `kron(B.T, I)`.
- `sprepost(a, b)` — Superoperator of `X -> A X B`: `kron(B.T, A)`.
- `superop_from_map(fn, d, d_out=None)` — Matrix of a linear map on `d x d` matrices acting on column-stacked vectors.

Internal helpers: `_frozen`, `_dense`, `_square_dim`, `_check_square`, `_operator_list`, `_operator_stack`, `_bands`, `_eye_like`

## `subsystems.py`

Operations on composite systems: partial trace, partial transpose, subsystem permutation and reshuffling (realignment).

`from myqubits import subsystems`  ·  4 public names

**Functions**

- `partial_trace(a, dims=None, sys=None, *, keep=None)` — Trace out the subsystems `sys` (an index or a list of indices).
- `partial_transpose(a, dims=None, sys=0)` — Partial transpose with respect to the subsystems `sys` (default: the first).
- `permute_subsystems(a, dims, perm)` — Reorder the tensor factors: subsystem `k` of the result is subsystem `perm[k]` of `a`.
- `reshuffle(a, dims=None)` — Realignment `R[(i,j),(k,l)] = A[(i,k),(j,l)]` (`Reshuffle`).

Internal helpers: `_resolve_dims`, `_resolve_sys`, `_split`, `_partial_trace_sparse`

## `measures.py`

Distance and similarity measures between quantum states.

`from myqubits import measures`  ·  6 public names

**Functions**

- `root_fidelity(rho, sigma)` — Root fidelity `Tr sqrt(sqrt(rho) sigma sqrt(rho))` (Mathematica `Fidelity`).
- `fidelity(rho, sigma)` — Uhlmann fidelity `(Tr sqrt(sqrt(rho) sigma sqrt(rho)))**2` (`QuantumFidelity`).
- `trace_norm(a)` — Sum of the singular values (absolute eigenvalues for Hermitian input).
- `trace_distance(rho, sigma)` — `||rho - sigma||_1 / 2`.
- `hilbert_schmidt_distance(rho, sigma)` — Frobenius norm `||rho - sigma||_2`.
- `purity(rho)` — `Tr rho^2` in O(d^2) operations; works on stacks `(..., d, d)`.

## `entropy.py`

Entropies and information measures, quantum and classical.

`from myqubits import entropy`  ·  11 public names

**Quantum entropies**

- `von_neumann(rho, base=2, tol=ATOL)` — Von Neumann entropy `-Tr rho log rho`.
- `renyi(rho, alpha, base=2, tol=ATOL)` — Rényi entropy `log(Tr rho^alpha) / (1 - alpha)` over the support of `rho`.
- `relative_entropy(rho, sigma, base=2, tol=ATOL)` — Quantum relative entropy `S(rho||sigma) = Tr rho (log rho - log sigma)` (`KullbackLeibler`).

**Classical information**

- `shannon_entropy(probs, base=2, tol=ATOL)` — Shannon entropy of a probability vector (of each row, for a 2-D input).
- `binary_entropy(p, base=2, tol=ATOL)` — `h(p) = -p log p - (1 - p) log(1 - p)`, elementwise.
- `marginal(probs, sizes, keep)` — Marginal distribution of the variables `keep` (`marginalProbabilityDistribution`).
- `classical_mutual_information(probs, sizes=None, base=2)` — `H(X) + H(Y) - H(X, Y)` of a joint distribution `P[x, y]`.
- `conditional_entropy(probs, sizes=None, given=1, base=2)` — Conditional Shannon entropy `H(X|Y) = H(X, Y) - H(Y)` (`given=1`) or `H(Y|X)` (`given=0`).

**Quantum information**

- `mutual_information(rho, dims=(2, 2), a=0, b=None, base=2)` — Quantum mutual information `I(A:B) = S(A) + S(B) - S(AB)`.
- `tripartite_information(rho, dims=(2, 2, 2), base=2)` — `I(A:B) + I(A:C) - I(A:BC) = S_A + S_B + S_C - S_AB - S_AC - S_BC + S_ABC`.
- `quantum_conditional_entropy(rho, dims=(2, 2), given=1, base=2)` — `S(A|B) = S(AB) - S(B)` where `B` are the subsystems `given`.

Internal helpers: `_scale`, `_out`, `_entropy_terms`, `_spectrum`, `_joint`, `_chop`, `_entropy_of`

## `entanglement.py`

Entanglement measures: concurrence, entanglement of formation, negativity.

`from myqubits import entanglement`  ·  5 public names

**Functions**

- `concurrence(state)` — Wootters concurrence of a two-qubit density matrix or ket.
- `concurrence_x(rho)` — Concurrence of a two-qubit X state (nonzero only on the diagonal and anti-diagonal).
- `entanglement_of_formation(state, base=2)` — Two-qubit entanglement of formation `h((1 + sqrt(1 - C^2))/2)`.
- `negativity(rho, dims=(2, 2), sys=0)` — Sum of the absolute values of the negative eigenvalues of the partial transpose.
- `log_negativity(rho, dims=(2, 2), sys=0)` — `log2 ||rho^{T_sys}||_1`.

Internal helpers: `_pt_spectrum`

## `discord.py`

Quantum and geometric discord of two-qubit states.

`from myqubits import discord`  ·  4 public names

**Functions**

- `two_qubit_correlations(rho)` — `(a, b, T)`: `a_i = Tr rho (s_i x I)`, `b_j = Tr rho (I x s_j)`, `T_ij = Tr rho (s_i x s_j)`.
- `conditional_entropy_after_measurement(rho, theta, phi, measured=0, base=2)` — Average entropy of the unmeasured qubit after measuring `measured` along `(theta, phi)`.
- `quantum_discord(rho, measured=0, base=2, *, grid=(21, 41), full_output=False)` — Quantum discord `S(M) - S(AB) + min_n sum_k p_k S(rho_other|k)` for projective measurements on `measured`.
- `geometric_discord(rho, measured=0)` — Geometric discord for measurements on `measured`, normalized to 1 for Bell states.

Internal helpers: `_direction`, `_avg_conditional_entropy`, `_measured_index`

## `bases.py`

Operator bases: generalized Gell-Mann matrices, orthonormal Hermitian bases, Pauli-string bases, matrix units and generalized Bloch vectors.

`from myqubits import bases`  ·  11 public names

**Functions**

- `gell_mann_symmetric(j, k, n)` — `|j><k| + |k><j|` (`Lambda1`), for `0 <= j < k < n`.
- `gell_mann_antisymmetric(j, k, n)` — `-i (|j><k| - |k><j|)` (`Lambda2`), for `0 <= j < k < n`.
- `gell_mann_diagonal(l, n)` — `sqrt(2/(l(l+1))) diag(1, ..., 1, -l, 0, ..., 0)` with `l` ones (`Lambda3[l+1, n]`), `1 <= l < n`.
- `gell_mann(n)` — The `n**2 - 1` generalized Gell-Mann matrices (`GeneralizedPauliMatrices`).
- `hermitian_basis(n)` — Orthonormal Hermitian basis `{I/sqrt(n), l_1/sqrt(2), ...}` of `n x n` matrices (`GEN`).
- `pauli_basis(n_qubits)` — Orthonormal basis of normalized Pauli strings `P_1 x ... x P_n / sqrt(2^n)`.
- `matrix_units(n)` — The `n**2` matrix units `|i><j|` in row-major order (`TAU`).
- `basis_coefficients(a, basis=None)` — Coefficients `Tr(G_k† A)` of `A` (or of a stack of matrices) in an orthonormal basis (`Vec2`).
- `from_coefficients(c, basis=None)` — `sum_k c_k G_k`, the inverse of `basis_coefficients` (`UnVec2`).
- `state_to_bloch(state)` — Generalized Bloch vector `sqrt(d) (Tr rho G_1, ..., Tr rho G_{d^2})` (`StateToBloch`).
- `bloch_to_state(v)` — Inverse of `state_to_bloch` (`BlochToState`).

Internal helpers: `_basis`, `_real_if_close`

## `channels.py`

Quantum channels and their representations.

`from myqubits import channels`  ·  26 public names

**Standard channels (they act on single matrices or on stacks of matrices)**

- `depolarizing(rho, p)` — `(1 - p) rho + p Tr(rho) I/d` (`Depo`).
- `amplitude_damping_kraus(gamma)` — Kraus operators `diag(1, sqrt(1 - gamma))` and `sqrt(gamma) sm` (decay `|1> -> |0>`).
- `amplitude_damping(rho, gamma)` — Qubit amplitude damping with decay probability `gamma` (`AD`).
- `pauli_kraus(px, py, pz)` — Kraus operators of `pauli_channel`.
- `pauli_channel(rho, probs)` — `(1 - (px+py+pz)/4) rho + (px sx rho sx + py sy rho sy + pz sz rho sz)/4` (`Pauli`).

**Applying a channel**

- `apply_kraus(kraus, rho)` — `sum_k K_k rho K_k†` (`KrausToPhi`); `rho` may be a stack of matrices.
- `apply_superop(superop, rho, d_out=None)` — `unvec(L @ vec(rho))`; `rho` may be a stack of matrices.
- `apply_choi(choi, rho, dims=None)` — `Phi(rho) = Tr_2[S (I x rho^T)] = sum_ij rho_ij S[:, i, :, j]`.
- `apply_ptm(ptm, rho, basis=None)` — Apply the channel with process matrix `F` (`FToPhi`).

**Conversions**

- `kraus_to_superop(kraus)` — `sum_k conj(K_k) x K_k` (Kraus operators may be rectangular).
- `kraus_to_choi(kraus)` — `sum_k res(K_k) res(K_k)†` (`KrausToS`).
- `kraus_to_ptm(kraus, basis=None)` — Process matrix of a Kraus channel (`KrausToF`).
- `choi_to_kraus(choi, dims=None, tol=ATOL)` — Minimal Kraus decomposition from the Choi matrix (`SToKraus`).
- `superop_to_choi(superop)` — Choi matrix from the column-stacking superoperator (an index permutation).
- `choi_to_superop(choi, dims=None)` — Inverse of `superop_to_choi`.
- `superop_to_ptm(superop, basis=None)` — `F = V† L V` with `V` the matrix whose columns are `vec(G_b)`.
- `ptm_to_superop(ptm, basis=None)` — `L = V F V†`, the inverse of `superop_to_ptm`.
- `superop_to_kraus(superop, tol=ATOL)` — Minimal Kraus operators from the superoperator (through the Choi matrix).
- `choi_to_ptm(choi, basis=None)` — Process matrix from the Choi matrix (`SToF`).
- `ptm_to_choi(ptm, basis=None)` — Choi matrix from the process matrix (`FToS`).
- `ptm_to_kraus(ptm, basis=None, tol=ATOL)` — Minimal Kraus operators from the process matrix (`FToKraus`).
- `map_to_superop(fn, d)` — Superoperator of a Python function acting on `d x d` matrices.
- `map_to_choi(fn, d)` — Choi matrix of a Python function acting on `d x d` matrices.
- `map_to_ptm(fn, d=None, basis=None)` — Process matrix of a Python function (`PhiToF`); `d` may be inferred from `basis`.
- `map_to_kraus(fn, d, tol=ATOL)` — Minimal Kraus operators of a completely positive Python function (`PhiToKraus`).

**Channel object**

- `class Channel(superop, dims=None)` — A linear map on matrices, stored as its column-stacking superoperator.
    - attributes: `superop`, `dims`
    - `from_kraus(kraus)` *(classmethod)* — Channel from Kraus operators (they may be rectangular).
    - `from_choi(choi, dims=None)` *(classmethod)* — Channel from its Choi matrix; `dims = (d_out, d_in)` for non-square maps.
    - `from_ptm(ptm, basis=None)` *(classmethod)* — Channel from its process matrix in an orthonormal `basis`.
    - `from_map(fn, d)` *(classmethod)* — Channel from a Python function acting on `d x d` matrices.
    - `from_unitary(u)` *(classmethod)* — Unitary channel `rho -> U rho U†`.
    - `identity(d)` *(classmethod)* — Identity channel on dimension `d`.
    - `__call__(rho)` — Apply the channel, `c(rho)`; `rho` may be a stack of matrices.
    - `__matmul__(other)` — Composition: `(self @ other)(rho) == self(other(rho))`.
    - `__pow__(n)` — `c ** n`: the channel applied `n` times.
    - `choi()` — Choi matrix.
    - `kraus(tol=ATOL)` — Minimal Kraus operators (warns if the map is not completely positive).
    - `ptm(basis=None)` — Process matrix in `basis` (default: `myqubits.bases.hermitian_basis`).
    - `is_cp(atol=1e-10)` — Completely positive: positive semidefinite Choi matrix.
    - `is_tp(atol=1e-10)` — `Tr Phi(X) = Tr X`, i.e. `vec(I)^T L = vec(I)^T`.
    - `is_cptp(atol=1e-10)` — Completely positive and trace preserving.
    - `is_unital(atol=1e-10)` — `Phi(I) = I`.

Internal helpers: `_choi_dims`, `_superop_dims`, `_basis_columns`, `_real`

## `lindblad.py`

Lindblad and cascaded master equations.

`from myqubits import lindblad`  ·  13 public names

**Generators acting on matrices**

- `dissipator(L, rho)` — `D[L](rho) = L rho L† - {L†L, rho}/2`.
- `adjoint_dissipator(L, x)` — Heisenberg-picture dissipator `L† X L - {L†L, X}/2` (`Dc`).
- `cascade_dissipator(a, b, rho)` — Cascaded-system generator `[A rho, B†] + [B, rho A†] + D[A](rho) + D[B](rho)`.
- `lindbladian(H, c_ops=())` — The generator `rho -> -i[H, rho] + sum_k D[L_k](rho)` as a function (`ME[H, ops]`).

**Superoperators (column stacking)**

- `hamiltonian_superop(H)` — `-i (I x H - H^T x I)`: `vec(-i[H, rho]) == hamiltonian_superop(H) @ vec(rho)` (`HToVec`).
- `dissipator_superop(L)` — `conj(L) x L - (L†L)^T x I / 2 - I x L†L / 2` (`DToVec`).
- `cascade_superop(a, b)` — Superoperator of `cascade_dissipator` (`DcascABToVec`).
- `liouvillian(H, c_ops=())` — Liouvillian matrix of `lindbladian` (`METoVec`).

**Solutions**

- `class MESolution(times, states)` — States of an evolution on a time grid.
    - attributes: `times`, `states`
    - `__call__(t)` — `sol(t)`: state at any time `t` (scalar or array) in the solved interval.
    - `__len__()` — `len(sol)`: number of time points.
    - `expect(op)` — `Tr(op rho(t))` on the time grid (real for Hermitian `op`).
- `propagate(liouv, rho0, times)` — Exact solution `vec(rho(t)) = exp(L (t - t0)) vec(rho0)` with `t0 = times[0]`.
- `solve_generator(generator, rho0, times, *, time_dependent=False, method='DOP853', rtol=1e-09, atol=1e-11, **ivp_kwargs)` — Integrate `drho/dt = generator(rho)` (or `generator(t, rho)` with `time_dependent=True`) from `rho(times[0]) = rho0` (`MESolvet`).
- `mesolve(H, c_ops, rho0, times, *, method='expm', **ivp_kwargs)` — Solve the Lindblad equation with Hamiltonian `H` and jump operators `c_ops`.
- `steady_state(liouv)` — Stationary state `L vec(rho) = 0` with `Tr rho = 1` (assumed unique).

Internal helpers: `_op`, `_times`

## `qubits.py`

Multi-qubit registers, spin-chain Hamiltonians and single-qubit Bloch vectors.

`from myqubits import qubits`  ·  6 public names

**Functions**

- `local_operator(op, i, dims, sparse=True)` — `I x ... x op x ... x I` with `op` on subsystem `i` of a system with dimensions `dims`.
- `qubit_operator(op, i, n, sparse=True)` — Single-qubit operator `op` acting on qubit `i` of an `n`-qubit register.
- `multi_qubit_operator(ops, n, sparse=True)` — Tensor product with `ops[i]` on qubit `i` and the identity elsewhere.
- `class QubitRegister(n, sparse=True)` — Pauli and ladder operators of an `n`-qubit register (`NQubits`).
    - attributes: `n`, `sparse`
    - `dim` *(property)* — Hilbert-space dimension `2**n`.
    - `op(a, i)` — `a` acting on qubit `i`.
    - `product(ops)` — See `multi_qubit_operator`.
    - `identity()` — Identity on the register.
    - `sx` *(cached property)* — Tuple of the `n` operators `sx` acting on each qubit.
    - `sy` *(cached property)* — Tuple of the `n` operators `sy` acting on each qubit.
    - `sz` *(cached property)* — Tuple of the `n` operators `sz` acting on each qubit.
    - `sp` *(cached property)* — Tuple of the `n` raising operators `|1><0|` acting on each qubit.
    - `sm` *(cached property)* — Tuple of the `n` lowering operators `|0><1|` acting on each qubit.
- `tight_binding(J, n, periodic=False, sparse=True)` — XX chain `sum_i J_i (sx_i sx_{i+1} + sy_i sy_{i+1})` on `n` qubits (`TightBinding`).
- `bloch_vector(state)` — `(Tr rho sx, Tr rho sy, Tr rho sz)` of a qubit ket, density matrix or stack `(..., 2, 2)` (`bv`).

## `random_states.py`

Random states, unitaries, POVMs and channels.

`from myqubits import random_states`  ·  9 public names

**Functions**

- `random_simplex(d, size=None, rng=None)` — Probability vector uniformly distributed on the simplex (flat Dirichlet).
- `random_ket(d, size=None, rng=None)` — Haar-random normalized ket(s): a normalized complex Gaussian vector.
- `ginibre(m, n=None, size=None, rng=None)` — `m x n` matrix with independent standard normal real and imaginary parts.
- `random_unitary(d, size=None, rng=None)` — Haar-random `d x d` unitary (QR of a Ginibre matrix with phases fixed).
- `random_special_unitary(d, size=None, rng=None)` — Haar-random unitary divided by a `d`-th root of its determinant.
- `random_orthogonal(d, size=None, rng=None)` — Haar-random real orthogonal `d x d` matrix.
- `random_state(d, kind='hs', size=None, rng=None)` — Random density matrix.
- `random_povm(d, k, rng=None)` — Random POVM: `k` positive `d x d` matrices summing to the identity, shape `(k, d, d)`.
- `random_kraus(d, k, rng=None)` — `k` random Kraus operators of a trace-preserving channel on dimension `d`, shape `(k, d, d)`.

Internal helpers: `_rng`, `_size`, `_haar_from_qr`

## `bloch.py`

Bloch-sphere plots with matplotlib (optional dependency).

`from myqubits import bloch`  ·  2 public names

**Functions**

- `bloch_points(states)` — Real Bloch coordinates, shape `(n, 3)`, of one or many qubit states.
- `class BlochSphere(ax=None, *, figsize=(5, 5), mesh=7, sphere_color='white', sphere_alpha=0.3, wire_color='gray', show_axes=True, labels=('x', 'y', 'z'))` — A Bloch sphere drawn on a 3-D matplotlib axis (`sfera` + `assi`).
    - attributes: `fig`, `ax`
    - `add_points(states, color='C0', size=30, **kwargs)` — Scatter the Bloch vectors of `states` (`point`).
    - `add_vectors(states, color='C0', **kwargs)` — Arrows from the origin to the Bloch vectors of `states` (`arrow`).
    - `add_line(states, color='C0', markers=True, **kwargs)` — Polyline through the Bloch vectors of `states`, e.g. a trajectory (`line`).
    - `show()` — Show the figure (`plt.show()`).
    - `save(path, **kwargs)` — Save the figure (`fig.savefig`).

## `symbolic.py`

Symbolic computations with SymPy (optional dependency).

`from myqubits import symbolic`  ·  31 public names

**Constants**

- `tau` — Default time symbol (Mathematica's global `\[Tau]`).
- `ket0` — `|0>` as a SymPy matrix.
- `ket1` — `|1>` as a SymPy matrix.
- `sx` — Pauli X.
- `sy` — Pauli Y.
- `sz` — Pauli Z.
- `sp` — Raising `|1><0|`.
- `sm` — Lowering `|0><1|`.
- `id2` — 2x2 identity.
- `hadamard` — Hadamard gate with an exact `1/sqrt(2)`.

**Construction and elementary operations**

- `sym_matrix(name, rows, cols=None, real=True)` — Matrix of symbols `name_ij` (1-based, as `SymMatrix`); real by default.
- `sym_vector(name, n, real=True)` — Column of symbols `name_i` (`SymVector`).
- `to_sympy(a, rational=True)` — Convert a NumPy array to a SymPy matrix, turning floats into exact numbers if `rational`.
- `conj_real(expr)` — Complex conjugate obtained by flipping the sign of `I`, i.e. treating every symbol as real (`Conj`).
- `simplify_real(expr)` — `simplify(expand_complex(expr))` elementwise (`cf`).
- `dag(m)` — Conjugate transpose (`m.H`).
- `comm(a, b)` — Commutator `A B - B A`.
- `acomm(a, b)` — Anticommutator `A B + B A`.
- `kron(*ms)` — Kronecker product of any number of SymPy matrices.
- `ket2dm(psi, real_symbols=False)` — `|psi><psi|`; with `real_symbols=True` conjugation is `conj_real` (`Todm`).
- `vec(m)` — Column stacking.
- `unvec(v, rows=None)` — Inverse of `vec` (columns of length `rows`, default: square matrix).

**Composite systems (reusing the NumPy algorithms on object arrays)**

- `partial_trace(m, dims=None, sys=None, *, keep=None)` — Symbolic partial trace (same conventions as `myqubits.subsystems.partial_trace`).
- `partial_transpose(m, dims=None, sys=0)` — Symbolic partial transpose (same conventions as `myqubits.subsystems.partial_transpose`).
- `concurrence_x(rho)` — Symbolic concurrence of a two-qubit X state (`ConcurrenceX`).

**Master equations**

- `dissipator(L, rho)` — `L rho L† - {L†L, rho}/2`.
- `lindbladian(H, c_ops=())` — `rho -> -I [H, rho] + sum_k D[L_k](rho)` (`ME[H, ops]`).
- `liouvillian(H, c_ops=())` — Liouvillian matrix acting on column-stacked vectors (`METoVec`).
- `superop_from_map(fn, d)` — Matrix of a linear map on `d x d` SymPy matrices, acting on column-stacked vectors.
- `solve_linear(generator, rho0, t=tau, simplify=True)` — Closed-form solution of `drho/dt = generator(rho)` with `rho(0) = rho0` (`MESolve`).
- `mesolve(H, c_ops, rho0, t=tau, simplify=True)` — Closed-form solution of the Lindblad equation (`MESolve[ME[H, ops], rho0]` / `METoVecSolve`).

Internal helpers: `_obj`, `_back`

## `__init__.py`

Re-exports everything in `core` at the top level (`from myqubits import sx, kron`), defines `__version__ = "0.0.4"`, and imports the other modules lazily on first access (`myqubits.entropy`, `myqubits.channels`, …), so `import myqubits` stays fast and the optional modules `bloch` (matplotlib) and `symbolic` (SymPy) are only loaded when used.

## Tests (`tests/`)

Run with `pytest` from the `python_code` folder.

- `test_channels.py` — `test_all_representations_agree`, `test_round_trips`, `test_standard_channels`, `test_channel_object`, `test_rectangular_kraus_partial_trace_channel`, `test_pauli_transfer_matrix_of_cnot_is_signed_permutation`
- `test_core.py` — `test_ladder_conventions`, `test_constants_are_read_only`, `test_states`, `test_kron_dense_and_sparse`, `test_direct_sum`, `test_tridiagonal_scalar_vector_block`, `test_vectorization`, `test_superoperators`, `test_matrix_functions`, `test_checks_and_as_dm`
- `test_information.py` — `test_bell_state_values`, `test_product_state_has_no_correlations`, `test_werner_state`, `test_concurrence_ket_matches_density_matrix_and_x_formula`, `test_discord_full_output_and_local_unitary_invariance`, `test_fidelity_and_distances`, `test_entropies`, `test_classical_information`
- `test_lindblad.py` — `test_generators_match_superoperators`, `test_sparse_liouvillian`, `test_amplitude_damping_analytic`, `test_propagation_paths_agree`, `test_time_dependent_generator`, `test_steady_state`, `test_von_neumann_equation_is_unitary`
- `test_mathematica.py` — `test_matches_mathematica`, `test_discord_matches_mathematica`
- `test_optional.py` — `test_symbolic`, `test_bloch_sphere`
- `test_qubits_bases_random.py` — `test_register_operators`, `test_tight_binding_matches_kron`, `test_bloch_vector`, `test_random_objects`, `test_bases_orthonormal`, `test_bases_specifics`
- `test_subsystems.py` — `test_partial_trace_against_loops`, `test_partial_trace_variants`, `test_partial_transpose`, `test_permute_subsystems`, `test_reshuffle`
