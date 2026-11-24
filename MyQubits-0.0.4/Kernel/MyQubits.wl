(* ::Package:: *)

(* :Name: MyQubits` *)
(* :Title: MyQubits -- quantum information and open quantum systems *)
(* :Version: 0.0.4 *)
(* :Mathematica Version: 13.3+ *)

(* :Summary:
   Utilities for finite-dimensional quantum information theory and open quantum systems:
   states, operators, partial traces, entropies and correlation measures, quantum channels
   (Kraus, process matrix, Choi matrix), Lindblad master equations and Bloch-sphere graphics. *)

(* :Conventions:
   - Kets are flat lists; operators and density matrices are square matrices (lists or SparseArray).
   - Computational basis: m = |0> = {1, 0}, p = |1> = {0, 1}.
     sm = |0><1| maps |1> to |0>, sp = |1><0| maps |0> to |1>.
   - Tensor products are ordered left to right: in A \[CircleTimes] B, A acts on subsystem 1.
   - Entropies use base-2 logarithms unless an explicit base is given.
   - Vec/Unvec stack columns, Res/Unres stack rows; the ...ToVec superoperators act on Vec.
   - Conj and Todm treat every symbol as real.
   - Functions solving master equations symbolically use the global symbol \[Tau] as time
     variable unless another symbol is given. *)


BeginPackage["MyQubits`"];


(* ::Section::Closed:: *)
(*HELPS*)


$MyQubitsVersion::usage = "$MyQubitsVersion is a string giving the version of the loaded MyQubits package.";


(* ::Subsection:: *)
(*Aliases*)


mf::usage = "mf[A] prints A in MatrixForm and returns A unchanged, so that it can be inserted anywhere in a computation.";
cf::usage = "cf[expr] applies ComplexExpand and then FullSimplify to expr, i.e. simplifies it treating all symbols as real.";
Dim::usage = "Dim[expr] is a shorthand for Dimensions[expr].";
Len::usage = "Len[expr] is a shorthand for Length[expr].";
Conj::usage = "Conj[expr] complex conjugates expr by flipping the sign of every explicit imaginary unit, treating all symbols as real. Unlike Conjugate it never produces Conjugate[\[Ellipsis]] wrappers. SparseArray objects are supported.";


(* ::Subsection::Closed:: *)
(*Commutator and AntiCommutators*)


Comm::usage = "Comm[A, B] gives the commutator A.B - B.A.";
AComm::usage = "AComm[A, B] gives the anticommutator A.B + B.A.";


(* ::Subsection:: *)
(*Symbolic Matrix and Vector*)


SymMatrix::usage = "SymMatrix[a, n] gives the n\[Times]n matrix with symbolic entries Subscript[a, i, j].\nSymMatrix[a, n1, n2] gives the n1\[Times]n2 matrix with symbolic entries Subscript[a, i, j].";
SymVector::usage = "SymVector[a, n] gives the n\[Times]1 column matrix SymMatrix[a, n, 1], with symbolic entries Subscript[a, i, 1]. Use Flatten[SymVector[a, n]] to obtain a flat list (ket).";


(* ::Subsection:: *)
(*KroneckerProduct, BraKet, DirectSum*)


CircleTimes::usage = "CircleTimes[A, B, \[Ellipsis]] or A \[CircleTimes] B \[CircleTimes] \[Ellipsis] gives the Kronecker (tensor) product of matrices, or of kets (flat lists), returning a matrix or a flat list respectively.\nCircleTimes[{A1, A2, \[Ellipsis]}] gives the Kronecker product of a list of matrices (a list of equal-length kets is a matrix, so use CircleTimes @@ list for kets).";
CircleDot::usage = "CircleDot[u, v] or u \[CircleDot] v gives the outer-product matrix with entries u[[i]] v[[j]], without complex conjugation: |\[Psi]\[RightAngleBracket]\[LeftAngleBracket]\[Phi]| is \[Psi] \[CircleDot] Conjugate[\[Phi]].";
CirclePlus::usage = "CirclePlus[A, B, \[Ellipsis]] or A \[CirclePlus] B \[CirclePlus] \[Ellipsis] gives the direct sum (block-diagonal matrix) of the matrices A, B, \[Ellipsis]; the empty list {} acts as the neutral element.\nCirclePlus[{A1, A2, \[Ellipsis]}] gives the direct sum of a list of matrices.";
Todm::usage = "Todm[\[Psi]] gives the projector \[Psi] \[CircleDot] Conj[\[Psi]] = |\[Psi]\[RightAngleBracket]\[LeftAngleBracket]\[Psi]| of the ket \[Psi]. The ket is not normalized, and symbols are treated as real (see Conj).";


(* ::Subsection:: *)
(*Commonly used matrices*)


m::usage = "m is the computational-basis ket |0\[RightAngleBracket] = {1, 0}.";
p::usage = "p is the computational-basis ket |1\[RightAngleBracket] = {0, 1}.";
sx::usage = "sx is the Pauli matrix \[Sigma]x = {{0, 1}, {1, 0}}.";
sy::usage = "sy is the Pauli matrix \[Sigma]y = {{0, -I}, {I, 0}}.";
sz::usage = "sz is the Pauli matrix \[Sigma]z = {{1, 0}, {0, -1}}.";
sm::usage = "sm is the qubit lowering operator (sx + I sy)/2 = |0\[RightAngleBracket]\[LeftAngleBracket]1| = {{0, 1}, {0, 0}}, which maps p = |1\[RightAngleBracket] to m = |0\[RightAngleBracket].";
sp::usage = "sp is the qubit raising operator (sx - I sy)/2 = |1\[RightAngleBracket]\[LeftAngleBracket]0| = {{0, 0}, {1, 0}}, which maps m = |0\[RightAngleBracket] to p = |1\[RightAngleBracket].";
id::usage = "id is the 2\[Times]2 identity matrix.";
wh::usage = "wh is the unnormalized Walsh\[Dash]Hadamard matrix {{1, 1}, {1, -1}}; the Hadamard gate is wh/Sqrt[2].";
cnot::usage = "cnot is the 4\[Times]4 controlled-NOT gate, with the first qubit as control and the second as target.";
Id::usage = "Id[n] gives the n\[Times]n identity matrix.";
OO::usage = "OO[n] gives the n\[Times]n zero matrix as a SparseArray. As a special case OO[1] gives the empty list {}, the neutral element of CirclePlus.";
Psip::usage = "Psip is the Bell state |\[CapitalPsi]+\[RightAngleBracket] = (|01\[RightAngleBracket] + |10\[RightAngleBracket])/Sqrt[2].";
Psim::usage = "Psim is the Bell state |\[CapitalPsi]-\[RightAngleBracket] = (|01\[RightAngleBracket] - |10\[RightAngleBracket])/Sqrt[2].";
Phip::usage = "Phip is the Bell state |\[CapitalPhi]+\[RightAngleBracket] = (|00\[RightAngleBracket] + |11\[RightAngleBracket])/Sqrt[2].";
Phim::usage = "Phim is the Bell state |\[CapitalPhi]-\[RightAngleBracket] = (|00\[RightAngleBracket] - |11\[RightAngleBracket])/Sqrt[2].";
W3::usage = "W3 is the three-qubit W state (|001\[RightAngleBracket] + |010\[RightAngleBracket] + |100\[RightAngleBracket])/Sqrt[3].";
GHZ3::usage = "GHZ3 is the three-qubit GHZ state (|000\[RightAngleBracket] + |111\[RightAngleBracket])/Sqrt[2].";
Trid::usage = "Trid[n, a, b, c] gives the n\[Times]n tridiagonal SparseArray with a on the upper diagonal, b on the main diagonal and c on the lower diagonal.\nTrid[n, {a1, \[Ellipsis]}, {b1, \[Ellipsis], bn}, {c1, \[Ellipsis]}] uses position-dependent diagonals; only the first n-1 elements of the off-diagonal lists are used.\nTrid[n, A, B, C] with k\[Times]k matrices A, B, C gives the (k n)\[Times](k n) block-tridiagonal SparseArray with blocks A above, B on and C below the block diagonal.";
Tridopen::usage = "Tridopen[n, up, diag, down] is the same as Trid[n, up, diag, down] (open boundary conditions).";
Tridclosed::usage = "Tridclosed[n, up, diag, down] gives the matrix Trid[n, up, diag, down] with periodic boundary conditions: the upper coupling is added at position (n, 1) and the lower coupling at (1, n). With lists, the n-th elements of the off-diagonal lists are used for the corners; with k\[Times]k matrices the corners are blocks.";


(* ::Subsection:: *)
(*Distance measures*)


MatrixSqrt::usage = "MatrixSqrt[A] gives the principal square root MatrixPower[A, 1/2] of the square matrix A.";
MatrixAbs::usage = "MatrixAbs[A] gives MatrixSqrt[A.A\[Dagger]], the absolute value of the square matrix A (equal to Sqrt[A\[Dagger].A] when A is normal).";
Fidelity::usage = "Fidelity[\[Rho], \[Sigma]] gives the root fidelity Tr[Sqrt[Sqrt[\[Rho]].\[Sigma].Sqrt[\[Rho]]]] of the density matrices \[Rho] and \[Sigma], computed as the sum of the square roots of the eigenvalues of \[Rho].\[Sigma]. QuantumFidelity gives its square.";
TraceNorm::usage = "TraceNorm[A] gives the trace norm (sum of the singular values) of the square matrix A.";
TraceDistance::usage = "TraceDistance[\[Rho], \[Sigma]] gives the trace distance TraceNorm[\[Rho] - \[Sigma]]/2.";


(* ::Subsection::Closed:: *)
(*Qubits Systems*)


NQubits::usage = "NQubits[n] defines Sx[i], Sy[i], Sz[i], Sp[i] and Sm[i] for i = 1, \[Ellipsis], n as the operators sx, sy, sz, sp and sm acting on qubit i of an n-qubit register (2^n\[Times]2^n SparseArray objects). Definitions made by a previous call are removed.";
Sx::usage = "Sx[i] is sx acting on qubit i of the register defined by the last call of NQubits.";
Sy::usage = "Sy[i] is sy acting on qubit i of the register defined by the last call of NQubits.";
Sz::usage = "Sz[i] is sz acting on qubit i of the register defined by the last call of NQubits.";
Sp::usage = "Sp[i] is the raising operator sp acting on qubit i of the register defined by the last call of NQubits.";
Sm::usage = "Sm[i] is the lowering operator sm acting on qubit i of the register defined by the last call of NQubits.";
TightBinding::usage = "TightBinding[J, n] gives the XX (tight-binding) Hamiltonian J \[Sum]_{i=1}^{n-1} (Sx[i].Sx[i+1] + Sy[i].Sy[i+1]) of an open chain of n qubits as a 2^n\[Times]2^n SparseArray. It does not depend on NQubits.";


(* ::Subsection::Closed:: *)
(*Random states*)


RandomSimplex::usage = "RandomSimplex[d] gives a random probability vector of length d, uniformly distributed on the probability simplex.";
RandomKet::usage = "RandomKet[d] gives a Haar-random normalized ket of dimension d.";
GinibreMatrix::usage = "GinibreMatrix[m, n] gives an m\[Times]n complex matrix whose real and imaginary parts are independent standard normal random numbers.";
RandomSpecialUnitary::usage = "RandomSpecialUnitary[d] gives a random d\[Times]d unitary matrix with unit determinant (RandomUnitary divided by a d-th root of its determinant).";
RandomUnitary::usage = "RandomUnitary[d] gives a Haar-random d\[Times]d unitary matrix.";
RandomOrthogonal::usage = "RandomOrthogonal[d] gives a Haar-random d\[Times]d real orthogonal matrix.";
RandomState::usage = "RandomState[d] gives a random d\[Times]d density matrix distributed according to the Hilbert\[Dash]Schmidt measure.\nRandomState[d, \"HS\"] is the same as RandomState[d].\nRandomState[d, \"Bures\"] samples the Bures measure.\nRandomState[d, k] with an integer k \[GreaterEqual] d samples the measure induced by tracing out a k-dimensional environment from a Haar-random pure state.";
RandomPOVM::usage = "RandomPOVM[d, k] gives a random POVM on a d-dimensional system: a list of k positive semidefinite d\[Times]d matrices summing to the identity.";


(* ::Subsection::Closed:: *)
(*Reshaping, vectorization and reshuffling*)


Vec::usage = "Vec[A] gives the column-stacking vectorization Flatten[Transpose[A]] of the matrix A.";
Unvec::usage = "Unvec[v] reshapes a vector of length n^2 into an n\[Times]n matrix by filling columns, inverting Vec.\nUnvec[v, r] fills columns of length r, giving an r\[Times](Length[v]/r) matrix.";
Res::usage = "Res[A] gives the row-stacking vectorization Flatten[A] of the matrix A.";
Unres::usage = "Unres[v] reshapes a vector of length n^2 into an n\[Times]n matrix by filling rows, inverting Res.\nUnres[v, c] fills rows of length c, giving a (Length[v]/c)\[Times]c matrix.";
Reshuffle::usage = "Reshuffle[A] gives the realignment (reshuffling) of the d^2\[Times]d^2 matrix A, with entries R[(i,j),(k,l)] = A[(i,k),(j,l)]: the (i,j)-th d\[Times]d block of A, flattened row-wise, becomes a row of R.\nReshuffle[A, {{n1, n2}, {m1, m2}}] reshuffles an (n1 n2)\[Times](m1 m2) matrix made of n2\[Times]m2 blocks, giving an (n1 m1)\[Times](n2 m2) matrix.";


(* ::Subsection::Closed:: *)
(*Partial trace and transposition*)


PartialTrace::usage = "PartialTrace[A, {d1, d2, \[Ellipsis]}, sys] traces out the subsystems listed in sys (an integer or a list of integers) from the operator A acting on a space with subsystem dimensions {d1, d2, \[Ellipsis]}. Tracing out every subsystem gives Tr[A].\nPartialTrace[A, sys] assumes two subsystems of equal dimension (A of size d^2\[Times]d^2), with sys equal to 1, 2 or {1, 2}.";
PartialTrace1::usage = "PartialTrace1[A] traces out the first of two equal-dimensional subsystems of A.\nPartialTrace1[A, {d1, d2}] traces out the first subsystem (of dimension d1) of a bipartite operator A.";
PartialTrace2::usage = "PartialTrace2[A] traces out the second of two equal-dimensional subsystems of A.\nPartialTrace2[A, {d1, d2}] traces out the second subsystem (of dimension d2) of a bipartite operator A.";
ListReshape::usage = "ListReshape[list, dims] flattens list and reshapes its elements, in order, into a nested list with dimensions dims (like ArrayReshape, but without padding).";
PartialTraceGeneral::usage = "PartialTraceGeneral[A, {d1, d2, \[Ellipsis]}, sys] traces out the subsystems in the list sys from A for any number of subsystems, without argument checks. It is the general algorithm behind PartialTrace, which should normally be preferred.";
PartialTranspose::usage = "PartialTranspose[\[Rho], {d1, d2, \[Ellipsis]}, sys] gives the partial transpose of \[Rho] with respect to the subsystems listed in sys (an integer or a list of integers), for subsystem dimensions {d1, d2, \[Ellipsis]}.";


(* ::Subsection::Closed:: *)
(*Channels*)


Depo::usage = "Depo[\[Rho], p] applies the depolarizing channel \[Rho] \[RightTeeArrow] (1 - p) \[Rho] + p Tr[\[Rho]] I/d to the d\[Times]d matrix \[Rho].";
AD::usage = "AD[\[Rho], \[Gamma]] applies the qubit amplitude-damping channel with decay probability \[Gamma] (decay |1\[RightAngleBracket] \[Rule] |0\[RightAngleBracket]; Kraus operators {{1, 0}, {0, Sqrt[1 - \[Gamma]]}} and Sqrt[\[Gamma]] sm) to the 2\[Times]2 matrix \[Rho].";
Pauli::usage = "Pauli[\[Rho], {px, py, pz}] applies the qubit Pauli channel \[Rho] \[RightTeeArrow] (1 - (px + py + pz)/4) \[Rho] + (px sx.\[Rho].sx + py sy.\[Rho].sy + pz sz.\[Rho].sz)/4 to the 2\[Times]2 matrix \[Rho]; that is, pk/4 is the probability of a \[Sigma]k error. With this parametrization Pauli[\[Rho], {p, p, p}] equals Depo[\[Rho], p].";


(* ::Subsection:: *)
(*Von Neumann, R\[EAcute]nyi, Relative entropy, Fidelity, Mutual Information*)


vonNeumann::usage = "vonNeumann[\[Rho]] gives the von Neumann entropy -Tr[\[Rho] Log2[\[Rho]]] of the density matrix \[Rho]; numerically zero or negative eigenvalues contribute 0.\nvonNeumann[\[Rho], b] uses logarithms to base b.";
Renyi::usage = "Renyi[\[Rho], \[Alpha]] gives the R\[EAcute]nyi entropy Log2[Tr[\[Rho]^\[Alpha]]]/(1 - \[Alpha]) of order \[Alpha] of the density matrix \[Rho], summing over the nonzero eigenvalues; for \[Alpha] = 1 it gives vonNeumann[\[Rho]].\nRenyi[\[Rho], \[Alpha], b] uses logarithms to base b.";
KullbackLeibler::usage = "KullbackLeibler[\[Rho], \[Sigma]] gives the quantum relative entropy S(\[Rho]||\[Sigma]) = Tr[\[Rho] (Log2[\[Rho]] - Log2[\[Sigma]])] of the density matrices \[Rho] and \[Sigma]; it is Infinity when the support of \[Rho] is not contained in that of \[Sigma].\nKullbackLeibler[\[Rho], \[Sigma], b] uses logarithms to base b.";
QuantumFidelity::usage = "QuantumFidelity[\[Rho], \[Sigma]] gives the Uhlmann fidelity (Tr[Sqrt[Sqrt[\[Rho]].\[Sigma].Sqrt[\[Rho]]]])^2 of the density matrices \[Rho] and \[Sigma], i.e. the square of Fidelity[\[Rho], \[Sigma]].";
MutualInformation::usage = "MutualInformation[\[Rho], {dA, dB}] gives the quantum mutual information S(A) + S(B) - S(AB) (base 2) of the bipartite density matrix \[Rho] with subsystem dimensions dA and dB.\nMutualInformation[\[Rho]] assumes two qubits.";
TripartiteInformation::usage = "TripartiteInformation[\[Rho]] gives the tripartite information I(A:B) + I(A:C) - I(A:BC) of a three-qubit density matrix \[Rho].\nTripartiteInformation[\[Rho], {dA, dB, dC}] allows arbitrary subsystem dimensions.";
(* information theory utilities *)
shannonEntropy::usage = "shannonEntropy[probs] gives the Shannon entropy (base 2) of the probability vector probs; zero probabilities contribute 0.\nshannonEntropy[probs, b] uses logarithms to base b.";
marginalProbabilityDistribution::usage = "marginalProbabilityDistribution[probs, {n1, n2, \[Ellipsis]}, k] gives the marginal distribution of the k-th variable, where probs is the joint distribution of variables taking n1, n2, \[Ellipsis] values, flattened in row-major order.";
MutualInformationClassic::usage = "MutualInformationClassic[probs, {nX, nY}] gives the classical mutual information H(X) + H(Y) - H(X, Y) (base 2) of the joint distribution probs of X and Y, a flat list in row-major order (X index first).\nMutualInformationClassic[probs] assumes {nX, nY} = {2, 2}.";
conditionalEntropyXcY::usage = "conditionalEntropyXcY[probs, {nX, nY}] gives the conditional Shannon entropy H(X|Y) (base 2) of the joint distribution probs (flat list, row-major, X index first); values of Y with zero probability contribute 0.\nconditionalEntropyXcY[probs] assumes {nX, nY} = {2, 2}.";
conditionalEntropyYcX::usage = "conditionalEntropyYcX[probs, {nX, nY}] gives the conditional Shannon entropy H(Y|X) (base 2) of the joint distribution probs (flat list, row-major, X index first); values of X with zero probability contribute 0.\nconditionalEntropyYcX[probs] assumes {nX, nY} = {2, 2}.";
Purity::usage = "Purity[\[Rho]] gives the purity Tr[\[Rho].\[Rho]] of the density matrix \[Rho].";


(* ::Subsection::Closed:: *)
(*Concurrence and entanglement of formation*)


Concurrence::usage = "Concurrence[\[Rho]] gives the Wootters concurrence of the two-qubit (4\[Times]4) density matrix \[Rho].\nConcurrence[\[Psi]] gives the concurrence 2 |\[Psi]1 \[Psi]4 - \[Psi]2 \[Psi]3| of the two-qubit ket \[Psi].";
ConcurrenceX::usage = "ConcurrenceX[\[Rho]] gives 2 Max[0, |\[Rho]23| - Sqrt[\[Rho]11 \[Rho]44], |\[Rho]14| - Sqrt[\[Rho]22 \[Rho]33]], the concurrence of a two-qubit X state (nonzero only on the diagonal and antidiagonal). It is valid only for X states but also works for symbolic entries.";
EntanglementOfFormation::usage = "EntanglementOfFormation[\[Rho]] gives the entanglement of formation h((1 + Sqrt[1 - C^2])/2) of a two-qubit density matrix or ket \[Rho], where C = Concurrence[\[Rho]] and h is the binary entropy (base 2).\nEntanglementOfFormation[\[Rho], b] uses logarithms to base b.";
Negativity::usage = "Negativity[\[Rho], {dA, dB}] gives the negativity (sum of the absolute values of the negative eigenvalues of the partial transpose) of the bipartite density matrix \[Rho] with subsystem dimensions dA and dB.\nNegativity[\[Rho]] assumes two qubits.";


(* ::Subsection::Closed:: *)
(*Discord*)


ClAB::usage = "ClAB[\[Rho]][\[Theta], \[Phi]] gives the average conditional entropy \[Sum]k pk S(\[Rho]B|k) (base 2) of qubit B after a projective measurement of qubit A of the two-qubit state \[Rho] along the Bloch direction (\[Theta], \[Phi]). Its minimum over the directions enters DiscZAB. The result is numerical.";
DiscZAB::usage = "DiscZAB[\[Rho]] gives the quantum discord S(A) - S(AB) + min ClAB[\[Rho]] (base 2) of the two-qubit state \[Rho] for projective measurements on qubit A. The minimum is found by a two-stage grid search over the Bloch sphere; the result is numerical.";
ClBA::usage = "ClBA[\[Rho]][\[Theta], \[Phi]] gives the average conditional entropy \[Sum]k pk S(\[Rho]A|k) (base 2) of qubit A after a projective measurement of qubit B of the two-qubit state \[Rho] along the Bloch direction (\[Theta], \[Phi]). Its minimum over the directions enters DiscZBA. The result is numerical.";
DiscZBA::usage = "DiscZBA[\[Rho]] gives the quantum discord S(B) - S(AB) + min ClBA[\[Rho]] (base 2) of the two-qubit state \[Rho] for projective measurements on qubit B. The minimum is found by a two-stage grid search over the Bloch sphere; the result is numerical.";
GeomDiscordAB::usage = "GeomDiscordAB[\[Rho]] gives the geometric discord of the two-qubit state \[Rho] for measurements on qubit B (the second subsystem), normalized to 1 for Bell states: (|y|^2 + ||T||^2 - kmax)/2, where y is the Bloch vector of B, T is the correlation matrix Tij = Tr[\[Rho] \[Sigma]i\[CircleTimes]\[Sigma]j] and kmax is the largest eigenvalue of y y^T + T^T.T. Note that, unlike DiscZAB, the measured qubit is the second one.";
GeomDiscordBA::usage = "GeomDiscordBA[\[Rho]] gives the geometric discord of the two-qubit state \[Rho] for measurements on qubit A (the first subsystem), normalized to 1 for Bell states: (|x|^2 + ||T||^2 - kmax)/2, where x is the Bloch vector of A, T is the correlation matrix Tij = Tr[\[Rho] \[Sigma]i\[CircleTimes]\[Sigma]j] and kmax is the largest eigenvalue of x x^T + T.T^T. Note that, unlike DiscZBA, the measured qubit is the first one.";


(* ::Subsection::Closed:: *)
(*Lindblad ME*)


\[ScriptCapitalD]::usage = "\[ScriptCapitalD][L][\[Rho]] gives the Lindblad dissipator L.\[Rho].L\[Dagger] - (L\[Dagger].L.\[Rho] + \[Rho].L\[Dagger].L)/2.";
\[ScriptCapitalD]c::usage = "\[ScriptCapitalD]c[L][X] gives the adjoint (Heisenberg-picture) dissipator L\[Dagger].X.L - (L\[Dagger].L.X + X.L\[Dagger].L)/2.";
ME::usage = "ME[H, {L1, L2, \[Ellipsis]}][\[Rho]] gives the right-hand side -I [H, \[Rho]] + \[Sum]k \[ScriptCapitalD][Lk][\[Rho]] of the Lindblad master equation. A single jump operator can be given without braces, and {} gives the von Neumann equation. ME[H, ops] can be passed as generator to MESolve and MESolvet.";
MESolvet::usage = "MESolvet[\[ScriptCapitalL], \[Rho]0, ti, tf] solves d\[Rho]/dt = \[ScriptCapitalL][\[Rho]] numerically with NDSolve on [ti, tf] with \[Rho](ti) = \[Rho]0, and returns the matrix-valued InterpolatingFunction \[Rho] (to be evaluated as \[Rho][t]). \[ScriptCapitalL] is any linear function of a matrix, such as ME[H, ops].";
MESolve::usage = "MESolve[\[ScriptCapitalL], \[Rho]0] solves d\[Rho]/d\[Tau] = \[ScriptCapitalL][\[Rho]] symbolically with DSolve with \[Rho](0) = \[Rho]0, and returns \[Rho](\[Tau]) as a matrix in the global symbol \[Tau] (which must have no value).\nMESolve[\[ScriptCapitalL], \[Rho]0, t] uses the symbol t as time variable. \[ScriptCapitalL] is any linear function of a matrix, such as ME[H, ops].";


(* ::Subsection::Closed:: *)
(*Cascade ME*)


\[ScriptCapitalD]cascAB::usage = "\[ScriptCapitalD]cascAB[A, B][\[Rho]] gives the cascaded-system generator [A.\[Rho], B\[Dagger]] + [B, \[Rho].A\[Dagger]] + \[ScriptCapitalD][A][\[Rho]] + \[ScriptCapitalD][B][\[Rho]], describing a source with jump operator A that drives unidirectionally a target with jump operator B.";


(* ::Subsection::Closed:: *)
(*Vectorization *)


Vec2::usage = "Vec2[A] gives the coefficients Tr[A.G] of the d\[Times]d matrix A in the orthonormal Hermitian basis G \[Element] GEN[d], so that UnVec2[Vec2[A]] == A; they are real for Hermitian A. This is not the column stacking on which the ...ToVec superoperators act (use Vec for that).";
UnVec2::usage = "UnVec2[c] gives the d\[Times]d matrix \[Sum]k c[[k]] GEN[d][[k]], with d = Sqrt[Length[c]], inverting Vec2.";
\[ScriptCapitalD]ToVec::usage = "\[ScriptCapitalD]ToVec[L] gives the matrix of the dissipator \[ScriptCapitalD][L] acting on column-stacked vectors: Vec[\[ScriptCapitalD][L][\[Rho]]] == \[ScriptCapitalD]ToVec[L].Vec[\[Rho]].";
\[ScriptCapitalD]cascABToVec::usage = "\[ScriptCapitalD]cascABToVec[A, B] gives the matrix of the cascaded generator \[ScriptCapitalD]cascAB[A, B] acting on column-stacked vectors: Vec[\[ScriptCapitalD]cascAB[A, B][\[Rho]]] == \[ScriptCapitalD]cascABToVec[A, B].Vec[\[Rho]].";
HToVec::usage = "HToVec[H] gives the matrix -I (I\[CircleTimes]H - H^T\[CircleTimes]I) of \[Rho] \[RightTeeArrow] -I [H, \[Rho]] acting on column-stacked vectors: Vec[-I Comm[H, \[Rho]]] == HToVec[H].Vec[\[Rho]].";
METoVec::usage = "METoVec[H, {L1, L2, \[Ellipsis]}] gives the Liouvillian matrix of ME[H, {L1, L2, \[Ellipsis]}] acting on column-stacked vectors: Vec[ME[H, ops][\[Rho]]] == METoVec[H, ops].Vec[\[Rho]].";
METoVecSolve::usage = "METoVecSolve[H, {L1, L2, \[Ellipsis]}, \[Rho]0] gives MatrixExp[METoVec[H, ops] \[Tau]].Vec[\[Rho]0], the column-stacked solution of the Lindblad master equation at time \[Tau] (global symbol); Unvec turns it back into a density matrix.\nMEToVecSolve[H, ops, \[Rho]0, t] uses the symbol t as time variable.";


(* ::Subsection::Closed:: *)
(*Representation*)


Lambda1::usage = "Lambda1[j, k, n] gives the symmetric generalized Gell-Mann matrix |j\[RightAngleBracket]\[LeftAngleBracket]k| + |k\[RightAngleBracket]\[LeftAngleBracket]j| of dimension n, for 1 \[LessEqual] j < k \[LessEqual] n.";
Lambda2::usage = "Lambda2[j, k, n] gives the antisymmetric generalized Gell-Mann matrix -I (|j\[RightAngleBracket]\[LeftAngleBracket]k| - |k\[RightAngleBracket]\[LeftAngleBracket]j|) of dimension n, for 1 \[LessEqual] j < k \[LessEqual] n.";
Lambda3::usage = "Lambda3[l, n] gives the diagonal generalized Gell-Mann matrix Sqrt[2/(l(l-1))] DiagonalMatrix[{1, \[Ellipsis], 1, -(l-1), 0, \[Ellipsis], 0}] of dimension n, for 2 \[LessEqual] l \[LessEqual] n.";
GeneralizedPauliMatrices::usage = "GeneralizedPauliMatrices[n] gives the n^2-1 generalized Gell-Mann matrices of dimension n (traceless, Hermitian, Tr[\[Lambda]i.\[Lambda]j] = 2 \[Delta]ij): all Lambda1, then all Lambda2, then all Lambda3. For n = 2 they are {sx, sy, sz}.";
GEN::usage = "GEN[n] gives the orthonormal Hermitian basis {Id[n]/Sqrt[n], \[Lambda]1/Sqrt[2], \[Lambda]2/Sqrt[2], \[Ellipsis]} of the n\[Times]n matrices, where \[Lambda]k are GeneralizedPauliMatrices[n]; Tr[Gi.Gj] = \[Delta]ij.";
TAU::usage = "TAU[n] gives the n^2 matrix units |i\[RightAngleBracket]\[LeftAngleBracket]j| of dimension n, in row-major order (i outer, j inner).";
StateToBloch::usage = "StateToBloch[\[Rho]] gives the generalized Bloch vector Sqrt[d] {Tr[\[Rho].G1], \[Ellipsis], Tr[\[Rho].G(d^2)]} of the d\[Times]d matrix \[Rho] in the basis G = GEN[d]. The first component is Tr[\[Rho]]; for a qubit the result is {1, Tr[\[Rho].sx], Tr[\[Rho].sy], Tr[\[Rho].sz]}.\nStateToBloch[\[Psi]] gives the Bloch vector of Todm[\[Psi]] for a ket \[Psi].";
BlochToState::usage = "BlochToState[v] inverts StateToBloch: for v of length d^2 it gives v.GEN[d]/Sqrt[d]. A vector of length d^2-1, such as the output of bv for a qubit, is taken to omit the leading component 1.";


(* ::Subsection::Closed:: *)
(*Kraus Operators*)


RandomKrausOPS::usage = "RandomKrausOPS[n, k] gives k random Kraus operators (2^n\[Times]2^n matrices) of a trace-preserving channel on n qubits, with \[Sum] K\[Dagger].K = I.";
KrausTo\[Phi]::usage = "KrausTo\[Phi][{K1, K2, \[Ellipsis]}, \[Rho]] applies the channel \[Rho] \[RightTeeArrow] \[Sum]k Kk.\[Rho].Kk\[Dagger] to \[Rho]. A single Kraus operator can be given without braces.";
KrausToF::usage = "KrausToF[{K1, K2, \[Ellipsis]}] gives the process matrix F of the channel with Kraus operators Kk: F[[\[Alpha], \[Beta]]] = Tr[G\[Alpha]\[Dagger].\[CapitalPhi][G\[Beta]]], where G = GEN[d] (for a qubit, F is the Pauli transfer matrix). The option \"Basis\" -> {G1, \[Ellipsis]} specifies another orthonormal operator basis.";
FTo\[Phi]::usage = "FTo\[Phi][F, \[Rho]] applies the channel with process matrix F (see KrausToF) to the d\[Times]d matrix \[Rho]. Option \"Basis\" as in KrausToF.";
\[Phi]ToF::usage = "\[Phi]ToF[\[CapitalPhi], d] gives the process matrix F (see KrausToF) of the linear map \[CapitalPhi], a function acting on d\[Times]d matrices.\n\[Phi]ToF[\[CapitalPhi], \"Basis\" -> basis] infers d from the given orthonormal operator basis.";
FToS::usage = "FToS[F] gives the Choi matrix S = \[Sum]ij \[CapitalPhi][|i\[RightAngleBracket]\[LeftAngleBracket]j|]\[CircleTimes]|i\[RightAngleBracket]\[LeftAngleBracket]j| (output space first) of the map \[CapitalPhi] with process matrix F. \[CapitalPhi] is completely positive if and only if S is positive semidefinite. Option \"Basis\" as in KrausToF.";
SToF::usage = "SToF[S] gives the process matrix F of the map with Choi matrix S, inverting FToS. Option \"Basis\" as in KrausToF.";
FToKraus::usage = "FToKraus[F] gives a minimal set of Kraus operators of the completely positive map with process matrix F, one for each nonzero eigenvalue of its Choi matrix (see SToKraus). Option \"Basis\" as in KrausToF.";
\[Phi]ToKraus::usage = "\[Phi]ToKraus[\[CapitalPhi], d] gives a minimal set of Kraus operators of the completely positive map \[CapitalPhi] acting on d\[Times]d matrices.";
KrausToS::usage = "KrausToS[{K1, K2, \[Ellipsis]}] gives the Choi matrix S (as in FToS) of the channel with Kraus operators Kk, equal to \[Sum]k Res[Kk] \[CircleDot] Conjugate[Res[Kk]].";
SToKraus::usage = "SToKraus[S] gives a minimal set of Kraus operators of the map with Choi matrix S (as in FToS): each eigenvector v of S with nonzero eigenvalue \[Lambda] yields Sqrt[\[Lambda]] Unres[v]. A message is issued if S has negative eigenvalues (map not completely positive).";


(* ::Subsection::Closed:: *)
(*Einstein Sum*)


einsum::usage = "einsum[{s1, s2, \[Ellipsis]} -> out, A1, A2, \[Ellipsis]] performs an Einstein summation: si lists the index labels of the array Ai, labels appearing twice are contracted, and out lists the remaining labels in the desired output order. For example einsum[{{i, j}, {j, k}} -> {i, k}, A, B] is A.B and einsum[{{i, i}} -> {}, A] is Tr[A].";
isum::usage = "isum[{s1, s2, \[Ellipsis]} -> out, {A1, A2, \[Ellipsis]}] is the low-level implementation of einsum, taking the arrays as a list; it returns $Failed, with a message, for inconsistent index specifications.";


(* ::Subsection:: *)
(*Bloch Sphere*)


bv::usage = "bv[\[Rho]] gives the Bloch vector {Tr[\[Rho].sx], Tr[\[Rho].sy], Tr[\[Rho].sz]} of the 2\[Times]2 matrix \[Rho].";
point::usage = "point[s, col] gives a Graphics3D point of color col at the Bloch vector of the qubit state s (a ket or a 2\[Times]2 density matrix).";
arrow::usage = "arrow[s, col] gives a Graphics3D arrow of color col from the origin to the Bloch vector of the qubit state s (a ket or a 2\[Times]2 density matrix).";
line::usage = "line[{r1, r2, \[Ellipsis]}, col] gives a Graphics3D polyline of color col through the 3D points ri (for example Bloch vectors from bv), marking each point.";
sfera::usage = "sfera is a list of graphics representing the Bloch sphere (meshed wireframe and translucent surface). Combine it with other objects using Show, e.g. Show[sfera, assi, point[\[Rho], Red]].";
assi::usage = "assi is a list of Graphics3D objects drawing the x, y and z axes of the Bloch sphere.";


(* ::Section:: *)
(*PRIVATE DEFINITIONS*)


Begin["`Private`"];


$MyQubitsVersion = "0.0.4";


(* ::Subsection:: *)
(*Shared helpers*)


(* Numerical threshold below which eigenvalues and probabilities are treated as zero. *)
$zeroTolerance = 10^-12;

numericZeroQ[x_] := NumericQ[x] && TrueQ[Abs[N[x]] < $zeroTolerance];

(* -x Log[b, x], with the convention 0 Log[0] = 0; numerically zero or negative numbers give 0
   and spurious imaginary parts of inexact numbers are dropped. *)
entropyTerm[x_, base_] := Which[
	NumericQ[x] && TrueQ[Re[N[x]] < $zeroTolerance], 0,
	InexactNumberQ[x], -Re[x] Log[base, Re[x]],
	True, -x Log[base, x]
];

(* Eigenvalues that are not numerically zero or negative. *)
supportEigenvalues[\[Rho]_] := If[InexactNumberQ[#], Re[#], #] & /@
	Select[Eigenvalues[\[Rho]], !(NumericQ[#] && TrueQ[Re[N[#]] < $zeroTolerance]) &];

(* Eigensystem of a Hermitian matrix with orthonormal eigenvectors: exact Eigensystem returns
   unnormalized (and, for degenerate eigenvalues, non-orthogonal) eigenvectors. Gram-Schmidt keeps
   every vector inside its eigenspace because eigenspaces of a Hermitian matrix are orthogonal. *)
hermitianEigensystem[a_] := Module[{vals, vecs},
	{vals, vecs} = Eigensystem[Normal[a]];
	{vals, Orthogonalize[vecs]}
];

(* Normalizes the operator argument(s) of ME, METoVec, KrausTo\[Phi], ...: a single matrix, a list of
   matrices, or a sequence of matrices (passed here wrapped in a list) all give a list of matrices. *)
toOperatorList[args_List] := Which[
	Length[args] == 1 && MatrixQ[First[args]], args,
	Length[args] == 1 && ListQ[First[args]], First[args],
	True, args
];

qubitMatrixQ[x_] := SquareMatrixQ[x] && Length[x] == 2;
twoQubitMatrixQ[x_] := SquareMatrixQ[x] && Length[x] == 4;
twoQubitStateQ[x_] := twoQubitMatrixQ[x] || (VectorQ[x] && Length[x] == 4);


(* ::Subsection:: *)
(*Aliases*)


mf[A_] := (Print[MatrixForm[A]]; A)
cf[expr_] := FullSimplify[ComplexExpand[expr]];
Dim[in_] := Dimensions[in]
Len[in_] := Length[in]
(* ReplaceAll does not descend into SparseArray objects, which are therefore rebuilt from their rules. *)
Conj[z_] := z /. {
	s_SparseArray :> SparseArray[ArrayRules[s] /. Complex[a_, b_] :> Complex[a, -b], Dimensions[s]],
	Complex[a_, b_] :> Complex[a, -b]}


(* ::Subsection::Closed:: *)
(*Commutator and AntiCommutators*)


Comm[A_, B_] := A . B - B . A;
AComm[A_, B_] := A . B + B . A;


(* ::Subsection::Closed:: *)
(*Symbolic Matrix and Vector*)


SymMatrix[sym_, d1_?IntegerQ] := SymMatrix[sym, d1, d1];
SymMatrix[sym_, d1_?IntegerQ, d2_?IntegerQ] := Table[Subscript[sym, i, j], {i, 1, d1}, {j, 1, d2}];
SymVector[sym_, d1_?IntegerQ] := SymMatrix[sym, d1, 1];


(* ::Subsection::Closed:: *)
(*KroneckerProduct, BraKet, DirectSum*)


(* The recursive rules are restricted to arrays: in 0.0.3 a single ket or any non-array argument
   (e.g. a symbolic a \[CircleTimes] b) recursed until $IterationLimit. *)
CircleTimes[x_?MatrixQ, y_?MatrixQ] := KroneckerProduct[x, y];
CircleTimes[x_?VectorQ, y_?VectorQ] := Flatten[KroneckerProduct[x, y]];
CircleTimes[x_?MatrixQ] := x;
CircleTimes[x_?VectorQ] := x;
CircleTimes[l_List] /; !MatrixQ[l] && l =!= {} && AllTrue[l, ArrayQ] := CircleTimes @@ l;
CircleTimes[a_?ArrayQ, b_?ArrayQ, c__?ArrayQ] := CircleTimes[a, CircleTimes[b, c]];


CircleDot[x_?VectorQ, y_?VectorQ] := Outer[Times, x, y];


CirclePlus[x_?MatrixQ, y_?MatrixQ] := ArrayFlatten[{{x, 0}, {0, y}}];
CirclePlus[x_?MatrixQ, {}] := x;
CirclePlus[{}, y_?MatrixQ] := y;
CirclePlus[{}, {}] := {};
CirclePlus[x_?MatrixQ] := x;
CirclePlus[l_List] /; !MatrixQ[l] && l =!= {} && AllTrue[l, MatrixQ[#] || # === {} &] := CirclePlus @@ l;
CirclePlus[a : (_?MatrixQ | {}), b : (_?MatrixQ | {}), c : (_?MatrixQ | {}) ..] := CirclePlus[a, CirclePlus[b, c]];


Todm[v_?VectorQ] := v \[CircleDot] Conj[v];


(* ::Subsection:: *)
(*Commonly used matrices*)


m = {1, 0}; p = {0, 1};
sx = {{0, 1}, {1, 0}};
sy = {{0, -I}, {I, 0}};
sz = {{1, 0}, {0, -1}};
id = {{1, 0}, {0, 1}};
sp = (sx - I sy)/2; sm = (sx + I sy)/2;
wh = {{1, 1}, {1, -1}};
cnot = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}};
Id[n_] := IdentityMatrix[n];
OO[n_] := SparseArray[{}, {n, n}];
OO[1] = {};

Phip = 1/Sqrt[2] (m \[CircleTimes] m + p \[CircleTimes] p);
Phim = 1/Sqrt[2] (m \[CircleTimes] m - p \[CircleTimes] p);
Psip = 1/Sqrt[2] (m \[CircleTimes] p + p \[CircleTimes] m);
Psim = 1/Sqrt[2] (m \[CircleTimes] p - p \[CircleTimes] m);

W3 = 1/Sqrt[3] (m \[CircleTimes] m \[CircleTimes] p + m \[CircleTimes] p \[CircleTimes] m + p \[CircleTimes] m \[CircleTimes] m);
GHZ3 = 1/Sqrt[2] (m \[CircleTimes] m \[CircleTimes] m + p \[CircleTimes] p \[CircleTimes] p);


Trid[n_Integer, up_?VectorQ, c_?VectorQ, dw_?VectorQ] := SparseArray[{
	Band[{1, 2}] -> up[[1 ;; n - 1]],
	Band[{1, 1}] -> c,
	Band[{2, 1}] -> dw[[1 ;; n - 1]]}, n];
Trid[n_Integer, up_?MatrixQ, c_?MatrixQ, dw_?MatrixQ] := With[{k = Length[c]},
	SparseArray[{
		Band[{1, 1 + k}] -> Table[up, n - 1],
		Band[{1, 1}] -> Table[c, n],
		Band[{1 + k, 1}] -> Table[dw, n - 1]}, k n]];
Trid[n_Integer, a_, b_, c_] := SparseArray[{
	Band[{1, 2}] -> Table[a, n - 1],
	Band[{1, 1}] -> Table[b, n],
	Band[{2, 1}] -> Table[c, n - 1]}, n];

Tridopen[n_, up_, c_, dw_] := Trid[n, up, c, dw];

(* Periodic versions: the corner couplings are added to the open matrix. *)
Tridclosed[n_Integer, up_?VectorQ, c_?VectorQ, dw_?VectorQ] :=
	Trid[n, up, c, dw] + SparseArray[{{1, n} -> dw[[-1]], {n, 1} -> up[[-1]]}, {n, n}];
Tridclosed[n_Integer, up_?MatrixQ, c_?MatrixQ, dw_?MatrixQ] := With[{k = Length[c]},
	Trid[n, up, c, dw] + SparseArray[{
		Band[{1, k (n - 1) + 1}] -> {dw},
		Band[{k (n - 1) + 1, 1}] -> {up}}, {k n, k n}]];
Tridclosed[n_Integer, a_, b_, c_] :=
	Trid[n, a, b, c] + SparseArray[{{1, n} -> c, {n, 1} -> a}, {n, n}];


(* ::Subsection::Closed:: *)
(*Distance measures*)


MatrixSqrt[a_?SquareMatrixQ] := MatrixPower[a, 1/2];

MatrixAbs[a_?SquareMatrixQ] := MatrixSqrt[a . ConjugateTranspose[a]];

Fidelity[a_?SquareMatrixQ, b_?SquareMatrixQ] := Chop[Total[Sqrt[Eigenvalues[a . b]]]];

TraceNorm[a_?SquareMatrixQ] := Total[SingularValueList[a]];

TraceDistance[a_?SquareMatrixQ, b_?SquareMatrixQ] := 1/2 TraceNorm[a - b];


(* ::Subsection:: *)
(*Qubits Systems*)


(* Single-qubit operator op acting on qubit i of an nn-qubit register. *)
qubitOperator[op_, i_Integer, nn_Integer] := SparseArray[KroneckerProduct[
	IdentityMatrix[2^(i - 1), SparseArray], SparseArray[op], IdentityMatrix[2^(nn - i), SparseArray]]];

NQubits[nn_Integer?Positive] := (
	Clear[Sx, Sy, Sz, Sp, Sm];
	Do[
		Sx[i] = qubitOperator[sx, i, nn];
		Sy[i] = qubitOperator[sy, i, nn];
		Sz[i] = qubitOperator[sz, i, nn];
		Sp[i] = qubitOperator[sp, i, nn];
		Sm[i] = qubitOperator[sm, i, nn],
	{i, 1, nn}];
);


TightBinding[J_, nn_Integer?Positive] := If[nn == 1,
	SparseArray[{}, {2, 2}],
	J Sum[
		qubitOperator[sx, i, nn] . qubitOperator[sx, i + 1, nn] +
		qubitOperator[sy, i, nn] . qubitOperator[sy, i + 1, nn], {i, 1, nn - 1}]
];


(* ::Subsection::Closed:: *)
(*Random states*)


RandomSimplex[d_Integer?Positive] := Module[{r},
	r = Sort[Table[RandomReal[{0, 1}], {i, 1, d - 1}]];
	Append[r, 1] - Prepend[r, 0]
];


RandomKet[n_?IntegerQ] := Module[{amplitudes, phases},
	amplitudes = Sqrt[RandomSimplex[n]];
	phases = Prepend[Exp[I*RandomReal[{0, 2 \[Pi]}, n - 1]], 1];
	amplitudes*phases
];


GinibreMatrix[rows_, cols_] := RandomReal[NormalDistribution[0, 1], {rows, cols}] + I RandomReal[NormalDistribution[0, 1], {rows, cols}];

RandomSpecialUnitary[dim_] := Module[{u},
	u = RandomUnitary[dim];
	u/Det[u]^(1/dim)
];

(* QR of a Ginibre matrix with the phases of diag(r) removed (Mezzadri). *)
RandomUnitary[dim_] := Module[{q, r, d, ph},
	{q, r} = QRDecomposition[GinibreMatrix[dim, dim]];
	d = Diagonal[r];
	ph = d/Abs[d];
	Transpose[Transpose[q]*ph]
];

RandomOrthogonal[dim_] := Module[{q, r, d, ph},
	{q, r} = QRDecomposition[RandomReal[NormalDistribution[0, 1], {dim, dim}]];
	d = Diagonal[r];
	ph = d/Abs[d];
	Transpose[Transpose[q]*ph]
];

RandomState[d_, dist_ : "HS"] := Module[{a, u},
	Switch[dist,
		"HS",
			a = GinibreMatrix[d, d];
			a = a . ConjugateTranspose[a];
			Chop[a/Tr[a]],
		"Bures",
			a = GinibreMatrix[d, d];
			u = RandomUnitary[d];
			a = (IdentityMatrix[d] + u) . a . ConjugateTranspose[a] . ConjugateTranspose[IdentityMatrix[d] + u];
			ConjugateTranspose[Chop[a]]/Tr[a],
		_,
			If[IntegerQ[dist] && dist >= d,
				a = GinibreMatrix[d, dist];
				a = a . ConjugateTranspose[a];
				Chop[a/Tr[a]],
				Message[RandomState::argerr, dist, d]; $Failed
			]
	]
];
RandomState::argerr = "The distribution specification `1` is not \"HS\", \"Bures\" or an integer greater than or equal to the dimension `2`.";

RandomPOVM[d_, s_] := Module[{w, sInv},
	w = # . ConjugateTranspose[#] & /@ Table[GinibreMatrix[d, s], s];
	sInv = MatrixPower[Total[w], -1/2];
	sInv . # . sInv & /@ w
];


(* ::Subsection::Closed:: *)
(*Reshaping, vectorization and reshuffling*)


Vec[a_] := Flatten[Transpose[a]];

Unvec[v_List, r_Integer : 0] := Module[{len = Length[v], rows = r},
	If[rows == 0, rows = Sqrt[len]];
	Which[
		r == 0 && !IntegerQ[rows], Message[Unvec::notsq, len]; $Failed,
		rows > 0 && Divisible[len, rows], Transpose[Partition[v, rows]],
		True, Message[Unvec::shape, len, r]; $Failed
	]
];
Unvec::notsq = "Unvec[v] needs a vector whose length is a perfect square (length `1` was given); use Unvec[v, r] to specify the number of rows.";
Unvec::shape = "A vector of length `1` cannot be split into columns of length `2`.";

Res[a_List] := Flatten[a];

Unres[v_List, c_Integer : 0] := Module[{len = Length[v], cols = c},
	If[cols == 0, cols = Sqrt[len]];
	Which[
		c == 0 && !IntegerQ[cols], Message[Unres::notsq, len]; $Failed,
		cols > 0 && Divisible[len, cols], Partition[v, cols],
		True, Message[Unres::shape, len, c]; $Failed
	]
];
Unres::notsq = "Unres[v] needs a vector whose length is a perfect square (length `1` was given); use Unres[v, c] to specify the number of columns.";
Unres::shape = "A vector of length `1` cannot be split into rows of length `2`.";

Reshuffle[\[Rho]_] := Module[{dim = Sqrt[Length[\[Rho]]]},
	If[SquareMatrixQ[\[Rho]] && IntegerQ[dim],
		Reshuffle[\[Rho], {{dim, dim}, {dim, dim}}],
		Message[Reshuffle::argerr]; $Failed
	]
];
Reshuffle::argerr = "Reshuffle[A] works only for square matrices of size d^2\[Times]d^2 with integer d; for other shapes use Reshuffle[A, {{n1, n2}, {m1, m2}}].";

Reshuffle[a_, {n_, m_}] := Flatten[
	Table[Flatten[Part[a, 1 + i1 ;; n[[2]] + i1, 1 + i2 ;; m[[2]] + i2]], {i1, 0, n[[1]] n[[2]] - 1, n[[2]]}, {i2, 0, m[[1]]*m[[2]] - 1, m[[2]]}]
, 1];


(* ::Subsection::Closed:: *)
(*Partial trace and transposition*)


PartialTrace[a_, {}] := a;
PartialTrace[a_, _, {}] := a;

PartialTrace[a_?SquareMatrixQ, s : (_Integer | {__Integer})] := Module[{d = Sqrt[Length[a]], sys = Flatten[{s}]},
	Which[
		!IntegerQ[d], Message[PartialTrace::dimerr, Dimensions[a]]; $Failed,
		sys === {1}, PartialTrace1[a],
		sys === {2}, PartialTrace2[a],
		Sort[sys] === {1, 2}, Tr[a],
		True, Message[PartialTrace::syserr, s]; $Failed
	]
];

PartialTrace[a_?SquareMatrixQ, dim : {__Integer?Positive}, s : (_Integer | {__Integer})] := Module[{n = Length[dim], sys = Flatten[{s}]},
	Which[
		Times @@ dim =!= Length[a], Message[PartialTrace::dimmatch, dim, Dimensions[a]]; $Failed,
		!AllTrue[sys, 1 <= # <= n &], Message[PartialTrace::sysspecerr, n, Select[sys, !(1 <= # <= n) &]]; $Failed,
		n == 2 && Length[sys] == 1, If[First[sys] == 1, PartialTrace1[a, dim], PartialTrace2[a, dim]],
		Union[sys] === Range[n], Tr[a],
		True, PartialTraceGeneral[a, dim, Union[sys]]
	]
];

PartialTrace::syserr = "The second argument is expected to be 1, 2 or {1,2} (`1` was given).";
PartialTrace::dimerr = "PartialTrace[A, sys] expects a square matrix of size d^2\[Times]d^2 (a matrix of size `1` was given); use PartialTrace[A, {d1, d2, \[Ellipsis]}, sys] for other dimensions.";
PartialTrace::sysspecerr = "The system specification is invalid: a `1`-partite system has no subsystems `2`.";
PartialTrace::dimmatch = "The subsystem dimensions `1` are not compatible with a matrix of size `2`.";

PartialTrace1[x_] := With[{d = Sqrt[Length[x]]}, Total[x[[1 + d # ;; d + d #, 1 + d # ;; d + d #]] & /@ Range[0, d - 1]]];
PartialTrace2[x_] := With[{d = Sqrt[Length[x]]}, Total[x[[1 + # ;; d^2 ;; d, 1 + # ;; d^2 ;; d]] & /@ Range[0, d - 1]]];
PartialTrace1[x_, {d1_, d2_}] := Total[x[[1 + d2 # ;; d2 + d2 #, 1 + d2 # ;; d2 + d2 #]] & /@ Range[0, d1 - 1]];
PartialTrace2[x_, {d1_, d2_}] := Total[x[[1 + # ;; d1*d2 ;; d2, 1 + # ;; d1*d2 ;; d2]] & /@ Range[0, d2 - 1]];

ListReshape[list_, shape_] :=
	FlattenAt[Fold[Partition[#1, #2] &, Flatten[list], Reverse[shape]], 1];

PartialTraceGeneral[a_, dim_?VectorQ, sys_?VectorQ] := Module[
	{offset, keep, dispose, keepdim, disposedim, perm1, perm2, perm, tensor},
	offset = Length[dim];
	keep = Complement[Range[offset], sys];
	dispose = Union[sys];
	perm1 = Join[dispose, keep];
	perm2 = perm1 + offset;
	perm = Ordering[Join[perm1, perm2]];
	tensor = ListReshape[a, Join[dim, dim]];
	keepdim = Apply[Times, Join[dim, dim][[keep]]];
	disposedim = Apply[Times, Join[dim, dim][[dispose]]];
	tensor = Transpose[tensor, perm];
	tensor = ListReshape[tensor, {disposedim, keepdim, disposedim, keepdim}];
	Sum[tensor[[i, All, i, All]], {i, 1, disposedim}]
];

(* The row and column indices of every subsystem in sys are swapped. *)
PartialTranspose[\[Rho]_?SquareMatrixQ, dim : {__Integer?Positive}, s : (_Integer | {___Integer})] := Module[
	{n = Length[dim], sys = Union[Flatten[{s}]], perm},
	Which[
		Times @@ dim =!= Length[\[Rho]], Message[PartialTranspose::dimmatch, dim, Dimensions[\[Rho]]]; $Failed,
		!AllTrue[sys, 1 <= # <= n &], Message[PartialTranspose::sysspecerr, n, Select[sys, !(1 <= # <= n) &]]; $Failed,
		sys === {}, \[Rho],
		True,
			perm = Range[2 n];
			perm[[sys]] = sys + n;
			perm[[sys + n]] = sys;
			ListReshape[Transpose[ListReshape[\[Rho], Join[dim, dim]], perm], Dimensions[\[Rho]]]
	]
];
PartialTranspose::dimmatch = "The subsystem dimensions `1` are not compatible with a matrix of size `2`.";
PartialTranspose::sysspecerr = "The system specification is invalid: a `1`-partite system has no subsystems `2`.";


(* ::Subsection::Closed:: *)
(*Channels*)


Depo[\[Rho]_?SquareMatrixQ, pp_] := (1 - pp) \[Rho] + pp Tr[\[Rho]] IdentityMatrix[Length[\[Rho]]]/Length[\[Rho]];
AD[\[Rho]_?qubitMatrixQ, \[Gamma]_] := {{\[Rho][[1, 1]] + \[Gamma] \[Rho][[2, 2]], Sqrt[1 - \[Gamma]] \[Rho][[1, 2]]}, {Sqrt[1 - \[Gamma]] \[Rho][[2, 1]], -(-1 + \[Gamma]) \[Rho][[2, 2]]}};
Pauli[\[Rho]_?qubitMatrixQ, {ppx_, ppy_, ppz_}] := {{\[Rho][[1, 1]] - 1/4 (ppx + ppy) (\[Rho][[1, 1]] - \[Rho][[2, 2]]),
	(1 - ppz/2) \[Rho][[1, 2]] - 1/4 (ppx + ppy) \[Rho][[1, 2]] + 1/4 (ppx - ppy) \[Rho][[2, 1]]},
	{(1 - ppz/2) \[Rho][[2, 1]] - 1/4 (ppx + ppy) \[Rho][[2, 1]] + 1/4 (ppx - ppy) \[Rho][[1, 2]],
	\[Rho][[2, 2]] + 1/4 (ppx + ppy) (\[Rho][[1, 1]] - \[Rho][[2, 2]])}};


(* ::Subsection:: *)
(*Von Neumann, R\[EAcute]nyi, Relative entropy, Fidelity, Mutual Information*)


vonNeumann[\[Rho]_?SquareMatrixQ, base_ : 2] := Total[entropyTerm[#, base] & /@ Eigenvalues[\[Rho]]];


Renyi[\[Rho]_?SquareMatrixQ, \[Alpha]_, base_ : 2] := If[TrueQ[\[Alpha] == 1],
	vonNeumann[\[Rho], base],
	1/(1 - \[Alpha]) Log[base, Total[supportEigenvalues[\[Rho]]^\[Alpha]]]
];


(* S(\[Rho]||\[Sigma]) = -S(\[Rho]) - Tr[\[Rho] Log[\[Sigma]]], evaluated in an orthonormal eigenbasis {v} of \[Sigma] with weights w = <v|\[Rho]|v>. *)
KullbackLeibler[\[Rho]_?SquareMatrixQ, \[Sigma]_?SquareMatrixQ, base_ : 2] := Module[{vals, vecs, weights},
	{vals, vecs} = hermitianEigensystem[\[Sigma]];
	weights = (Conjugate[#] . \[Rho] . #) & /@ vecs;
	Chop[-vonNeumann[\[Rho], base] + Total[MapThread[crossEntropyTerm[#1, #2, base] &, {weights, vals}]]]
];
crossEntropyTerm[w_, \[Lambda]_, base_] := Which[
	numericZeroQ[w], 0,
	numericZeroQ[\[Lambda]], Infinity,
	True, -w Log[base, \[Lambda]]
];


QuantumFidelity[\[Rho]_?SquareMatrixQ, \[Sigma]_?SquareMatrixQ] := Chop[Tr[MatrixPower[MatrixPower[\[Rho], 1/2] . \[Sigma] . MatrixPower[\[Rho], 1/2], 1/2]]^2];


MutualInformation[\[Rho]_?SquareMatrixQ, dims : {_Integer, _Integer}] := Chop[
	vonNeumann[PartialTrace[\[Rho], dims, {2}]] + vonNeumann[PartialTrace[\[Rho], dims, {1}]] - vonNeumann[\[Rho]], 10^-12];
(* When subsystems are qubits *)
MutualInformation[\[Rho]_?SquareMatrixQ] := MutualInformation[\[Rho], {2, 2}];

TripartiteInformation[\[Rho]_?SquareMatrixQ, {dA_Integer, dB_Integer, dC_Integer}] := With[{dims = {dA, dB, dC}},
	MutualInformation[PartialTrace[\[Rho], dims, {3}], {dA, dB}] +
	MutualInformation[PartialTrace[\[Rho], dims, {2}], {dA, dC}] -
	MutualInformation[\[Rho], {dA, dB dC}]
];
TripartiteInformation[\[Rho]_?SquareMatrixQ] := TripartiteInformation[\[Rho], {2, 2, 2}];


(* information theory utilities *)
shannonEntropy[probs_List, base_ : 2] := Total[entropyTerm[#, base] & /@ probs];
(* Sum of each slice at fixed value k of the kept variable. In 0.0.3 Total[t, Complement[...]] was
   used, which for three or more variables also summed the kept level (Total[t, {1, 3}] sums
   levels 1 through 3). *)
marginalProbabilityDistribution[probs_List, sizes : {__Integer}, remainingDof_Integer] /; 1 <= remainingDof <= Length[sizes] := With[
	{probsTensor = ArrayReshape[probs, sizes]},
	Table[
		Total[Flatten[{probsTensor[[Sequence @@ ReplacePart[ConstantArray[All, Length[sizes]], remainingDof -> k]]]}]],
		{k, 1, sizes[[remainingDof]]}]
];
MutualInformationClassic[probs_List, sizes : {_Integer, _Integer} : {2, 2}] := Plus[
	shannonEntropy[marginalProbabilityDistribution[probs, sizes, 1]],
	shannonEntropy[marginalProbabilityDistribution[probs, sizes, 2]],
	- shannonEntropy @ probs
];
(* \[Sum]_rows P(row) H(row/P(row)), skipping rows of zero probability *)
weightedRowEntropy[mat_] := Total[If[PossibleZeroQ[Total[#]], 0, Total[#] shannonEntropy[#/Total[#]]] & /@ mat];
conditionalEntropyXcY[probs_List, sizes : {_Integer, _Integer} : {2, 2}] := weightedRowEntropy[Transpose[ArrayReshape[probs, sizes]]];
conditionalEntropyYcX[probs_List, sizes : {_Integer, _Integer} : {2, 2}] := weightedRowEntropy[ArrayReshape[probs, sizes]];


Purity[rho_?SquareMatrixQ] := Tr[rho . rho];


(* ::Subsection::Closed:: *)
(*Concurrence and entanglement of formation*)


Concurrence[\[Rho]_?twoQubitMatrixQ] := Module[{yy, \[Rho]t, \[Rho]sq, eigs},
	yy = KroneckerProduct[sy, sy];
	\[Rho]t = yy . Conjugate[\[Rho]] . yy;
	\[Rho]sq = MatrixPower[\[Rho], 1/2];
	(* numerical (not canonical) ordering: Sort puts e.g. Sqrt[3]/10 after 1/Sqrt[2] *)
	eigs = Reverse[SortBy[Chop[Eigenvalues[MatrixPower[\[Rho]sq . \[Rho]t . \[Rho]sq, 1/2]]], N[Re[#]] &]];
	Max[0, Re[eigs[[1]] - eigs[[2]] - eigs[[3]] - eigs[[4]]]]
];
Concurrence[psi_?VectorQ] /; Length[psi] == 4 := 2 Abs[psi[[1]] psi[[4]] - psi[[2]] psi[[3]]];

ConcurrenceX[\[Rho]_?twoQubitMatrixQ] := Module[{k1, k2},
	k1 = Abs[\[Rho][[2, 3]]] - Sqrt[\[Rho][[1, 1]] \[Rho][[4, 4]]];
	k2 = Abs[\[Rho][[1, 4]]] - Sqrt[\[Rho][[2, 2]] \[Rho][[3, 3]]];
	2*Max[0, k1, k2]
];

EntanglementOfFormation[\[Rho]_?twoQubitStateQ, b_ : 2] := Module[{x},
	x = (1 + Sqrt[1 - Concurrence[\[Rho]]^2])/2;
	entropyTerm[x, b] + entropyTerm[1 - x, b]
];

(* (|\[Lambda]| - \[Lambda])/2 is |\[Lambda]| for negative \[Lambda] and 0 otherwise, and also works for symbolic eigenvalues. *)
Negativity[\[Rho]_?SquareMatrixQ, dims : {_Integer, _Integer}] :=
	Chop[Total[(Abs[#] - #)/2 & /@ Eigenvalues[PartialTranspose[\[Rho], dims, {1}]]]];
Negativity[\[Rho]_?SquareMatrixQ] := Negativity[\[Rho], {2, 2}];


(* ::Subsection::Closed:: *)
(*Discord*)


(* Projectors on the eigenvectors of the Bloch direction (\[Theta], \[Phi]). *)
blochProjectors[\[Theta]_, \[Phi]_] := Module[{v},
	v = {{Cos[\[Theta]/2], E^(-I \[Phi]) Sin[\[Theta]/2]}, {E^(I \[Phi]) Sin[\[Theta]/2], -Cos[\[Theta]/2]}};
	{(v . m) \[CircleDot] (m . v), (v . p) \[CircleDot] (p . v)}
];

(* \[Sum]k pk S(\[Rho]|k) for the unmeasured qubit after measuring qubit `measured` with the projectors;
   outcomes of zero probability contribute 0. *)
averageConditionalEntropy[\[Rho]_, projectors_List, measured_Integer] := N @ Total[
	Function[proj, Module[{b, post, pk},
		b = If[measured == 1, proj \[CircleTimes] id, id \[CircleTimes] proj];
		post = b . \[Rho] . b;
		pk = Tr[post];
		If[numericZeroQ[pk], 0, pk vonNeumann[PartialTrace[post, {2, 2}, {measured}]/pk]]
	]] /@ projectors];

ClAB[\[Rho]\[Rho]_?twoQubitMatrixQ][\[Theta]_, \[Phi]_] := averageConditionalEntropy[\[Rho]\[Rho], blochProjectors[\[Theta], \[Phi]], 1];
ClBA[\[Rho]\[Rho]_?twoQubitMatrixQ][\[Theta]_, \[Phi]_] := averageConditionalEntropy[\[Rho]\[Rho], blochProjectors[\[Theta], \[Phi]], 2];

(* {\[Theta], \[Phi], f[\[Theta], \[Phi]]} with the smallest f on a regular grid. *)
gridMinimum[f_, {\[Theta]A_, \[Theta]B_, d\[Theta]_}, {\[Phi]A_, \[Phi]B_, d\[Phi]_}] := Sort[
	Flatten[Table[{\[Theta]\[Theta], \[Phi]\[Phi], Chop[N[f[\[Theta]\[Theta], \[Phi]\[Phi]]]]}, {\[Theta]\[Theta], \[Theta]A, \[Theta]B, d\[Theta]}, {\[Phi]\[Phi], \[Phi]A, \[Phi]B, d\[Phi]}], 1],
	#1[[3]] < #2[[3]] &][[1]];

(* Minimum of f[\[Theta], \[Phi]] over the sphere: coarse grid with step \[Pi]/20, then a 40 times finer grid around the best point. *)
minimizeOverBlochSphere[f_] := Module[{d\[Theta] = \[Pi]/20, d\[Phi] = \[Pi]/20, best},
	best = gridMinimum[f, {0, \[Pi], d\[Theta]}, {0, 2 \[Pi], d\[Phi]}];
	best = gridMinimum[f,
		{best[[1]] - d\[Theta]/2, best[[1]] + d\[Theta]/2, d\[Theta]/40},
		{best[[2]] - d\[Phi]/2, best[[2]] + d\[Phi]/2, d\[Phi]/40}];
	best[[3]]
];

DiscZAB[\[Rho]\[Rho]_?twoQubitMatrixQ] := With[{\[Rho]n = N[\[Rho]\[Rho]]},
	vonNeumann[PartialTrace[\[Rho]n, {2, 2}, {2}]] - vonNeumann[\[Rho]n] + minimizeOverBlochSphere[ClAB[\[Rho]n]]];

DiscZBA[\[Rho]\[Rho]_?twoQubitMatrixQ] := With[{\[Rho]n = N[\[Rho]\[Rho]]},
	vonNeumann[PartialTrace[\[Rho]n, {2, 2}, {1}]] - vonNeumann[\[Rho]n] + minimizeOverBlochSphere[ClBA[\[Rho]n]]];

(* Local Bloch vectors {a, b} and correlation matrix t[[i, j]] = Tr[\[Rho] \[Sigma]i\[CircleTimes]\[Sigma]j]. *)
twoQubitCorrelations[\[Rho]_] := With[{\[Sigma] = {sx, sy, sz}}, {
	Table[Tr[\[Rho] . (\[Sigma][[j]] \[CircleTimes] id)], {j, 1, 3}],
	Table[Tr[\[Rho] . (id \[CircleTimes] \[Sigma][[j]])], {j, 1, 3}],
	Chop[Table[Tr[\[Rho] . (\[Sigma][[i]] \[CircleTimes] \[Sigma][[j]])], {i, 1, 3}, {j, 1, 3}]]}];

(* Normalized geometric discord for measurements on the party with Bloch vector b, with t
   arranged so that t^T.t is that party's correlation Gram matrix. *)
geometricDiscord[b_, t_] := Module[{k},
	k = Max[Chop[Eigenvalues[Chop[Outer[Times, b, b] + Transpose[t] . t]]]];
	1/2 (Norm[b]^2 + Tr[Transpose[t] . t] - k)
];

GeomDiscordAB[\[Rho]_?twoQubitMatrixQ] := With[{c = twoQubitCorrelations[\[Rho]]}, geometricDiscord[c[[2]], c[[3]]]];
GeomDiscordBA[\[Rho]_?twoQubitMatrixQ] := With[{c = twoQubitCorrelations[\[Rho]]}, geometricDiscord[c[[1]], Transpose[c[[3]]]]];


(* ::Subsection::Closed:: *)
(*Lindblad ME*)


\[ScriptCapitalD][OP_][\[Rho]_] := OP . \[Rho] . ConjugateTranspose[OP] - 1/2 ConjugateTranspose[OP] . OP . \[Rho] - 1/2 \[Rho] . ConjugateTranspose[OP] . OP
\[ScriptCapitalD]c[OP_][\[Rho]_] := ConjugateTranspose[OP] . \[Rho] . OP - 1/2 ConjugateTranspose[OP] . OP . \[Rho] - 1/2 \[Rho] . ConjugateTranspose[OP] . OP


ME[HH_, OPS__][\[Rho]_] := -I Comm[HH, \[Rho]] + Total[\[ScriptCapitalD][#][\[Rho]] & /@ toOperatorList[{OPS}]]


MESolve[\[ScriptCapitalL]_, \[Rho]0_?SquareMatrixQ] := MESolve[\[ScriptCapitalL], \[Rho]0, Global`\[Tau]];
MESolve[\[ScriptCapitalL]_, \[Rho]0_?SquareMatrixQ, t_Symbol] := Module[{d = Length[\[Rho]0], r, rr, eqs, sol},
	rr = Table[r[k, j][t], {k, 1, d}, {j, 1, d}];
	eqs = Join[
		Thread[Flatten[D[rr, t]] == Flatten[Normal[\[ScriptCapitalL][rr]]]],
		Thread[Flatten[rr /. t -> 0] == Flatten[Normal[\[Rho]0]]]];
	sol = DSolve[eqs, Flatten[rr], t];
	If[MatchQ[sol, {{__Rule}, ___}],
		rr /. First[sol],
		Message[MESolve::nosol]; $Failed
	]
];
MESolve[\[ScriptCapitalL]_, \[Rho]0_?SquareMatrixQ, t_] := (Message[MESolve::tvar, t]; $Failed);
MESolve::tvar = "The time variable `1` is not a symbol without value.";
MESolve::nosol = "DSolve could not solve the master equation.";


MESolvet[\[ScriptCapitalL]_, \[Rho]0_?SquareMatrixQ, ti_?NumericQ, tf_?NumericQ] := Module[{\[Rho], t, sol},
	sol = NDSolve[{\[Rho]'[t] == \[ScriptCapitalL][\[Rho][t]], \[Rho][ti] == Normal[\[Rho]0]}, \[Rho], {t, ti, tf}];
	If[MatchQ[sol, {{__Rule}, ___}], \[Rho] /. First[sol], $Failed]
];


(* ::Subsection::Closed:: *)
(*Cascade ME*)


\[ScriptCapitalD]cascAB[OPA_, OPB_][\[Rho]_] := Comm[OPA . \[Rho], ConjugateTranspose[OPB]] + Comm[OPB, \[Rho] . ConjugateTranspose[OPA]] + \[ScriptCapitalD][OPB][\[Rho]] + \[ScriptCapitalD][OPA][\[Rho]]


(* ::Subsection:: *)
(*Vectorization *)


Vec2[x_?SquareMatrixQ] := Tr[x . #] & /@ GEN[Length[x]];
UnVec2[x_List] := With[{d = Sqrt[Length[x]]},
	If[IntegerQ[d], x . GEN[d], Message[UnVec2::argerr, Length[x]]; $Failed]];
UnVec2::argerr = "UnVec2 needs a vector of length d^2 for an integer d (length `1` was given).";


(* Column stacking: Vec[A.X.B] == (B^T \[CircleTimes] A).Vec[X], hence
   Vec[L\[Rho]L\[Dagger] - 1/2{L\[Dagger]L, \[Rho]}] = (L\[Conjugate]\[CircleTimes]L - 1/2 (L\[Dagger]L)^T\[CircleTimes]\[DoubleStruckCapitalI] - 1/2 \[DoubleStruckCapitalI]\[CircleTimes]L\[Dagger]L) Vec[\[Rho]] *)
\[ScriptCapitalD]ToVec[L_] := With[{\[ScriptCapitalI] = Id[Length@L]},
	Conjugate[L] \[CircleTimes] L - 1/2 Transpose[ConjugateTranspose[L] . L] \[CircleTimes] \[ScriptCapitalI] - 1/2 \[ScriptCapitalI] \[CircleTimes] (ConjugateTranspose[L] . L)]
(* [A\[Rho], B\[Dagger]] = A\[Rho]B\[Dagger] - B\[Dagger]A\[Rho]  \[RightArrow]  B\[Conjugate]\[CircleTimes]A - \[DoubleStruckCapitalI]\[CircleTimes](B\[Dagger]A)
   [B, \[Rho]A\[Dagger]] = B\[Rho]A\[Dagger] - \[Rho]A\[Dagger]B  \[RightArrow]  A\[Conjugate]\[CircleTimes]B - (A\[Dagger]B)^T\[CircleTimes]\[DoubleStruckCapitalI] *)
\[ScriptCapitalD]cascABToVec[OPA_, OPB_] := With[{\[ScriptCapitalI] = Id[Length@OPA]},
	Conjugate[OPB] \[CircleTimes] OPA - \[ScriptCapitalI] \[CircleTimes] (ConjugateTranspose[OPB] . OPA) +
	Conjugate[OPA] \[CircleTimes] OPB - Transpose[ConjugateTranspose[OPA] . OPB] \[CircleTimes] \[ScriptCapitalI]] +
	\[ScriptCapitalD]ToVec[OPB] + \[ScriptCapitalD]ToVec[OPA]
(* -i[H,\[Rho]] = -i(\[DoubleStruckCapitalI] \[CircleTimes] H - H^T \[CircleTimes] \[DoubleStruckCapitalI])Vec[\[Rho]] *)
HToVec[H_] := With[{\[ScriptCapitalI] = Id[Length@H]}, -I (\[ScriptCapitalI] \[CircleTimes] H - Transpose[H] \[CircleTimes] \[ScriptCapitalI])]

METoVec[HH_, OPS__] := HToVec[HH] + Total[\[ScriptCapitalD]ToVec /@ toOperatorList[{OPS}]];


(* Uses Vec (column stacking), the convention of METoVec; in 0.0.3 the Gell-Mann
   coefficients Vec2 were used here by mistake. *)
METoVecSolve[HH_?SquareMatrixQ, OPS__, \[Rho]0_?SquareMatrixQ] := METoVecSolve[HH, OPS, \[Rho]0, Global`\[Tau]];
METoVecSolve[HH_?SquareMatrixQ, OPS__, \[Rho]0_?SquareMatrixQ, t_Symbol] :=
	Simplify[MatrixExp[METoVec[HH, OPS] t, Vec[Normal[\[Rho]0]]]];


(* ::Subsection:: *)
(*Representations*)


Lambda1[i_, j_, n_] := Table[KroneckerDelta[j, \[Mu]] KroneckerDelta[i, \[Nu]] + KroneckerDelta[j, \[Nu]] KroneckerDelta[i, \[Mu]], {\[Mu], 1, n}, {\[Nu], 1, n}];

Lambda2[i_, j_, n_] := Table[-I (KroneckerDelta[i, \[Mu]] KroneckerDelta[j, \[Nu]] - KroneckerDelta[i, \[Nu]] KroneckerDelta[j, \[Mu]]), {\[Mu], 1, n}, {\[Nu], 1, n}];

Lambda3[i_, n_] := Sqrt[2/(i^2 - i)] DiagonalMatrix[Join[Append[Table[1, {i - 1}], -(i - 1)], Table[0, {n - i}]]];

GeneralizedPauliMatrices[n_] := Module[{l1, l2, l3},
	l1 = Flatten[Table[Lambda1[i, j, n], {i, 1, n}, {j, i + 1, n}], 1];
	l2 = Flatten[Table[Lambda2[i, j, n], {i, 1, n}, {j, i + 1, n}], 1];
	l3 = Table[Lambda3[i, n], {i, 2, n}];
	Join[l1, l2, l3]
];
GEN[dim_] := Join[{Id[dim]/Sqrt[dim]}, 1/Sqrt[2] GeneralizedPauliMatrices[dim]];
TAU[dim_] := Flatten[Table[\[Alpha]1 \[CircleDot] \[Alpha]2, {\[Alpha]1, Id[dim]}, {\[Alpha]2, Id[dim]}], 1];


(* GEN takes the dimension (not the number of qubits) since 0.0.3. *)
StateToBloch[a_?VectorQ] := StateToBloch[Todm[a]];
StateToBloch[a_?SquareMatrixQ] := With[{d = Length[a]}, Sqrt[d] (Tr[a . #] & /@ GEN[d])];

BlochToState[vec_List] := Module[{len = Length[vec], d, v = vec},
	Which[
		IntegerQ[Sqrt[len]] && len > 0, d = Sqrt[len],
		IntegerQ[Sqrt[len + 1]] && len > 0, d = Sqrt[len + 1]; v = Prepend[vec, 1],
		True, Message[BlochToState::argerr, vec]; Return[$Failed]
	];
	v . GEN[d]/Sqrt[d]
];
BlochToState::argerr = "The vector `1` is not a Bloch vector of any dimension (its length must be d^2 or d^2-1).";


(* ::Subsection::Closed:: *)
(*Kraus Operators*)


RandomKrausOPS[n_, nOps_] := Module[{gin, h},
	gin = GinibreMatrix[2^n, 2^n] & /@ Range[nOps];
	h = Total[ConjugateTranspose[#] . # & /@ gin];
	# . MatrixPower[h, -1/2] & /@ gin
];

KrausTo\[Phi][Ak_, \[Rho]_] := Total[# . \[Rho] . ConjugateTranspose[#] & /@ toOperatorList[{Ak}]];


(* Process-matrix functions. In 0.0.3 they relied on the undocumented global variables
   GG1 (operator basis) and \[Tau] (matrix units); the basis is now GEN[d], with d inferred from the
   arguments, or the value of the option "Basis". *)
Options[KrausToF] = Options[FTo\[Phi]] = Options[\[Phi]ToF] = Options[FToS] = Options[SToF] = Options[FToKraus] = {"Basis" -> Automatic};

Scan[(MessageName[#, "basis"] = "The value `1` of the option \"Basis\" is not a list of `2`^2 matrices of size `2`\[Times]`2`.") &,
	{KrausToF, FTo\[Phi], \[Phi]ToF, FToS, SToF, FToKraus}];
Scan[(MessageName[#, "dim"] = "The matrix must be of size d^2\[Times]d^2 for an integer d (a matrix of size `1` was given).") &,
	{FToS, SToF, SToKraus}];
\[Phi]ToF::nodim = "The dimension of the input space cannot be inferred: use \[Phi]ToF[\[CapitalPhi], d] or give the option \"Basis\".";
\[Phi]ToKraus::nodim = "The dimension of the input space cannot be inferred: use \[Phi]ToKraus[\[CapitalPhi], d].";
SToKraus::notcp = "The Choi matrix has negative eigenvalues, so the map is not completely positive; the absolute values of the eigenvalues are used.";

operatorBasis[head_, d_, basis_] := Which[
	basis === Automatic, GEN[d],
	ListQ[basis] && Dimensions[basis] === {d^2, d, d}, basis,
	True, Message[MessageName[head, "basis"], basis, d]; $Failed
];

(* F[[\[Alpha], \[Beta]]] = Tr[G\[Alpha]\[Dagger].\[CapitalPhi][G\[Beta]]] *)
processMatrix[\[CapitalPhi]_, basis_] := Outer[Tr[ConjugateTranspose[#1] . \[CapitalPhi][#2]] &, basis, basis, 1];
(* S = \[Sum]ij \[CapitalPhi][|i><j|] \[CircleTimes] |i><j| *)
choiMatrix[\[CapitalPhi]_, d_] := Total[(\[CapitalPhi][#] \[CircleTimes] #) & /@ TAU[d]];
(* \[CapitalPhi][X] = Tr_2[S.(\[DoubleStruckCapitalI] \[CircleTimes] X^T)] *)
choiAction[s_, x_, d_] := PartialTrace2[s . KroneckerProduct[IdentityMatrix[d], Transpose[x]], {d, d}];

KrausToF[Ak_, opts : OptionsPattern[]] := Module[{ops = toOperatorList[{Ak}], basis},
	basis = operatorBasis[KrausToF, Length[First[ops]], OptionValue["Basis"]];
	If[basis === $Failed, $Failed, processMatrix[KrausTo\[Phi][ops, #] &, basis]]
];

FTo\[Phi][F_?SquareMatrixQ, \[Rho]_?SquareMatrixQ, opts : OptionsPattern[]] := Module[{basis},
	basis = operatorBasis[FTo\[Phi], Length[\[Rho]], OptionValue["Basis"]];
	If[basis === $Failed, $Failed,
		Total[(F . (Tr[ConjugateTranspose[#] . \[Rho]] & /@ basis)) basis]]
];

\[Phi]ToF[\[CapitalPhi]_, d_Integer?Positive, opts : OptionsPattern[]] := Module[{basis},
	basis = operatorBasis[\[Phi]ToF, d, OptionValue["Basis"]];
	If[basis === $Failed, $Failed, processMatrix[\[CapitalPhi], basis]]
];
\[Phi]ToF[\[CapitalPhi]_, opts : OptionsPattern[]] := With[{basis = OptionValue["Basis"]},
	If[ListQ[basis] && basis =!= {} && SquareMatrixQ[First[basis]],
		\[Phi]ToF[\[CapitalPhi], Length[First[basis]], opts],
		Message[\[Phi]ToF::nodim]; $Failed]
];

FToS[F_?SquareMatrixQ, opts : OptionsPattern[]] := With[{d = Sqrt[Length[F]]},
	If[IntegerQ[d],
		choiMatrix[FTo\[Phi][F, #, opts] &, d],
		Message[FToS::dim, Dimensions[F]]; $Failed]
];

SToF[S_?SquareMatrixQ, opts : OptionsPattern[]] := Module[{d = Sqrt[Length[S]], basis},
	If[!IntegerQ[d], Message[SToF::dim, Dimensions[S]]; Return[$Failed]];
	basis = operatorBasis[SToF, d, OptionValue["Basis"]];
	If[basis === $Failed, $Failed, processMatrix[choiAction[S, #, d] &, basis]]
];

KrausToS[Ak_] := With[{ops = toOperatorList[{Ak}]}, choiMatrix[KrausTo\[Phi][ops, #] &, Length[First[ops]]]];

(* Each eigenvector v of S, reshaped row-wise, is a Kraus operator up to the factor Sqrt[\[Lambda]];
   eigenvectors with zero eigenvalue are dropped. *)
SToKraus[S_?SquareMatrixQ] := Module[{d = Sqrt[Length[S]], vals, vecs},
	If[!IntegerQ[d], Message[SToKraus::dim, Dimensions[S]]; Return[$Failed]];
	{vals, vecs} = hermitianEigensystem[S];
	If[AnyTrue[vals, NumericQ[#] && TrueQ[Re[N[#]] < -$zeroTolerance] &], Message[SToKraus::notcp]];
	MapThread[Sqrt[Abs[#1]] Partition[#2, d] &,
		{Pick[vals, numericZeroQ /@ vals, False], Pick[vecs, numericZeroQ /@ vals, False]}]
];

FToKraus[F_?SquareMatrixQ, opts : OptionsPattern[]] := With[{s = FToS[F, opts]},
	If[s === $Failed, $Failed, SToKraus[s]]];

\[Phi]ToKraus[\[CapitalPhi]_, d_Integer?Positive] := SToKraus[choiMatrix[\[CapitalPhi], d]];
\[Phi]ToKraus[\[CapitalPhi]_] := (Message[\[Phi]ToKraus::nodim]; $Failed);


(* ::Subsection::Closed:: *)
(*Einstein Sum*)


einsum[in_List -> out_, arrays__] := Module[{res = isum[in -> out, {arrays}]},
	res /; res =!= $Failed
]

isum[in_List -> out_, arrays_List] := Catch@Module[
	{indices, contracted, uncontracted, contractions, transpose},

	If[Length[in] != Length[arrays],
		Message[einsum::length, Length[in], Length[arrays]];
		Throw[$Failed]
	];

	(* the rank check is skipped for arrays of unknown rank *)
	MapThread[
		If[IntegerQ@TensorRank[#2] && Length[#1] != TensorRank[#2],
			Message[einsum::shape, #1, #2];
			Throw[$Failed]
		] &,
		{in, arrays}
	];

	indices = Tally[Flatten[in, 1]];

	If[DeleteCases[indices, {_, 1 | 2}] =!= {},
		Message[einsum::repeat, Cases[indices, {x_, Except[1 | 2]} :> x]];
		Throw[$Failed]
	];

	uncontracted = Cases[indices, {x_, 1} :> x];

	If[Sort[uncontracted] =!= Sort[out],
		Message[einsum::output, uncontracted, out];
		Throw[$Failed]
	];

	contracted = Cases[indices, {x_, 2} :> x];
	contractions = Flatten[Position[Flatten[in, 1], #]] & /@ contracted;
	transpose = FindPermutation[uncontracted, out];
	Activate @ TensorTranspose[
		TensorContract[
			Inactive[TensorProduct] @@ arrays,
			contractions
		],
		transpose
	]
]

einsum::length = "Number of index specifications (`1`) does not match the number of arrays (`2`).";
einsum::shape = "Index specification `1` does not match the array depth of `2`.";
einsum::repeat = "Index specifications `1` are repeated more than twice.";
einsum::output = "The uncontracted indices `1` do not match the desired output `2`.";


(* ::Subsection:: *)
(*Bloch Sphere*)


bv[r_] := {(r[[1, 2]] + r[[2, 1]]), I r[[1, 2]] - I r[[2, 1]], (r[[1, 1]] - r[[2, 2]])}

(* Real numerical Bloch coordinates of a ket or density matrix, for graphics. *)
blochCoordinates[state_?VectorQ] := blochCoordinates[state \[CircleDot] Conjugate[state]];
blochCoordinates[state_?MatrixQ] := Re[Chop[N[bv[state]]]];

point[state_?(VectorQ[#] || MatrixQ[#] &), Col_] := Graphics3D[{Col, Point[blochCoordinates[state]]}]
arrow[state_?(VectorQ[#] || MatrixQ[#] &), Col_] := With[{r = blochCoordinates[state]},
	Graphics3D[{Col, Thickness@0.004, Point[{0, 0, 0}], Point[r], Arrow[{{0, 0, 0}, r}]}]]
line[points_, Col_] := Graphics3D[{Col, Thickness@0.003, Sequence @@ {Sequence @@ Point /@ #, Line@#} &@points}]
sfera = {SphericalPlot3D[1, {\[Theta], 0, Pi}, {\[Phi], 0, 2 Pi}, PlotStyle -> None, Mesh -> 7, MeshStyle -> Gray, ImageSize -> 300, AxesLabel -> {"x", "y", "z"}],
	Graphics3D[{White, Opacity@0.3, Sphere[{0, 0, 0}, 1], Opacity@1, Thickness@0.004, PointSize@0.02}]};
assi = {line[{{0, 0, 1}, {0, 0, -1}}, Black], line[{{1, 0, 0}, {-1, 0, 0}}, Black], line[{{0, 1, 0}, {0, -1, 0}}, Black]};


(* ::Section:: *)
(*END*)


End[];


EndPackage[];
