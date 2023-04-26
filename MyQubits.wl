(* ::Package:: *)

BeginPackage["MyQubits`"];


(*$PrePrint = If[SquareMatrixQ[#], MatrixForm[#], #]&;*)


(* ::Section:: *)
(*HELPS*)


(* ::Subsection:: *)
(*Aliases*)


mf::usage = "";
cf::usage = "";
Dim::usage = "";
Len::usage = "";
Conj::usage = "";


(* ::Subsection::Closed:: *)
(*Commutator and AntiCommutators*)


Comm::usage = "";
AComm::usage = "";


(* ::Subsection:: *)
(*Symbolic Matrix and Vector*)


SymMatrix::usage = "SymMatrix[symbol,dim1,dim2] if square SymMatrix[symbol,dim1]";
SymVector::usage = "SymMatrix[symbol,dim]";


(* ::Subsection:: *)
(*KroneckerProduct, BraKet, DirectSum*)


CircleTimes::usage = "";
CircleDot::usage = "";
CirclePlus::usage = "";
Todm::usage = "";


(* ::Subsection:: *)
(*Commonly used matrices*)


m::usage = "";
p::usage = "";
sx::usage = "";
sy::usage = "";
sz::usage = "";
sm::usage = "";
sp::usage = "";
id::usage = "";
Id::usage = "";
OO::usage = "";
Psip::usage = "";
Psim::usage = "";
Phip::usage = "";
Phim::usage = "";
W3::usage = "";
GHZ3::usage = "";
Trid::usage = "";
Tridopen::usage = "";
Tridclosed::usage = "";


(* ::Subsection:: *)
(*Distance measures*)


MatrixSqrt::usage = "";

MatrixAbs::usage = "";

Fidelity::usage = "";

TraceNorm::usage = "";

TraceDistance::usage = "";



(* ::Subsection::Closed:: *)
(*Qubits Systems*)


NQubits::usage = "";
Sx::usage = "";
Sy::usage = "";
Sz::usage = "";
Sp::usage = "";
Sm::usage = "";
TightBinding::usage = "";


(* ::Subsection::Closed:: *)
(*Random states*)


RandomSimplex::usage = "";
RandomKet::usage = "";
GinibreMatrix::usage = "";
RandomSpecialUnitary::usage = "";
RandomUnitary::usage = "";
RandomOrthogonal::usage = "";
RandomState::usage = "";
RandomPOVM::usage = "";


(* ::Subsection::Closed:: *)
(*Reshaping, vectorization and reshuffling*)


Vec::usage = "";
Unvec::usage = "";
Res::usage = "";
Unres::usage = "";
Reshuffle::usage = "";


(* ::Subsection::Closed:: *)
(*Partial trace and transposition*)


PartialTrace::usage = "";
PartialTrace1::usage = "";
PartialTrace2::usage = "";
ListReshape::usage = "";
PartialTraceGeneral::usage = "";
PartialTranspose::usage = "";


(* ::Subsection::Closed:: *)
(*Channels*)


Depo::usage = "";
AD::usage = "";
Pauli::usage = "";


(* ::Subsection:: *)
(*Von Neumann, R\[EAcute]nyi, Relative entropy, Fidelity, Mutual Information*)


vonNeumann::usage = "";
Renyi::usage = "";
KullbackLeibler::usage = "";
QuantumFidelity::usage = "";
MutualInformation::usage = "";
TripartiteInformation::usage = "";
(* information theory utilities *)
shannonEntropy::usage = "";
marginalProbabilityDistribution::usage = "";
MutualInformationClassic::usage = "";
conditionalEntropyXcY::usage = "";
conditionalEntropyYcX::usage = "";
Purity::usage = "";


(* ::Subsection::Closed:: *)
(*Concurrence and entanglement of formation*)


Concurrence::usage = "";
ConcurrenceX::usage = "";
EntanglementOfFormation::usage = "";
Negativity::usage = "";


(* ::Subsection::Closed:: *)
(*Discord*)


ClAB::usage = "";
DiscZAB::usage = "";
ClBA::usage = "";
DiscZBA::usage = "";
GeomDiscordAB::usage = "";
GeomDiscordBA::usage = "";


(* ::Subsection::Closed:: *)
(*Lindblad ME*)


\[ScriptCapitalD]::usage = "";
\[ScriptCapitalD]c::usage = "";
ME::usage = "";
MESolvet::usage = "";
MESolve::usage = "";


(* ::Subsection::Closed:: *)
(*Cascade ME*)


\[ScriptCapitalD]cascAB::usage = "";


(* ::Subsection::Closed:: *)
(*Vectorization *)


Vec2::usage = "";
UnVec2::usage = "";


\[ScriptCapitalD]ToVec::usage = "";
\[ScriptCapitalD]cascABToVec::usage = "";
HToVec::usage = "";
METoVec::usage = "";


METoVecSolve::usage = "";


(* ::Subsection::Closed:: *)
(*Representation*)


Lambda1::usage = "";
Lambda2::usage = "";
Lambda3::usage = "";
GeneralizedPauliMatrices::usage = "";
GEN::usage = "";
TAU::usage = "";


StateToBloch::usage = "";
BlochToState::usage = "";


(* ::Subsection::Closed:: *)
(*Kraus Operators*)


RandomKrausOPS::usage = "";
KrausTo\[Phi]::usage = "";
KrausToF::usage = "";
FTo\[Phi]::usage = "";
\[Phi]ToF::usage = "";
FToS::usage = "";
SToF::usage = "";
FToKraus::usage = "";
\[Phi]ToKraus::usage = "";


(* ::Subsection::Closed:: *)
(*Einstein Sum*)


einsum::usage = "";
isum::usage = "";

einsum::length = "Number of index specifications (`1`) does not match the number of arrays (`2`)";
einsum::shape = "Index specification `1` does not match the array depth of `2`";
einsum::repeat = "Index specifications `1` are repeated more than twice";
einsum::output = "The uncontracted indices don't match the desired output";


(* ::Subsection::Closed:: *)
(*Bloch Sphere*)


bv::usage = "";
point::usage = "";
point::usage = "";
arrow::usage = "";
arrow::usage = "";
line::usage = "";
sfera::usage = "";
assi::usage = "";


(* ::Section:: *)
(*PRIVATE DEFINITIONS*)


Begin["`Private`"];


(* ::Subsection:: *)
(*Aliases*)


mf[A_]:=(Print[MatrixForm@A];A)
cf[expr_]:=FullSimplify@ComplexExpand@expr;
Dim[in_]:=Dimensions[in]
Len[in_]:=Length[in]
Conj[z_]:=z/.Complex[a_,b_] -> Complex[a,-b]


(* ::Subsection::Closed:: *)
(*Commutator and AntiCommutators*)


Comm[A_,B_]:=A . B-B . A;
AComm[A_,B_]:=A . B+B . A;


(* ::Subsection::Closed:: *)
(*Symbolic Matrix and Vector*)


SymMatrix[sym_,d1_?IntegerQ]:=SymMatrix[sym,d1,d1];
SymMatrix[sym_,d1_?IntegerQ,d2_?IntegerQ] := Table[Subscript[sym, i,j], {i,1,d1}, {j,1,d2}];
SymVector[sym_,d1_]:= SymMatrix[sym,d1,1];


(* ::Subsection::Closed:: *)
(*KroneckerProduct, BraKet, DirectSum*)


CircleTimes[x_?MatrixQ,y_?MatrixQ] := KroneckerProduct[x,y];
CircleTimes[x_?VectorQ,y_?VectorQ] := Flatten[KroneckerProduct[x,y]];
CircleTimes[M_?MatrixQ] := M;
CircleTimes[l_] := CircleTimes@@l;
CircleTimes[a_,b__] := CircleTimes[a, CircleTimes[b]];


CircleDot[x_?VectorQ,y_?VectorQ]:= Outer[Times,x,y];


CirclePlus[x_?MatrixQ,y_?MatrixQ] := ArrayFlatten[{{x,0},{0,y}}];
CirclePlus[x_?MatrixQ,{}] := x;
CirclePlus[{},y_?MatrixQ] := y;


CirclePlus[l_] := CirclePlus@@l;
CirclePlus[a_,b__] := CirclePlus[a, CirclePlus[b]];


Todm[v_]:=v\[CircleDot]Conj[v];


(* ::Subsection::Closed:: *)
(*Commonly used matrices*)


m={0,1};p={1,0};
sx = {{0,1},{1,0}};
sy = {{0,-I},{I,0}};
sz = {{1,0},{0,-1}};
id = {{1,0},{0,1}};
sp =(sx + I sy)/2;sm = (sx - I sy)/2;
wh = {{1,1},{1,-1}};
cnot = {{1,0,0,0},{0,1,0,0},{0,0,0,1},{0,0,1,0}};
Id[n_]:=IdentityMatrix[n];
OO[n_]:=SparseArray[{},  {n, n}];
OO[1]={};

Phip=1/Sqrt[2] (m\[CircleTimes]m+p\[CircleTimes]p)
Phim=1/Sqrt[2] (m\[CircleTimes]m-p\[CircleTimes]p)
Psip=1/Sqrt[2] (m\[CircleTimes]p+p\[CircleTimes]m)
Psim=1/Sqrt[2] (m\[CircleTimes]p-p\[CircleTimes]m)

W3=1/Sqrt[3] ((m\[CircleTimes]m\[CircleTimes]p+m\[CircleTimes]p\[CircleTimes]m+p\[CircleTimes]m\[CircleTimes]m))
GHZ3=1/Sqrt[2] (m\[CircleTimes]m\[CircleTimes]m+p\[CircleTimes]p\[CircleTimes]p)


Trid[n_,Bandup_?VectorQ,Bandc_?VectorQ,Banddw_?VectorQ]:=SparseArray[{
			Band[{1,2}]->Bandup[[1;;n-1]],
			Band[{1,1}]->Bandc,
			Band[{2,1}]->Banddw[[1;;n-1]]},n]
Trid[n_,a_,b_,c_]:=SparseArray[{
			Band[{1,2}]->(a&/@Range[1,n-1]),
			Band[{1,1}]->(b&/@Range[1,n]),
			Band[{2,1}]->(c&/@Range[1,n-1])},n]
Trid[ndimer_,Mup_?MatrixQ,Mc_?MatrixQ,Mdw_?MatrixQ]:=
			SparseArray[{
			Band[{1,3}]->(Mup&/@Range[1,ndimer-1]),
			Band[{1,1}]->(Mc&/@Range[1,ndimer]),
			Band[{3,1}]->(Mdw&/@Range[1,ndimer-1])},2*ndimer]
Tridopen[n_,Bandup_?VectorQ,Bandc_?VectorQ,Banddw_?VectorQ]:=
			Trid[n,Bandup,Bandc,Banddw]
Tridopen[n_,a_,b_,c_]:=
			Trid[n,a,b,c]
Tridopen[ndimer_,Mup_?MatrixQ,Mc_?MatrixQ,Mdw_?MatrixQ]:=
			Trid[ndimer,Mup,Mc,Mdw]
Tridclosed[n_,Bandup_?VectorQ,Bandc_?VectorQ,Banddw_?VectorQ]:=SparseArray[{
			{1,n}->Banddw[[-1]],{n,1}->Bandup[[-1]],
			Band[{1,2}]->Bandup[[1;;n-1]],
			Band[{1,1}]->Bandc,
			Band[{2,1}]->Banddw[[1;;n-1]]},n]
Tridclosed[n_,a_,b_,c_]:=SparseArray[{
			{1,n}->c,{n,1}->a,
			Band[{1,2}]->(a&/@Range[1,n-1]),
			Band[{1,1}]->(b&/@Range[1,n-1]),
			Band[{2,1}]->(c&/@Range[1,n-1])},n]
Tridclosed[ndimer_,Mup_?MatrixQ,Mc_?MatrixQ,Mdw_?MatrixQ]:=SparseArray[{
			{{1,2*ndimer-1},{1,2ndimer},{2,2*ndimer-1},{2,2*ndimer}}->(Mdw//Flatten),
			{{2*ndimer-1,1},{2*ndimer-1,2},{2*ndimer,1},{2*ndimer,2}}->(Mup//Flatten),
			Band[{1,3}]->(Mup&/@Range[1,ndimer-1]),
			Band[{1,1}]->(Mc&/@Range[1,ndimer]),
			Band[{3,1}]->(Mdw&/@Range[1,ndimer-1])},2*ndimer]
			



(* ::Subsection::Closed:: *)
(*Distance measures*)


MatrixSqrt[m_?SquareMatrixQ]:=MatrixPower[m,1/2];

MatrixAbs[a_?SquareMatrixQ]:=MatrixSqrt[a . (a\[ConjugateTranspose])];

Fidelity[a_?SquareMatrixQ,b_?SquareMatrixQ]:=(Plus@@(Sqrt[Eigenvalues[a . b]]));

TraceNorm[a_?SquareMatrixQ]:=Plus@@SingularValueList[a];

TraceDistance[a_?SquareMatrixQ,b_?SquareMatrixQ]:=1/2*TraceNorm[a-b];



(* ::Subsection::Closed:: *)
(*Qubits Systems*)


NQubits[NN_]:=Module[{\[Sigma]0,\[Sigma]x,\[Sigma]y,\[Sigma]z,\[Sigma]p,\[Sigma]m,eye,eyeL,eyeR},
\[Sigma]0 = SparseArray[{{1,0},{0,1}}];
\[Sigma]x = SparseArray[{{0,1},{1,0}}];
\[Sigma]y = SparseArray[{{0,-I},{I,0}}];
\[Sigma]z =SparseArray[{{1,0},{0,-1}}];
\[Sigma]p =(\[Sigma]x + I \[Sigma]y)/2;\[Sigma]m = (\[Sigma]x - I \[Sigma]y)/2;
eye = Id[2^NN];
Do[
eyeL = Id[2^(i-1)];
eyeR = Id[2^(NN-i)];
Sx[i] = SparseArray[CircleTimes[eyeL,\[Sigma]x,eyeR]];
Sy[i] = SparseArray[CircleTimes[eyeL,\[Sigma]y,eyeR]];
Sz[i] = SparseArray[CircleTimes[eyeL,\[Sigma]z,eyeR]];
Sp[i] = SparseArray[CircleTimes[eyeL,\[Sigma]p,eyeR]];
Sm[i] = SparseArray[CircleTimes[eyeL,\[Sigma]m,eyeR]];,{i,1,NN}];];


TightBinding[J_,NN_]:=J Total[(Sx[#] . Sx[#+1]+Sy[#] . Sy[#+1])&/@Range[1,NN-1]]


(* ::Subsection::Closed:: *)
(*Random states*)


RandomSimplex[d_]:=Block[{r,r1,r2},
	r=Sort[Table[RandomReal[{0,1}],{i,1,d-1}]];
	r1=Append[r,1];r2=Prepend[r,0];r1-r2
];


RandomKet[n_?IntegerQ]:=Block[{p,ph},
	p=Sqrt[RandomSimplex[n]];
	ph=Exp[I*RandomReal[{0,2\[Pi]},n-1]];
	ph=Prepend[ph,1];
	p*ph
];


GinibreMatrix[m_,n_]:=RandomReal[NormalDistribution[0,1],{m,n}] + I RandomReal[NormalDistribution[0,1],{m,n}];

RandomSpecialUnitary[dim_]:=Module[{U},
	U=RandomUnitary[dim];
	U/Det[U]^(1/dim)
];

RandomUnitary[dim_]:=Module[{q,r,d,ph},
	{q,r}=QRDecomposition[GinibreMatrix[dim,dim]];
	d=Diagonal[r];
	ph=d/Abs[d];
	Transpose[Transpose[q]*ph]
];

RandomOrthogonal[dim_]:=Module[{q,r,d,ph},
    {q,r}=QRDecomposition[RandomReal[NormalDistribution[0,1],{dim,dim}]];
	d=Diagonal[r];
	ph=d/Abs[d];
	Transpose[Transpose[q]*ph]
];

RandomState[d_,dist_:"HS"]:=Block[{A,U},
	Switch[dist,
		"HS",
			A=GinibreMatrix[d,d];
			A=(A . ConjugateTranspose[A]);
			A=Chop[A/Tr[A]],
		"Bures",
			A=GinibreMatrix[d,d];
			U=RandomUnitary[d];
			A=(IdentityMatrix[d]+U) . A . A\[ConjugateTranspose] . (IdentityMatrix[d]+U)\[ConjugateTranspose];
			Chop[A]\[ConjugateTranspose]/Tr[A],		
		_, 
			If[IntegerQ[dist] && dist >=d,
				A=GinibreMatrix[d,dist];
				A=(A . ConjugateTranspose[A]);
				A=Chop[A/Tr[A]],
				Message[RandomState::argerr,dist]
			]
	]
];

RandomPOVM[d_,s_]:=Module[{W,S},
				W=# . #\[ConjugateTranspose]&/@Table[GinibreMatrix[d,s],s];
				S=MatrixPower[Total[W],-1/2];
				Return[S . # . S&/@W]]


(* ::Subsection::Closed:: *)
(*Reshaping, vectorization and reshuffling*)


Vec[m_]:=Flatten[Transpose[m]]; 

Unvec[v_List,cols_:0]:=Which[
 (cols== 0)&&IntegerQ[\[Sqrt]Length[v]],Transpose[Partition[v,\[Sqrt]Length[v]]],
 Mod[Length[v],cols]==0,Transpose[Partition[v,cols]]
];

Res[m_List]:=Flatten[m]; 

Unres[v_List,cols_:0]:=Which[
 (cols== 0)&&IntegerQ[\[Sqrt]Length[v]],Partition[v,\[Sqrt]Length[v]],
 Mod[Length[v],cols]==0,Partition[v,cols]
];

Reshuffle[\[Rho]_]:=Block[{dim},
	dim = Sqrt[Length[\[Rho]]];
	If[And [SquareMatrixQ[\[Rho]] , IntegerQ[dim]] ,
		Reshuffle[\[Rho], {{dim,dim},{dim,dim}}]
	(*else*),
		Message[Reshuffle::argerr]
	]
]
Reshuffle::argerr = "Reshuffle works only for square matrices of dimension \!\(\*SuperscriptBox[\"d\", \"2\"]\)\[Times]\!\(\*SuperscriptBox[\"d\", \"2\"]\), \
where d is an Integer, for other dimensions use ReshuffleGeneral";

Reshuffle[A_,{n_,m_}]:=Flatten[
	Table[Flatten[Part[A,1+i1;;n[[2]]+i1,1+i2;;m[[2]]+i2]],{i1,0,n[[1]] n[[2]]-1,n[[2]]},{i2,0,m[[1]]*m[[2]]-1,m[[2]]}]
,1];


(* ::Subsection::Closed:: *)
(*Partial trace and transposition*)


PartialTrace[A_, {}]:=A;
PartialTrace[A_, dim_, {}]:=A;

PartialTrace[A_, s_] := Block[ {d = Length[A], sys},
    If[ IntegerQ[s],
      sys = {s}, (*else*)
      sys = s
    ];  
    If[  IntegerQ[Sqrt[d]],
    If[ Length[sys]==1 ,
      Switch[sys[[1]],
       1, PartialTrace1[A],
       2, PartialTrace2[A],
       _,Print["B1"]; Message[PartialTrace::syserr, sys]
       ], (*else*)
      If[ sys=={1,2} || sys=={2,1},
        Tr[A], (*else*)
        Print["B2"]; Message[PartialTrace::syserr, sys]
      ]
    ]
    ,
    Message[PartialTrace::dimerr, Dimensions[A]]    
    ]
  ];
  


PartialTrace[A_,dim_?VectorQ,s_]:=Block[{sys},
	If[IntegerQ[s], sys={s}, (*else*) sys=s];
    If[Length[dim] == 2 && Length[sys]==1,
	   Switch[sys[[1]],
        1, PartialTrace1[A,dim],
        2, PartialTrace2[A,dim],
        _, Message[PartialTrace::sysspecerr, Length[dim], Select[sys, Or[#>Length[dim],  # < 1]&]]
        ]
        ,(*else*)
        If[Fold[#1 && (1<= #2 <= Length[dim])&,True,sys], 
        	If[Union[sys] == Range[Length[dim]], 
        		Tr[A], (*else*)
                PartialTraceGeneral[A,dim,sys]
        	],
        (*else*)
            Message[PartialTrace::sysspecerr, Length[dim], Select[sys, Or[#>Length[dim],  # < 1]&]]
        ]
    ]
];

PartialTrace::syserr = "The second argument is expected to be 1, 2 or {1,2} (`1` was given)";
PartialTrace::dimerr = "This function expects square matrix of size d^2\[Times]d^2 (the matrix of size `1` was given)";
PartialTrace::sysspecerr = "The system specification is invalid. In the case of `1`-partite systems, it is inpossible to trace out with respect to sub-systems `2`.";

PartialTrace1[X_] := Block[{d = Sqrt[Length[X]]}, Total[X[[1 + d # ;; d + d #, 1 + d # ;; d + d # ]] & /@ Range[0, d - 1]]];
PartialTrace2[X_] := Block[{d = Sqrt[Length[X]]}, Total[X[[1 + # ;; d^2 ;; d, 1 + # ;; d^2 ;; d]] & /@ Range[0, d - 1]]];
PartialTrace1[X_, {d1_, d2_}] := Total[X[[1 + d2 # ;; d2 + d2 #, 1 + d2 # ;; d2 + d2 # ]] & /@ Range[0, d1 - 1]];
PartialTrace2[X_, {d1_, d2_}] := Total[X[[1 + # ;; d1*d2 ;; d2, 1 + # ;; d1*d2 ;; d2]] & /@ Range[0, d2 - 1]];

ListReshape[list_, shape_] := 
  FlattenAt[Fold[Partition[#1, #2] &, Flatten[list], Reverse[shape]], 
   1];

PartialTraceGeneral[A_,dim_?VectorQ,sys_?VectorQ] := Block[
	{offset, keep, dispose, keepdim, disposedim, perm1, perm2, perm, tensor},
	offset=Length[dim];
	keep=Complement[Range[offset], sys];
	dispose=Union[sys];
	perm1=Join[dispose,keep];
	perm2=perm1+offset;
	perm=Ordering[Join[perm1,perm2]];
	tensor=ListReshape[A, Join[dim,dim]];
	keepdim=Apply[Times, Join[dim, dim][[keep]]];
	disposedim=Apply[Times, Join[dim, dim][[dispose]]];
	tensor=Transpose[tensor,perm];
	tensor=ListReshape[tensor,{disposedim,keepdim,disposedim,keepdim}];
	Sum[tensor[[i,All,i,All]],{i,1,disposedim}]
];

PartialTranspose[\[Rho]_,dim_?VectorQ,sys_?VectorQ]:=Block[{offset,tensor,perm,idx1,idx2,s,targetsys},
	offset=Length[dim];
	tensor=ListReshape[\[Rho], Join[dim,dim]];
	targetsys=Union[sys];
	perm=Range[offset*2];
	For[s=1, s<=Length[targetsys], s+=1, 
		idx1 = Position[perm, targetsys[[s]]][[1, 1]];
		idx2 = Position[perm, targetsys[[s]] + offset][[1, 1]];
		{perm[[idx1]],perm[[idx2]]}={perm[[idx2]],perm[[idx1]]};
	];
	tensor=Transpose[tensor,InversePermutation[perm]];
	ListReshape[tensor,Dimensions[\[Rho]]]
];


(* ::Subsection::Closed:: *)
(*Channels*)


Depo[\[Rho]_,pp_]:={{1/2 (-(-2+pp) \[Rho][[1,1]]+pp \[Rho][[2,2]]),-(-1+pp) \[Rho][[1,2]]},
				{-(-1+pp) \[Rho][[2,1]],1/2 (pp \[Rho][[1,1]]-(-2+pp) \[Rho][[2,2]])}};
AD[\[Rho]_,\[Gamma]_]:={{\[Rho][[1,1]]+\[Gamma] \[Rho][[2,2]],Sqrt[1-\[Gamma]] \[Rho][[1,2]]},{Sqrt[1-\[Gamma]] \[Rho][[2,1]],-(-1+\[Gamma]) \[Rho][[2,2]]}};
Pauli[\[Rho]_,{ppx_,ppy_,ppz_}]:={{\[Rho][[1,1]]-1/4 (ppx+ppy) (\[Rho][[1,1]]-\[Rho][[2,2]]),
	(1-ppz/2) \[Rho][[1,2]]-1/4 (ppx+ppy) \[Rho][[1,2]]+1/4 (ppx-ppy) \[Rho][[2,1]]},
	{(1-ppz/2) \[Rho][[2,1]]-1/4 (ppx+ppy) \[Rho][[2,1]]+1/4 (ppx-ppy) \[Rho][[1,2]],
	\[Rho][[2,2]]+1/4 (ppx+ppy) (\[Rho][[1,1]]-\[Rho][[2,2]])}};


(* ::Subsection:: *)
(*Von Neumann, R\[EAcute]nyi, Relative entropy, Fidelity, Mutual Information*)


vonNeumann[\[Rho]_,base_:2]:= Module[{eigs,func},
	eigs = Eigenvalues[\[Rho]];
	func[x_]:= If[NumberQ[x]&&Re@x<10^-12,0,-x Log[base,x]];
	Sum[func[eigs[[i]]],{i,1,Length@eigs}]]


Renyi[\[Rho]_,\[Alpha]_,base_:2]:= Module[{eigs,func},
	If[\[Alpha] ==1.0,vonNeumann[\[Rho]],
	(eigs = Eigenvalues[\[Rho]];
	1/(1-\[Alpha]) Log[base,Sum[(eigs[[i]])^\[Alpha],{i,1,Length@eigs}]])]]


(* D(\[Rho]||\[Sigma]) *)
KullbackLeibler[\[Rho]_,\[Sigma]_]:=Module[{S1,S2,\[Lambda],Q,log},
	S1 = vonNeumann[\[Rho]];
	{\[Lambda],Q} = Eigensystem[\[Sigma]];
	Q = Q\[Transpose]; 
	log = DiagonalMatrix@Log[2,\[Lambda]];
	S2 = - Tr[Q\[ConjugateTranspose] . \[Rho] . Q . log];
	-S1+S2//Chop];


QuantumFidelity[\[Rho]_,\[Sigma]_]:=Tr[MatrixPower[MatrixPower[\[Rho],1/2] . \[Sigma] . MatrixPower[\[Rho],1/2],1/2]]^2


MutualInformation[\[Rho]_,locdimlist_]:=Module[{\[Rho]A,\[Rho]B,\[Rho]AB,n},
	n= Length@locdimlist;
	\[Rho]A = PartialTrace[\[Rho],locdimlist,{1}];
	\[Rho]B = PartialTrace[\[Rho],locdimlist,{2}];
	Chop[vonNeumann[\[Rho]A]+vonNeumann[\[Rho]B]-vonNeumann[\[Rho]],10^-12]];
(* When subsystems are qubits *)
MutualInformation[\[Rho]_]:= MutualInformation[\[Rho],{2,2}]
TripartiteInformation[\[Rho]_]:=MutualInformation[PartialTrace[\[Rho],{2,2,2},{3}],{2,2}]+MutualInformation[PartialTrace[\[Rho],{2,2,2},{2}],{2,2}]-MutualInformation[\[Rho],{2,4}]


(* information theory utilities *)
shannonEntropy[probs_, base_:2] := DeleteCases[probs, _?PossibleZeroQ] // -Total[# * Log[2, #]] &;
marginalProbabilityDistribution[probs_List, sizes : {__Integer}, remainingDof_Integer] := With[
	{probsMatrix = ArrayReshape[probs, sizes]},
	Total[probsMatrix, Complement[Range @ Length @ sizes, {remainingDof}]]
];
MutualInformationClassic[probs_List, sizes : {_Integer, _Integer} : {2, 2}] := Plus[
	shannonEntropy[marginalProbabilityDistribution[probs, sizes, 1]],
	shannonEntropy[marginalProbabilityDistribution[probs, sizes, 2]],
	- shannonEntropy @ probs
];
conditionalEntropyXcY[probs_List, sizes : {sizeX_Integer, sizeY_Integer} : {2, 2}] := Total @ With[
	{probsMat = ArrayReshape[probs, sizes]},
	Transpose @ probsMat // Map[Total @ # * shannonEntropy[# / Total @ #] &]
];
conditionalEntropyYcX[probs_List, sizes : {sizeX_Integer, sizeY_Integer} : {2, 2}] := Total @ With[
	{probsMat = ArrayReshape[probs, sizes]},
	probsMat // Map[Total @ # * shannonEntropy[# / Total @ #] &]
];





Purity[rho_?HermitianMatrixQ]:=Tr[rho . rho]


(* ::Subsection::Closed:: *)
(*Concurrence and entanglement of formation*)


Concurrence[\[Rho]_?MatrixQ]:=Module[{\[Sigma]y,\[Rho]t,\[Rho]sq,R,eigs},
	\[Sigma]y = {{0,-I},{I,0}};
	\[Rho]t = (\[Sigma]y\[CircleTimes]\[Sigma]y) . \[Rho]\[Conjugate] . (\[Sigma]y\[CircleTimes]\[Sigma]y);
	\[Rho]sq = MatrixPower[\[Rho],1/2];
	R = MatrixPower[\[Rho]sq . \[Rho]t . \[Rho]sq,1/2];
	eigs = Chop@Reverse@Sort@Eigenvalues[R];
	Max[0,Re[eigs[[1]]-eigs[[2]]-eigs[[3]]-eigs[[4]]]]];
Concurrence[psi_?VectorQ]:=Module[{},
	2 Abs[psi[[1]]psi[[4]]-psi[[2]]psi[[3]]]];	
ConcurrenceX[\[Rho]_]:=Module[{K1,K2},
	K1=Abs[\[Rho][[2,3]]]-Sqrt[\[Rho][[1,1]]\[Rho][[4,4]]];
	K2=Abs[\[Rho][[1,4]]]-Sqrt[\[Rho][[2,2]]\[Rho][[3,3]]];
	2*Max[0,K1,K2]];
	
EntanglementOfFormation[\[Rho]_,b_:2]:=Module[{c,h},
	h[x_]:=If[(x==0)||(x==1),0,-x Log[b,x]-(1-x)Log[b,1-x]];
	c = Concurrence[\[Rho]];
	h[(1+Sqrt[1-c^2])/2]];

Negativity[\[Rho]_, {m_, n_}] := Plus@@Abs[Select[Eigenvalues[PartialTranspose[\[Rho], {m, n}, {1}]], # < 0 &]]


(* ::Subsection::Closed:: *)
(*Discord*)


ClAB[\[Rho]\[Rho]_][\[Theta]_,\[Phi]_]:=Module[{B1,B2,p1,p2,r1,r2,SA,SAB,V,Vc},
	V={{Cos[\[Theta]/2],E^(-I \[Phi]) Sin[\[Theta]/2]},{E^(I \[Phi]) Sin[\[Theta]/2],-Cos[\[Theta]/2]}};
	B1=((V . m)\[CircleDot](m . V))\[CircleTimes]id;B2=((V . p)\[CircleDot](p . V))\[CircleTimes]id;
	p1=Tr[B1 . \[Rho]\[Rho] . B1];r1=PartialTrace[B1 . \[Rho]\[Rho] . B1,{2,2},{1}]/p1;
	p2=Tr[B2 . \[Rho]\[Rho] . B2];r2=PartialTrace[B2 . \[Rho]\[Rho] . B2,{2,2},{1}]/p2;
	p1 vonNeumann[r1]+p2 vonNeumann[r2]//N];

DiscZAB[\[Rho]\[Rho]_]:=Module[{SPBr,Sr,\[Theta]A,\[Theta]B,\[CapitalDelta]\[Theta],\[Phi]A,\[Phi]B,\[CapitalDelta]\[Phi],th,phi,Cla},
	SPBr=vonNeumann[PartialTrace[\[Rho]\[Rho],{2,2},{2}]];
	Sr=vonNeumann[\[Rho]\[Rho]];
	\[Theta]A=0;\[Theta]B=\[Pi];\[CapitalDelta]\[Theta]=\[Pi]/20;
	\[Phi]A=0;\[Phi]B=2\[Pi];\[CapitalDelta]\[Phi]=\[Pi]/20;
	{th[1],phi[1],Cla[1]}=Sort[Flatten[Table[{\[Theta]\[Theta],\[Phi]\[Phi],ClAB[\[Rho]\[Rho]][\[Theta]\[Theta],\[Phi]\[Phi]]//N//Chop},{\[Theta]\[Theta],\[Theta]A,\[Theta]B,\[CapitalDelta]\[Theta]},{\[Phi]\[Phi],\[Phi]A,\[Phi]B,\[CapitalDelta]\[Phi]}],1],#1[[3]]<#2[[3]]&][[1]];
	\[Theta]A=th[1]-\[CapitalDelta]\[Theta]/2;\[Theta]B=th[1]+\[CapitalDelta]\[Theta]/2;\[CapitalDelta]\[Theta]=\[CapitalDelta]\[Theta]/40;
	\[Phi]A=phi[1]-\[CapitalDelta]\[Phi]/2;\[Phi]B=phi[1]+\[CapitalDelta]\[Phi]/2;\[CapitalDelta]\[Phi]=\[CapitalDelta]\[Phi]/40;
	{th[2],phi[2],Cla[2]}=Sort[Flatten[Table[{\[Theta]\[Theta],\[Phi]\[Phi],ClAB[\[Rho]\[Rho]][\[Theta]\[Theta],\[Phi]\[Phi]]//N//Chop},{\[Theta]\[Theta],\[Theta]A,\[Theta]B,\[CapitalDelta]\[Theta]},{\[Phi]\[Phi],\[Phi]A,\[Phi]B,\[CapitalDelta]\[Phi]}],1],#1[[3]]<#2[[3]]&][[1]];
	SPBr-Sr+Cla[2](*{th[2],phi[2],SPBr-Sr+Cla[2]}*)];

ClBA[\[Rho]\[Rho]_][\[Theta]_,\[Phi]_]:=Module[{B1,B2,p1,p2,r1,r2,SA,SAB,V,Vc},
	V={{Cos[\[Theta]/2],E^(-I \[Phi]) Sin[\[Theta]/2]},{E^(I \[Phi]) Sin[\[Theta]/2],-Cos[\[Theta]/2]}};
	B1=id\[CircleTimes]((V . m)\[CircleDot](m . V));
	B2=id\[CircleTimes]((V . p)\[CircleDot](p . V));
	p1=Tr[B1 . \[Rho]\[Rho] . B1];r1=PartialTrace[B1 . \[Rho]\[Rho] . B1,{2,2},{2}]/p1;
	p2=Tr[B2 . \[Rho]\[Rho] . B2];r2=PartialTrace[B2 . \[Rho]\[Rho] . B2,{2,2},{2}]/p2;
	p1 vonNeumann[r1]+p2 vonNeumann[r2]//N];

DiscZBA[\[Rho]\[Rho]_]:=Module[{SPAr,Sr,\[Theta]A,\[Theta]B,\[CapitalDelta]\[Theta],\[Phi]A,\[Phi]B,\[CapitalDelta]\[Phi],th,phi,Cla},
	SPAr=vonNeumann[PartialTrace[\[Rho]\[Rho],{2,2},{1}]];
	Sr=vonNeumann[\[Rho]\[Rho]];
	\[Theta]A=0;\[Theta]B=\[Pi];\[CapitalDelta]\[Theta]=\[Pi]/20;
	\[Phi]A=0;\[Phi]B=2\[Pi];\[CapitalDelta]\[Phi]=\[Pi]/20;
	{th[1],phi[1],Cla[1]}=Sort[Flatten[Table[{\[Theta]\[Theta],\[Phi]\[Phi],ClBA[\[Rho]\[Rho]][\[Theta]\[Theta],\[Phi]\[Phi]]//N//Chop},{\[Theta]\[Theta],\[Theta]A,\[Theta]B,\[CapitalDelta]\[Theta]},{\[Phi]\[Phi],\[Phi]A,\[Phi]B,\[CapitalDelta]\[Phi]}],1],#1[[3]]<#2[[3]]&][[1]];
	\[Theta]A=th[1]-\[CapitalDelta]\[Theta]/2;\[Theta]B=th[1]+\[CapitalDelta]\[Theta]/2;\[CapitalDelta]\[Theta]=\[CapitalDelta]\[Theta]/40;
	\[Phi]A=phi[1]-\[CapitalDelta]\[Phi]/2;\[Phi]B=phi[1]+\[CapitalDelta]\[Phi]/2;\[CapitalDelta]\[Phi]=\[CapitalDelta]\[Phi]/40;
	{th[2],phi[2],Cla[2]}=Sort[Flatten[Table[{\[Theta]\[Theta],\[Phi]\[Phi],ClBA[\[Rho]\[Rho]][\[Theta]\[Theta],\[Phi]\[Phi]]//N//Chop},{\[Theta]\[Theta],\[Theta]A,\[Theta]B,\[CapitalDelta]\[Theta]},{\[Phi]\[Phi],\[Phi]A,\[Phi]B,\[CapitalDelta]\[Phi]}],1],#1[[3]]<#2[[3]]&][[1]];
	SPAr-Sr+Cla[2](*{th[2],phi[2],SPAr-Sr+Cla[2]}*)];

GeomDiscordAB[m_]:=Module[{\[Sigma],b,t,M,k},
	\[Sigma][0]=id;\[Sigma][1]=sx;\[Sigma][2]=sy;\[Sigma][3]=sz;
	b=Table[Tr[m . (\[Sigma][0]\[CircleTimes]\[Sigma][j])],{j,1,3}];
	t=Table[Tr[m . (\[Sigma][i]\[CircleTimes]\[Sigma][j])],{i,1,3},{j,1,3}]//Chop;
	M=List[b]\[Transpose] . List[b]+t\[Transpose] . t//Chop;
	k=Max[Eigenvalues[M]//Chop];
	1/2 (Norm[b]^2+Tr[t\[Transpose] . t]-k)];

GeomDiscordBA[m_]:=Module[{\[Sigma],b,t,M,k},
	\[Sigma][0]=id;\[Sigma][1]=sx;\[Sigma][2]=sy;\[Sigma][3]=sz;
	b=Table[Tr[m . (\[Sigma][j]\[CircleTimes]\[Sigma][0])],{j,1,3}];
	t=Table[Tr[m . (\[Sigma][i]\[CircleTimes]\[Sigma][j])],{j,1,3},{i,1,3}]//Chop;
	M=List[b]\[Transpose] . List[b]+t\[Transpose] . t//Chop;
	k=Max[Eigenvalues[M]//Chop];
	1/2 (Norm[b]^2+Tr[t\[Transpose] . t]-k)];


(* ::Subsection::Closed:: *)
(*Lindblad ME*)


\[ScriptCapitalD][OP_][\[Rho]_]:=OP . \[Rho] . OP\[ConjugateTranspose]-1/2OP\[ConjugateTranspose] . OP . \[Rho]-1/2 \[Rho] . OP\[ConjugateTranspose] . OP
\[ScriptCapitalD]c[OP_][\[Rho]_]:=OP\[ConjugateTranspose] . \[Rho] . OP-1/2OP\[ConjugateTranspose] . OP . \[Rho]-1/2 \[Rho] . OP\[ConjugateTranspose] . OP


ME[HH_,OPS__][\[Rho]_]:=-I Comm[HH,\[Rho]]+Total[\[ScriptCapitalD][#][\[Rho]]&/@OPS]


MESolve[\[ScriptCapitalL]_,\[Rho]0_]:=Module[{\[Rho]\[Rho],d\[Rho]\[Rho],\[Tau]=Global`\[Tau],r},
\[Rho]\[Rho][\[Tau]_]=Table[r[k,j][\[Tau]],{k,1,Length[\[Rho]0]},{j,1,Length[\[Rho]0]}];
d\[Rho]\[Rho][\[Tau]_]=Table[r[k,j]'[\[Tau]],{k,1,Length[\[Rho]0]},{j,1,Length[\[Rho]0]}];
\[Rho]\[Rho][\[Tau]]/.DSolve[{d\[Rho]\[Rho][\[Tau]]==\[ScriptCapitalL][\[Rho]\[Rho][\[Tau]]],\[Rho]\[Rho][0]==\[Rho]0},Flatten[\[Rho]\[Rho][\[Tau]]],\[Tau]][[1]]]


MESolvet[\[ScriptCapitalL]_,\[Rho]0_,ti_,tf_]:=NDSolve[{D[\[Rho][t],t]==\[ScriptCapitalL][\[Rho][t]],\[Rho][ti]==\[Rho]0},\[Rho],{t,ti,tf}][[1]][[1,2]]


(* ::Subsection::Closed:: *)
(*Cascade ME*)


\[ScriptCapitalD]cascAB[OPA_,OPB_][\[Rho]_]:=Comm[OPA . \[Rho],OPB\[ConjugateTranspose]]+Comm[OPB,\[Rho] . OPA\[ConjugateTranspose]]+\[ScriptCapitalD][OPB][\[Rho]]+\[ScriptCapitalD][OPA][\[Rho]]


(* ::Subsection:: *)
(*Vectorization *)


Vec2[X_]:=Flatten[X\[Transpose]];
UnVec2[x_]:=With[{d = Sqrt[Length@x]},Partition[x,d]\[Transpose]];


(*vec(L\[Rho]L^\[ConjugateTranspose]-1/2{L\[ConjugateTranspose]L,\[Rho]}) = (L\[Conjugate]\[CircleTimes]L - 1/2\[DoubleStruckCapitalI] \[CircleTimes] L\[ConjugateTranspose]L - 1/2L\[ConjugateTranspose]L \[CircleTimes] \[DoubleStruckCapitalI]) vec(\[Rho])*)
\[ScriptCapitalD]ToVec[L_]:=With[{\[ScriptCapitalI] = Id[Length@L]}, L\[Conjugate]\[CircleTimes]L-1/2 (L\[ConjugateTranspose] . L)\[Transpose]\[CircleTimes]\[ScriptCapitalI]-1/2 \[ScriptCapitalI]\[CircleTimes](L\[ConjugateTranspose] . L)]
\[ScriptCapitalD]cascABToVec[OPA_,OPB_]:=With[{\[ScriptCapitalI]=Id[Length@OPA]},
OPB\[Conjugate]\[CircleTimes]OPA-(OPB\[ConjugateTranspose] . OPA)\[Transpose]\[CircleTimes]\[ScriptCapitalI]+OPA\[Conjugate]\[CircleTimes]OPB-\[ScriptCapitalI]\[CircleTimes](OPA\[ConjugateTranspose] . OPB)]+\[ScriptCapitalD]ToVec[OPB]+\[ScriptCapitalD]ToVec[OPA]
(*-i[H,\[Rho]] = -i(\[DoubleStruckCapitalI] \[CircleTimes] H - H\[Transpose] \[CircleTimes] \[DoubleStruckCapitalI])vec(\[Rho])*)
HToVec[H_]:=With[{\[ScriptCapitalI] = Id[Length@H]},-I (\[ScriptCapitalI]\[CircleTimes]H- H\[Transpose]\[CircleTimes]\[ScriptCapitalI])]

METoVec[HH_,OPS__]:=HToVec[HH]+Total[\[ScriptCapitalD]ToVec[#]&/@OPS];


METoVecSolve[HH_,OPS__,\[Rho]0_]:=Module[{\[Rho]\[Rho],d\[Rho]\[Rho],\[Tau]=Global`\[Tau]},
MatrixExp[METoVec[HH,OPS] \[Tau],Vec2[\[Rho]0]]//Simplify]


(* ::Subsection:: *)
(*Representations*)


Lambda1[i_ ,j_,n_]:=Table[KroneckerDelta[j,\[Mu]]KroneckerDelta[i,\[Nu]] + KroneckerDelta[j,\[Nu]]KroneckerDelta[i,\[Mu]] ,{\[Mu],1,n},{\[Nu],1,n}];

Lambda2[i_ ,j_,n_]:=Table[-I(KroneckerDelta[i,\[Mu]]KroneckerDelta[j,\[Nu]] - KroneckerDelta[i,\[Nu]]KroneckerDelta[j,\[Mu]]) ,{\[Mu],1,n},{\[Nu],1,n}];

Lambda3[i_,n_]:=Sqrt[2/(i^2-i)]DiagonalMatrix[Join[Append[Table[1,{i-1}],-(i-1)],Table[0,{n-i}]]];

GeneralizedPauliMatrices[n_]:=Block[{l1,l2,l3,i,j},
    l1=Flatten[Table[Lambda1[i,j,n],{i,1,n},{j,i+1,n}],1];
    l2=Flatten[Table[Lambda2[i,j,n],{i,1,n},{j,i+1,n}],1];
    l3=Table[Lambda3[i,n],{i,2,n}];
    Join[l1,l2,l3]
];
GEN[n_]:=Join[{Id[2^n]/Sqrt[2^n]},1/Sqrt[2]GeneralizedPauliMatrices[2^n]];
TAU[n_]:=Flatten[Table[\[Alpha]1\[CircleDot]\[Alpha]2,{\[Alpha]1,Id[2^n]},{\[Alpha]2,Id[2^n]}],1];


StateToBloch[A_?VectorQ]:=Block[{dim},
  dim=Log[2,Length[A]]; (Tr[A . (#\[CircleDot]#\[Conjugate])]&/@GEN[dim])
];
StateToBloch[A_?MatrixQ]:=Block[{dim},
  dim=Log[2,Length[A]]; (Tr[2^(dim/2) A . #]&/@GEN[dim])
];

BlochToState[vec_]:=Block[{dim},
	If[IntegerQ[Sqrt[Length[vec]]],
		dim= Sqrt[Length[vec]]-1;
		(*1/dim IdentityMatrix[dim] + vec . GeneralizedPauliMatrices[dim]/Sqrt[2],*)
		2^(-dim/2) (vec . GEN[dim]),
		Message[StateFromBlochVector::argerr, vec];
		Beep[];
	]
];
BlochToState::argerr= "Given vector (`1`) is not a Bloch vector of any dimension.";



(* ::Subsection::Closed:: *)
(*Kraus Operators*)


RandomKrausOPS[n_,M_]:=Module[{Gin,H,Ak},
	Gin=GinibreMatrix[2^n,2^n]&/@Range[M];
	H=Total[#\[ConjugateTranspose] . #&/@Gin];
	Return[# . MatrixPower[H,-1/2]&/@Gin]];
KrausTo\[Phi][Ak_,\[Rho]_]:=Total[# . \[Rho] . #\[ConjugateTranspose]&/@Ak]
KrausToF[Ak_]:=Module[{GG1=Global`GG1},Outer[Tr[KrausTo\[Phi][Ak,#2] . #1]&,GG1,GG1,1]];
FTo\[Phi][F_,\[Rho]_]:=Module[{GG1=Global`GG1},Total[F . (Tr[\[Rho] . #]&/@GG1)GG1]];
\[Phi]ToF[\[Phi]_]:=Module[{GG1=Global`GG1},Outer[Tr[\[Phi][#2] . #1\[ConjugateTranspose]]&,GG1,GG1,1]];
FToS[F_]:=Module[{GG1=Global`GG1,\[Tau]=Global`\[Tau]},
	Outer[Function[{\[Tau]1,\[Tau]2},Tr[F . Outer[Tr[#1 . \[Tau]1\[ConjugateTranspose] . #2 . \[Tau]2]&,GG1,GG1,1]]],\[Tau],\[Tau],1]];
SToF[S_]:=Module[{GG1=Global`GG1,\[Tau]=Global`\[Tau]},
	Outer[Function[{GG1a,GG1b},Tr[S . Outer[Tr[GG1b . #1\[ConjugateTranspose] . GG1a . #2]&,\[Tau],\[Tau],1]]],GG1,GG1,1]];
FToKraus[F_]:=Module[{SS,\[Lambda],X,V,\[Tau]=Global`\[Tau]},
	SS=FToS[F];
	{\[Lambda],X}=Eigensystem[SS];
	V=Sqrt[Abs[\[Lambda]]]X;
	Return[Total[# \[Tau]]&/@V]];
\[Phi]ToKraus[\[Phi]_]:=FToKraus[\[Phi]ToF[\[Phi]]];


(* ::Subsection::Closed:: *)
(*Einstein Sum*)


einsum[in_List->out_, arrays__] := Module[{res = isum[in->out, {arrays}]},
    res /; res=!=$Failed
]

isum[in_List -> out_, arrays_List] := Catch@Module[
    {indices, contracted, uncontracted, contractions, transpose},

    If[Length[in] != Length[arrays],
        Message[einsum::length, Length[in], Length[arrays]];
        Throw[$Failed]
    ];

    MapThread[
        If[IntegerQ@TensorRank[#1] && Length[#1] != TensorRank[#2],
            Message[einsum::shape, #1, #2];
            Throw[$Failed]
        ]&,
        {in, arrays}
    ];

    indices = Tally[Flatten[in, 1]];

    If[DeleteCases[indices, {_, 1|2}] =!= {},
        Message[einsum::repeat, Cases[indices, {x_, Except[1|2]}:>x]];
        Throw[$Failed]
    ];

    uncontracted = Cases[indices, {x_, 1} :> x];

    If[Sort[uncontracted] =!= Sort[out],
        Message[einsum::output, uncontracted, out];
        Throw[$Failed]
    ];

    contracted = Cases[indices, {x_, 2} :> x];
    contractions = Flatten[Position[Flatten[in, 1], #]]& /@ contracted;
    transpose = FindPermutation[uncontracted, out];
    Activate @ TensorTranspose[
        TensorContract[
            Inactive[TensorProduct] @@ arrays,
            contractions
        ],
        transpose
    ]
]

einsum::length = "Number of index specifications (`1`) does not match the number of arrays (`2`)";
einsum::shape = "Index specification `1` does not match the array depth of `2`";
einsum::repeat = "Index specifications `1` are repeated more than twice";
einsum::output = "The uncontracted indices don't match the desired output";


(* ::Subsection:: *)
(*Bloch Sphere*)


bv[r_]:={(r[[1,2]]+r[[2,1]]),I r[[1,2]]-I r[[2,1]],(r[[1,1]]-r[[2,2]])}
point[state_?VectorQ,Col_]:=Graphics3D[{Col,Point[bv[state\[CircleDot]state\[Conjugate]]]}]
point[state_?MatrixQ,Col_]:=Graphics3D[{Col,Point[bv[state]]}]
arrow[state_?VectorQ,Col_]:=Graphics3D[{Col,Thickness@0.004,Sequence@@{Sequence@@Point/@#,Arrow@#}&@{{0,0,0},bv[state\[CircleDot]state\[Conjugate]]}}]
arrow[state_?MatrixQ,Col_]:=Graphics3D[{Col,Thickness@0.004,Sequence@@{Sequence@@Point/@#,Arrow@#}&@{{0,0,0},bv[state]}}]
line[points_,Col_]:=Graphics3D[{Col,Thickness@0.003,Sequence@@{Sequence@@Point/@#,Line@#}&@points}]
sfera={SphericalPlot3D[1,{\[Theta],0,Pi},{\[Phi],0,2 Pi},PlotStyle->None,Mesh->7,MeshStyle->Gray,ImageSize->300,AxesLabel->{"x","y","z"}],
Graphics3D[{White,Opacity@0.3,Sphere[{0,0,0},1],Opacity@1,Thickness@0.004,PointSize@0.02}]};
assi={line[{{0,0,1},{0,0,-1}},Black],line[{{1,0,0},{-1,0,0}},Black],line[{{0,1,0},{0,-1,0}},Black]};


(* ::Section:: *)
(*END*)


End[];


EndPackage[];
