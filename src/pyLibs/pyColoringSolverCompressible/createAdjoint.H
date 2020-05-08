// adjoint
AdjointIO adjIO(mesh);
if(adjIO.useColoring==0) FatalErrorIn("")<<"Set useColoring to true!"<<abort(FatalError);

autoPtr<AdjointSolverRegistry> adjReg
(
    AdjointSolverRegistry::New(mesh)
);

nearWallDist d(mesh);
autoPtr<AdjointRASModel> adjRAS
(
    AdjointRASModel::New(mesh,adjIO,d,thermo)
);

AdjointIndexing adjIdx(mesh,adjIO,adjReg(),adjRAS());

autoPtr<AdjointJacobianConnectivity> adjCon
(
    AdjointJacobianConnectivity::New(mesh,adjIO,adjReg(),adjRAS(),adjIdx)
);

AdjointObjectiveFunction adjObj(mesh,adjIO,adjRAS(),adjIdx,adjCon());
