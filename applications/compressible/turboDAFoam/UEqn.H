    // Solve the Momentum equation

    MRF.correctBoundaryVelocity(U);

    tmp<fvVectorMatrix> tUEqn
    (
        fvm::div(phi, U)
      + MRF.DDt(rho, U)
      //+ turbulence->divDevRhoReff(U)
      + adjRAS->divDevRhoReff(U)
     ==
       fvOptions(rho, U)
    );
    fvVectorMatrix& UEqn = tUEqn.ref();

    UEqn.relax();
    
    fvOptions.constrain(UEqn);

    solve(UEqn == -fvc::grad(p));

    fvOptions.correct(U);

    // bound U
    adjIO.boundVar(U);
    U.correctBoundaryConditions();
