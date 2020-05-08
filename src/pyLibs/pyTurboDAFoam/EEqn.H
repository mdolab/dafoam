{
    volScalarField& he = thermo.he();
    volSymmTensorField Teff = - turbulence->devRhoReff();

    URel == U;
    MRF.makeRelative(URel);

    fvScalarMatrix EEqn
    (
        fvm::div(phi, he)
      + (
            he.name() == "e"
          ? fvc::div(phi, volScalarField("Ekp", 0.5*magSqr(U) + p/rho))
          : fvc::div(phi, volScalarField("K", 0.5*magSqr(U))) - fvc::div( Teff.T()&U ) + fvc::div( p*(U-URel) )
        )
      - fvm::Sp(fvc::div(phi), he)
      - fvm::laplacian(turbulence->alphaEff(), he)
      == 
        fvOptions(rho, he)
    );

    EEqn.relax();
 
    fvOptions.constrain(EEqn);
 
    EEqn.solve(); 
 
    fvOptions.correct(he);

    // bound he
    adjIO.boundVar(he);

    thermo.correct();
}
