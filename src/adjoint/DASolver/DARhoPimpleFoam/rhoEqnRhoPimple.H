
{
    fvScalarMatrix rhoEqn(
        fvm::ddt(rho)
        + fvc::div(phi));

    rhoEqn.solve();
}
