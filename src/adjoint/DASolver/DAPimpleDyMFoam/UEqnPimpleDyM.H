// Solve the Momentum equation

tmp<fvVectorMatrix> tUEqn(
    fvm::ddt(U)
    + fvm::div(phi, U)
    + turbulence->divDevReff(U));

fvVectorMatrix& UEqn = tUEqn.ref();

UEqn.relax();

if (pimple.momentumPredictor())
{
    solve(UEqn == -fvc::grad(p));
}
