// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
// Get SA member functions

// Note: scalar consts are now declared in DAIrkPimpleFoam.H
// but fields such as nu, nut, nuTilda, and y still need to be passed as arguments

tmp<volScalarField> DAIrkPimpleFoam::chi(
    volScalarField& nuTilda,
    const volScalarField& nu)
{
    return nuTilda / nu;
}

tmp<volScalarField> DAIrkPimpleFoam::fv1(
    volScalarField& chi)
{
    volScalarField chi3(pow3(chi));
    return chi3 / (chi3 + pow3(Cv1));
}

tmp<volScalarField> DAIrkPimpleFoam::fv2(
    volScalarField& chi,
    volScalarField& fv1)
{
    return 1.0 / pow3(scalar(1) + chi / Cv2);
}

tmp<volScalarField> DAIrkPimpleFoam::fv3(
    volScalarField& chi,
    volScalarField& fv1)
{

    const volScalarField chiByCv2((1 / Cv2) * chi);

    return (scalar(1) + chi * fv1)
        * (1 / Cv2)
        * (3 * (scalar(1) + chiByCv2) + sqr(chiByCv2))
        / pow3(scalar(1) + chiByCv2);
}

tmp<volScalarField> DAIrkPimpleFoam::fw(
    volScalarField& Stilda,
    volScalarField& nuTilda,
    volScalarField& y)
{
    volScalarField r(
        min(
            nuTilda
                / (max(
                       Stilda,
                       dimensionedScalar("SMALL", Stilda.dimensions(), SMALL))
                   * sqr(kappa * y)),
            scalar(10.0)));
    r.boundaryFieldRef() == 0.0;

    // Keep or remove const in adjoint?
    const volScalarField g(r + Cw2 * (pow6(r) - r));

    return g * pow((1.0 + pow6(Cw3)) / (pow6(g) + pow6(Cw3)), 1.0 / 6.0);
}

tmp<volScalarField> DAIrkPimpleFoam::DnuTildaEff(
    volScalarField& nuTilda,
    const volScalarField& nu)
{
    return tmp<volScalarField>(
        new volScalarField("DnuTildaEff", (nuTilda + nu) / sigmaNut));
}

void DAIrkPimpleFoam::correctNut(
    volScalarField& nut,
    volScalarField& nuTilda,
    const volScalarField& nu)
{
    volScalarField chi(this->chi(nuTilda, nu));
    volScalarField fv1(this->fv1(chi));
    nut = nuTilda * fv1;

    nut.correctBoundaryConditions();

    // Below is in DAFoam, but disabled here
    // this is basically BasicTurbulenceModel::correctNut();
    //this->correctAlphat();

    return;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
