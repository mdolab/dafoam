word divScheme = "div(phi,U)";
if (daOptionPtr_->getSubDictOption<label>("adjEqnOption", "fpPCUpwind"))
{
    divScheme = "div(pc)";
}
// Duplicate U as pseudoU
volVectorField pseudoU("pseudoU", U);
// Get the pseudoUEqn,
// the most important thing here is to make sure the l.h.s. matches that of UEqn.
fvVectorMatrix pseudoUEqn(
    fvm::div(phi, pseudoU, divScheme)
    - fvm::laplacian(nuEff, pseudoU));
pseudoUEqn.relax(relaxUEqn_);

// Swap upper() and lower()
swap(pseudoUEqn.upper(), pseudoUEqn.lower());
// Get UDiag as D
List<scalar> UDiag = pseudoUEqn.D();
// Get rAU
volScalarField rAU(1.0 / pseudoUEqn.A());

// *********************************** //
// Duplicate p as pseudoP
volScalarField pseudoP("pseudoP", p);
// Get the pseudoPEqn,
// the most important thing here is to make sure the l.h.s. matches that of pEqn.
fvScalarMatrix pseudoPEqn(
    fvm::laplacian(rAU, pseudoP));
// Swap upper() and lower()
swap(pseudoPEqn.upper(), pseudoPEqn.lower());

// *********************************** //
// Use lookupObject to access pseudoNuTilda
volScalarField& pseudoNuTilda = const_cast<volScalarField&>(db.lookupObject<volScalarField>("pseudoNuTilda"));
// Construct pseudoNuTildaEqn
daTurbulenceModelPtr_->constructPseudoNuTildaEqn();

/*
// Get myChi and myFv1
volScalarField myChi("myChi", chi(pseudoNuTilda, nu));
volScalarField myFv1("myFv1", fv1(myChi, Cv1));
// Alternatively, we can do below
//volScalarField myChi = chi(nuTilda, nu);
//volScalarField myFv1 = fv1(myChi, Cv1);

// Get myStilda
volScalarField myStilda(
    "myStilda",
    fv3(myChi, myFv1, Cv2) * ::sqrt(2.0) * mag(skew(fvc::grad(U))) + fv2(myChi, myFv1, Cv2) * nuTilda / sqr(kappa * y));
// Alternatively, we can do below
//volScalarField myStilda = Stilda(myChi, myFv1, U, nuTilda, y, kappa, Cs);

// Get the pseudoNuTildaEqn,
// the most important thing here is to make sure the l.h.s. mathces that of nuTildaEqn.
// Some explicit terms that only contributes to the r.h.s. are diabled
fvScalarMatrix pseudoNuTildaEqn(
    fvm::ddt(pseudoNuTilda)
        + fvm::div(phi, pseudoNuTilda)
        - fvm::laplacian(DnuTildaEff(pseudoNuTilda, nu, sigmaNut), pseudoNuTilda)
    //- Cb2 / sigmaNut * magSqr(fvc::grad(pseudoNuTilda))
    ==
    //Cb1 * myStilda * pseudoNuTilda
    -fvm::Sp(Cw1 * fw(myStilda, pseudoNuTilda, y, kappa, Cw2, Cw3) * pseudoNuTilda / sqr(y), pseudoNuTilda));
pseudoNuTildaEqn.relax();

// Swap upper() and lower()
swap(pseudoNuTildaEqn.upper(), pseudoNuTildaEqn.lower());
*/