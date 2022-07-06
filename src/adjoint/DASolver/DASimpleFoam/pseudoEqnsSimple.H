// Duplicate U as pseudo_U
volVectorField pseudo_U("pseudo_U", U);
// Get the pseudo_UEqn,
// the most important thing here is to make sure the l.h.s. matches that of UEqn.
fvVectorMatrix pseudo_UEqn(
    fvm::div(phi, pseudo_U)
    - fvm::laplacian(nuEff, pseudo_U));
pseudo_UEqn.relax();

// Swap upper() and lower()
swap(pseudo_UEqn.upper(), pseudo_UEqn.lower());
// Get UDiag as D
List<scalar> UDiag = pseudo_UEqn.D();
// Get rAU
volScalarField rAU(1.0 / pseudo_UEqn.A());

// *********************************** //
// Duplicate p as pseudo_p
volScalarField pseudo_p("pseudo_p", p);
// Get the pseudo_pEqn,
// the most important thing here is to make sure the l.h.s. matches that of pEqn.
fvScalarMatrix pseudo_pEqn(
    fvm::laplacian(rAU, pseudo_p));
// Swap upper() and lower()
swap(pseudo_pEqn.upper(), pseudo_pEqn.lower());

// *********************************** //
// Duplicate pseudo_nuTilda as nuTilda
volScalarField pseudo_nuTilda("pseudo_nuTilda", nuTilda);
// Get myChi and myFv1
volScalarField myChi("myChi", chi(pseudo_nuTilda, nu));
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

// Get the pseudo_nuTildaEqn,
// the most important thing here is to make sure the l.h.s. mathces that of nuTildaEqn.
// Some explicit terms that only contributes to the r.h.s. are diabled
fvScalarMatrix pseudo_nuTildaEqn(
    fvm::ddt(pseudo_nuTilda)
        + fvm::div(phi, pseudo_nuTilda)
        - fvm::laplacian(DnuTildaEff(pseudo_nuTilda, nu, sigmaNut), pseudo_nuTilda)
    //- Cb2 / sigmaNut * magSqr(fvc::grad(pseudo_nuTilda))
    ==
    //Cb1 * myStilda * pseudo_nuTilda
    -fvm::Sp(Cw1 * fw(myStilda, pseudo_nuTilda, y, kappa, Cw2, Cw3) * pseudo_nuTilda / sqr(y), pseudo_nuTilda));
pseudo_nuTildaEqn.relax();

// Swap upper() and lower()
swap(pseudo_nuTildaEqn.upper(), pseudo_nuTildaEqn.lower());