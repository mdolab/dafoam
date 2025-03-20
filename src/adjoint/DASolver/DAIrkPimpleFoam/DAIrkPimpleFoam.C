/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    This class is modified from OpenFOAM's source code
    applications/solvers/incompressible/pimpleFoam

    OpenFOAM: The Open Source CFD Toolbox

    Copyright (C): 2011-2016 OpenFOAM Foundation

    OpenFOAM License:

        OpenFOAM is free software: you can redistribute it and/or modify it
        under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.
    
        OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
        ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
        FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
        for more details.
    
        You should have received a copy of the GNU General Public License
        along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "DAIrkPimpleFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAIrkPimpleFoam, 0);
addToRunTimeSelectionTable(DASolver, DAIrkPimpleFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAIrkPimpleFoam::DAIrkPimpleFoam(
    char* argsAll,
    PyObject* pyOptions)
    : DASolver(argsAll, pyOptions),
      // Radau23 coefficients and weights
      D10(-2),
      D11(3.0 / 2),
      D12(1.0 / 2),
      D20(2),
      D21(-9.0 / 2),
      D22(5.0 / 2),
      w1(3.0 / 4),
      w2(1.0 / 4),

      // SA-fv3 model coefficients
      sigmaNut(0.66666),
      kappa(0.41),
      Cb1(0.1355),
      Cb2(0.622),
      Cw1(Cb1 / sqr(kappa) + (1.0 + Cb2) / sigmaNut),
      Cw2(0.3),
      Cw3(2.0),
      Cv1(7.1),
      Cv2(5.0)
{
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Functions for SA-fv3 model
#include "mySAModel.H"

// Some utilities, move to DAUtility later
#include "myUtilities.H"

void DAIrkPimpleFoam::calcPriResIrkOrig(
    volVectorField& U0, // oldTime U
    volVectorField& U1, // 1st stage
    volScalarField& p1,
    surfaceScalarField& phi1,
    volScalarField& nuTilda1,
    volScalarField& nut1,
    volVectorField& U2, // 2nd stage
    volScalarField& p2,
    surfaceScalarField& phi2,
    volScalarField& nuTilda2,
    volScalarField& nut2,
    const volScalarField& nu,
    const scalar& deltaT, // current dt
    volVectorField& U1Res, // Residual for 1st stage
    volScalarField& p1Res,
    surfaceScalarField& phi1Res,
    volVectorField& U2Res, // Residual for end stage
    volScalarField& p2Res,
    surfaceScalarField& phi2Res,
    const scalar& relaxUEqn)
{
    // Numerical settings
    word divUScheme = "div(phi,U)";
    word divGradUScheme = "div((nuEff*dev2(T(grad(U)))))";

    // Update boundaries
    U0.correctBoundaryConditions(); // oldTime U
    U1.correctBoundaryConditions(); // 1st stage
    p1.correctBoundaryConditions();
    nuTilda1.correctBoundaryConditions();
    U2.correctBoundaryConditions(); // 2nd stage
    p2.correctBoundaryConditions();
    nuTilda2.correctBoundaryConditions();

    // --- 1st stage
    {
        // Get nuEff1
        this->correctNut(nut1, nuTilda1, nu);
        volScalarField nuEff1("nuEff1", nu + nut1);

        // Initialize U1Eqn w/o ddt term
        fvVectorMatrix U1Eqn(
            fvm::div(phi1, U1, divUScheme)
            //+ turbulence->divDevReff(U)
            - fvm::laplacian(nuEff1, U1)
            - fvc::div(nuEff1 * dev2(T(fvc::grad(U1))), divGradUScheme));

        // Update U1Eqn with pseudo-spectral terms
        forAll(U1, cellI)
        {
            scalar meshV = U1.mesh().V()[cellI];

            // Add D11 / halfDeltaT[i] * V() to diagonal
            U1Eqn.diag()[cellI] += D11 / deltaT * meshV; // Use one seg for now: halfDeltaTList[segI]

            // Minus D10 / halfDeltaT[i] * T0 * V() to source term
            U1Eqn.source()[cellI] -= D10 / deltaT * U0[cellI] * meshV;

            // Minus D12 / halfDeltaT[i] * T2 * V() to source term
            U1Eqn.source()[cellI] -= D12 / deltaT * U2[cellI] * meshV;
        }

        U1Res = (U1Eqn & U1) + fvc::grad(p1);

        // We use relaxation factor = 1.0, cannot skip the step below
        U1Eqn.relax(relaxUEqn);

        volScalarField rAU1(1.0 / U1Eqn.A());
        volVectorField HbyA1(constrainHbyA(rAU1 * U1Eqn.H(), U1, p1));
        surfaceScalarField phiHbyA1(
            "phiHbyA1",
            fvc::flux(HbyA1));
        tmp<volScalarField> rAtU1(rAU1);

        fvScalarMatrix p1Eqn(
            fvm::laplacian(rAtU1(), p1) == fvc::div(phiHbyA1));

        p1Res = p1Eqn & p1;

        // Then do phiRes
        phi1Res = phiHbyA1 - p1Eqn.flux() - phi1;
    }

    // --- 2nd stage
    {
        // Get nuEff2
        this->correctNut(nut2, nuTilda2, nu);
        volScalarField nuEff2("nuEff2", nu + nut2);

        // Initialize U2Eqn w/o ddt term
        fvVectorMatrix U2Eqn(
            fvm::div(phi2, U2, divUScheme)
            //+ turbulence->divDevReff(U)
            - fvm::laplacian(nuEff2, U2)
            - fvc::div(nuEff2 * dev2(T(fvc::grad(U2))), divGradUScheme));

        // Update U2Eqn with pseudo-spectral terms
        forAll(U2, cellI)
        {
            scalar meshV = U2.mesh().V()[cellI];

            // Add D22 / halfDeltaT[i] * V() to diagonal
            U2Eqn.diag()[cellI] += D22 / deltaT * meshV; // Use one seg for now: halfDeltaTList[segI]

            // Minus D20 / halfDeltaT[i] * T0 * V() to source term
            U2Eqn.source()[cellI] -= D20 / deltaT * U0[cellI] * meshV;

            // Minus D21 / halfDeltaT[i] * T2 * V() to source term
            U2Eqn.source()[cellI] -= D21 / deltaT * U1[cellI] * meshV;
        }

        U2Res = (U2Eqn & U2) + fvc::grad(p2);

        // We use relaxation factor = 1.0, cannot skip the step below
        U2Eqn.relax(relaxUEqn);

        volScalarField rAU2(1.0 / U2Eqn.A());
        volVectorField HbyA2(constrainHbyA(rAU2 * U2Eqn.H(), U2, p2));
        surfaceScalarField phiHbyA2(
            "phiHbyA2",
            fvc::flux(HbyA2));
        tmp<volScalarField> rAtU2(rAU2);

        // Non-orthogonal pressure corrector loop
        fvScalarMatrix p2Eqn(
            fvm::laplacian(rAtU2(), p2) == fvc::div(phiHbyA2));

        p2Res = p2Eqn & p2;

        // Then do phiRes
        phi2Res = phiHbyA2 - p2Eqn.flux() - phi2;
    }
}

void DAIrkPimpleFoam::calcPriSAResIrkOrig(
    volScalarField& nuTilda0, // oldTime nuTilda
    volVectorField& U1, // 1st stage
    surfaceScalarField& phi1,
    volScalarField& nuTilda1,
    volVectorField& U2, // 2nd stage
    surfaceScalarField& phi2,
    volScalarField& nuTilda2,
    volScalarField& y,
    const volScalarField& nu,
    const scalar& deltaT, // current dt
    volScalarField& nuTilda1Res, // Residual for 1st stage
    volScalarField& nuTilda2Res) // Residual for 2nd stage
{
    // Numerical settings
    word divNuTildaScheme = "div(phi,nuTilda)";

    // Update boundaries
    nuTilda0.correctBoundaryConditions();
    U1.correctBoundaryConditions();
    nuTilda1.correctBoundaryConditions();
    U2.correctBoundaryConditions();
    nuTilda2.correctBoundaryConditions();

    // --- 1st stage
    {
        // Get chi1 and fv11
        volScalarField chi1("chi1", chi(nuTilda1, nu));
        volScalarField fv11("fv11", fv1(chi1));

        // Get Stilda1
        volScalarField Stilda1(
            "Stilda1",
            fv3(chi1, fv11) * ::sqrt(2.0) * mag(skew(fvc::grad(U1))) + fv2(chi1, fv11) * nuTilda1 / sqr(kappa * y));

        // Construct nuTilda1Eqn w/o ddt term
        fvScalarMatrix nuTilda1Eqn(
            fvm::div(phi1, nuTilda1, divNuTildaScheme)
                - fvm::laplacian(DnuTildaEff(nuTilda1, nu), nuTilda1)
                - Cb2 / sigmaNut * magSqr(fvc::grad(nuTilda1))
            == Cb1 * Stilda1 * nuTilda1 // If field inversion, beta should be multiplied here
                - fvm::Sp(Cw1 * fw(Stilda1, nuTilda1, y) * nuTilda1 / sqr(y), nuTilda1));

        // Update nuTilda1Eqn with pseudo-spectral terms
        forAll(nuTilda1, cellI)
        {
            scalar meshV = nuTilda1.mesh().V()[cellI];

            // Add D11 / halfDeltaT[i] * V() to diagonal
            nuTilda1Eqn.diag()[cellI] += D11 / deltaT * meshV;

            // Minus D10 / halfDeltaT[i] * T0 * V() to source term
            nuTilda1Eqn.source()[cellI] -= D10 / deltaT * nuTilda0[cellI] * meshV;

            // Minus D12 / halfDeltaT[i] * T2 * V() to source term
            nuTilda1Eqn.source()[cellI] -= D12 / deltaT * nuTilda2[cellI] * meshV;
        }

        nuTilda1Res = nuTilda1Eqn & nuTilda1;
    }

    // --- 2nd stage
    {
        // Get chi2 and fv12
        volScalarField chi2("chi2", chi(nuTilda2, nu));
        volScalarField fv12("fv12", fv1(chi2));

        // Get Stilda2
        volScalarField Stilda2(
            "Stilda2",
            fv3(chi2, fv12) * ::sqrt(2.0) * mag(skew(fvc::grad(U2))) + fv2(chi2, fv12) * nuTilda2 / sqr(kappa * y));

        // Construct nuTilda2Eqn w/o ddt term
        fvScalarMatrix nuTilda2Eqn(
            fvm::div(phi2, nuTilda2, divNuTildaScheme)
                - fvm::laplacian(DnuTildaEff(nuTilda2, nu), nuTilda2)
                - Cb2 / sigmaNut * magSqr(fvc::grad(nuTilda2))
            == Cb1 * Stilda2 * nuTilda2 // If field inversion, beta should be multiplied here
                - fvm::Sp(Cw1 * fw(Stilda2, nuTilda2, y) * nuTilda2 / sqr(y), nuTilda2));

        // Update nuTilda2Eqn with pseudo-spectral terms
        forAll(nuTilda2, cellI)
        {
            scalar meshV = nuTilda2.mesh().V()[cellI];

            // Add D22 / halfDeltaT[i] * V() to diagonal
            nuTilda2Eqn.diag()[cellI] += D22 / deltaT * meshV;

            // Minus D20 / halfDeltaT[i] * T0 * V() to source term
            nuTilda2Eqn.source()[cellI] -= D20 / deltaT * nuTilda0[cellI] * meshV;

            // Minus D21 / halfDeltaT[i] * T2 * V() to source term
            nuTilda2Eqn.source()[cellI] -= D21 / deltaT * nuTilda1[cellI] * meshV;
        }

        nuTilda2Res = nuTilda2Eqn & nuTilda2;
    }
}

void DAIrkPimpleFoam::initSolver()
{
    /*
    Description:
        Initialize variables for DASolver
    */
    daOptionPtr_.reset(new DAOption(meshPtr_(), pyOptions_));
}

label DAIrkPimpleFoam::solvePrimal()
{
    /*
    Description:
        Call the primal solver to get converged state variables

    Output:
        state variable vector
    */

    Foam::argList& args = argsPtr_();
#include "createTime.H"
#include "createMesh.H"
#include "initContinuityErrs.H"
#include "createControl.H"
#include "createFieldsIrkPimple.H"
#include "CourantNo.H"

    // Turbulence disabled
    //turbulence->validate();

    // Get nu, nut, nuTilda, y
    volScalarField& nu = const_cast<volScalarField&>(mesh.thisDb().lookupObject<volScalarField>("nu"));
    volScalarField& nut = const_cast<volScalarField&>(mesh.thisDb().lookupObject<volScalarField>("nut"));
    volScalarField& nuTilda = const_cast<volScalarField&>(mesh.thisDb().lookupObject<volScalarField>("nuTilda"));
    volScalarField& y = const_cast<volScalarField&>(mesh.thisDb().lookupObject<volScalarField>("yWall"));

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info << "\nStarting time loop\n"
         << endl;

    // get IRKDict settings, default to Radau23 for now
    IOdictionary IRKDict(
        IOobject(
            "IRKDict",
            mesh.time().system(),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE));

    scalar relaxU = 1.0;
    if (IRKDict.found("relaxU"))
    {
        if (IRKDict.getScalar("relaxU") > 0)
        {
            relaxU = IRKDict.getScalar("relaxU");
        }
    }

    scalar relaxP = 1.0;
    if (IRKDict.found("relaxP"))
    {
        if (IRKDict.getScalar("relaxP") > 0)
        {
            relaxP = IRKDict.getScalar("relaxP");
        }
    }

    scalar relaxPhi = 1.0;
    if (IRKDict.found("relaxPhi"))
    {
        if (IRKDict.getScalar("relaxPhi") > 0)
        {
            relaxPhi = IRKDict.getScalar("relaxPhi");
        }
    }

    scalar relaxNuTilda = 1.0;
    if (IRKDict.found("relaxNuTilda"))
    {
        if (IRKDict.getScalar("relaxNuTilda") > 0)
        {
            relaxNuTilda = IRKDict.getScalar("relaxNuTilda");
        }
    }

    scalar relaxStage1 = 0.8;
    if (IRKDict.found("relaxStage1"))
    {
        if (IRKDict.getScalar("relaxStage1") > 0)
        {
            relaxStage1 = IRKDict.getScalar("relaxStage1");
        }
    }

    scalar relaxStage2 = 0.8;
    if (IRKDict.found("relaxStage2"))
    {
        if (IRKDict.getScalar("relaxStage2") > 0)
        {
            relaxStage2 = IRKDict.getScalar("relaxStage2");
        }
    }

    scalar relaxUEqn = 1.0;
    scalar relaxNuTildaEqn = 1.0;
    if (IRKDict.found("relaxNuTildaEqn"))
    {
        if (IRKDict.getScalar("relaxNuTildaEqn") > 0)
        {
            relaxNuTildaEqn = IRKDict.getScalar("relaxNuTildaEqn");
        }
    }

    label maxSweep = 10;
    if (IRKDict.found("maxSweep"))
    {
        if (IRKDict.getLabel("maxSweep") > 0)
        {
            maxSweep = IRKDict.getLabel("maxSweep");
        }
    }

    // Duplicate state variables for stages
    volVectorField U1("U1", U);
    volVectorField U2("U2", U);
    volScalarField p1("p1", p);
    volScalarField p2("p2", p);
    surfaceScalarField phi1("phi1", phi);
    surfaceScalarField phi2("phi2", phi);
    // SA turbulence model
    volScalarField nuTilda1("nuTilda1", nuTilda);
    volScalarField nuTilda2("nuTilda2", nuTilda);
    volScalarField nut1("nut1", nut);
    volScalarField nut2("nut2", nut);

    // Settings for stage pressure
    mesh.setFluxRequired(p1.name());
    mesh.setFluxRequired(p2.name());

    // Initialize primal residuals
    volVectorField U1Res(
        IOobject(
            "U1Res",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE),
        mesh,
        dimensionedVector("U1Res", dimensionSet(0, 1, -2, 0, 0, 0, 0), vector::zero),
        zeroGradientFvPatchField<vector>::typeName);
    volVectorField U2Res("U2Res", U1Res);

    volScalarField p1Res(
        IOobject(
            "p1Res",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE),
        mesh,
        dimensionedScalar("p1Res", dimensionSet(0, 0, -1, 0, 0, 0, 0), 0.0),
        zeroGradientFvPatchField<scalar>::typeName);
    volScalarField p2Res("p2Res", p1Res);

    surfaceScalarField phi1Res(
        IOobject(
            "phi1Res",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE),
        phi * 0.0);
    surfaceScalarField phi2Res("phi2Res", phi1Res);

    volScalarField nuTilda1Res(
        IOobject(
            "nuTilda1Res",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE),
        mesh,
        dimensionedScalar("nuTilda1Res", dimensionSet(0, 2, -2, 0, 0, 0, 0), 0.0),
        zeroGradientFvPatchField<scalar>::typeName);
    volScalarField nuTilda2Res("nuTilda2Res", nuTilda1Res);

    // Initialize oldTime() for under-relaxation
    U1.oldTime() = U1;
    U2.oldTime() = U2;
    p1.oldTime() = p1;
    p2.oldTime() = p2;
    phi1.oldTime() = phi1;
    phi2.oldTime() = phi2;
    nuTilda1.oldTime() = nuTilda1;
    nuTilda2.oldTime() = nuTilda2;

    // Numerical settings
    word divUScheme = "div(phi,U)";
    word divGradUScheme = "div((nuEff*dev2(T(grad(U)))))";
    word divNuTildaScheme = "div(phi,nuTilda)";

    const fvSolution& myFvSolution = mesh.thisDb().lookupObject<fvSolution>("fvSolution");
    dictionary solverDictU = myFvSolution.subDict("solvers").subDict("U");
    dictionary solverDictP = myFvSolution.subDict("solvers").subDict("p");
    dictionary solverDictNuTilda = myFvSolution.subDict("solvers").subDict("nuTilda");

    while (runTime.run())
    {

#include "CourantNo.H"

        ++runTime;

        Info << "Time = " << runTime.timeName() << nl << endl;

        scalar deltaT = runTime.deltaTValue();

        // --- GS sweeps for IRK-PIMPLE
        label sweepIndex = 0;
        while (sweepIndex < maxSweep)
        {
            Info << "Block GS sweep = " << sweepIndex + 1 << endl;

            // --- 1st stage
            {
#include "U1EqnIrkPimple.H"

                while (pimple.correct())
                {
#include "p1EqnIrkPimple.H"
                }

                // --- Correct turbulence, using our own SAFv3
#include "nuTilda1EqnIrkPimple.H"
            }

            // --- 2nd stage
            {
#include "U2EqnIrkPimple.H"

                while (pimple.correct())
                {
#include "p2EqnIrkPimple.H"
                }

                // --- Correct turbulence, using our own SAFv3
#include "nuTilda2EqnIrkPimple.H"
            }

            this->calcPriResIrkOrig(U, U1, p1, phi1, nuTilda1, nut1, U2, p2, phi2, nuTilda2, nut2, nu, deltaT, U1Res, p1Res, phi1Res, U2Res, p2Res, phi2Res, relaxUEqn);
            this->calcPriSAResIrkOrig(nuTilda, U1, phi1, nuTilda1, U2, phi2, nuTilda2, y, nu, deltaT, nuTilda1Res, nuTilda2Res);

            Info << "L2 norm of U1Res: " << this->L2norm(U1Res.primitiveField()) << endl;
            Info << "L2 norm of U2Res: " << this->L2norm(U2Res.primitiveField()) << endl;
            Info << "L2 norm of p1Res: " << this->L2norm(p1Res.primitiveField()) << endl;
            Info << "L2 norm of p2Res: " << this->L2norm(p2Res.primitiveField()) << endl;
            Info << "L2 norm of phi1Res: " << this->L2norm(phi1Res, phi1.mesh().magSf()) << endl;
            Info << "L2 norm of phi2Res: " << this->L2norm(phi2Res, phi2.mesh().magSf()) << endl;
            Info << "L2 norm of nuTilda1Res: " << this->L2norm(nuTilda1Res.primitiveField()) << endl;
            Info << "L2 norm of nuTilda2Res: " << this->L2norm(nuTilda2Res.primitiveField()) << endl;

            sweepIndex++;
        }

        // Update new step values before write-to-disk
        U = U2;
        U.correctBoundaryConditions();
        p = p2;
        p.correctBoundaryConditions();
        phi = phi2;
        nuTilda = nuTilda2;
        nuTilda.correctBoundaryConditions();
        nut = nut2;
        nut.correctBoundaryConditions();

        // Write to disk
        runTime.write(); // This writes U, p, phi, nuTilda, nut
        // Also write internal stages to disk (Radau23)
        U1.write();
        p1.write();
        phi1.write();
        nuTilda1.write();
        nut1.write();

        // Use old step as initial guess for the next step
        U1 = U;
        U1.correctBoundaryConditions();
        p1 = p;
        p1.correctBoundaryConditions();
        phi1 = phi;
        nuTilda1 = nuTilda;
        nuTilda1.correctBoundaryConditions();
        nut1 = nut;
        nut1.correctBoundaryConditions();

        runTime.printExecutionTime(Info);
    }

    Info << "End\n"
         << endl;

    return 0;
}

} // End namespace Foam

// ************************************************************************* //
