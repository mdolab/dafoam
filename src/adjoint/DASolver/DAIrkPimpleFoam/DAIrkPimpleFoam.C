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
    : DASolver(argsAll, pyOptions)
{
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

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

    // Get nu from memory
    volScalarField& nu = const_cast<volScalarField&>(mesh.thisDb().lookupObject<volScalarField>("nu"));

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info << "\nStarting time loop\n"
         << endl;

    // Radau23 coefficients and weights
    scalar D10 = -2;
    scalar D11 = 3.0 / 2;
    scalar D12 = 1.0 / 2;
    scalar D20 = 2;
    scalar D21 = -9.0 / 2;
    scalar D22 = 5.0 / 2;
    scalar w1 = 3.0 / 4;
    scalar w2 = 1.0 / 4;

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

    // Settings for stage pressure
    mesh.setFluxRequired(p1.name());
    mesh.setFluxRequired(p2.name());

    // Note: below is not working somehow...
    /*
    // IO settings for internal stages
    U1.writeOpt() = IOobject::AUTO_WRITE;
    p1.writeOpt() = IOobject::AUTO_WRITE;
    phi1.writeOpt() = IOobject::AUTO_WRITE;
    */

    // Initialize oldTime() for under-relaxation
    U1.oldTime() = U1;
    U2.oldTime() = U2;
    p1.oldTime() = p1;
    p2.oldTime() = p2;
    phi1.oldTime() = phi1;
    phi2.oldTime() = phi2;

    // Numerical settings
    word divUScheme = "div(phi,U)";
    //word divGradUScheme = "div((nuEff*dev2(T(grad(U)))))";

    const fvSolution& myFvSolution = mesh.thisDb().lookupObject<fvSolution>("fvSolution");
    dictionary solverDictU = myFvSolution.subDict("solvers").subDict("U");
    dictionary solverDictP = myFvSolution.subDict("solvers").subDict("p");

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

            {
#include "U1EqnIrkPimple.H"

                while (pimple.correct())
                {
#include "p1EqnIrkPimple.H"
                }
            }

            {
#include "U2EqnIrkPimple.H"

                while (pimple.correct())
                {
#include "p2EqnIrkPimple.H"
                }
            }

            sweepIndex++;
        }

        // Update new step values before write-to-disk
        U = U2;
        p = p2;
        phi = phi2;

        runTime.write();
        // Also write internal stages to disk (Radau23)
        U1.write();
        p1.write();
        phi1.write();

        // Use old step as initial guess for the next step
        U1 = U;
        U1.correctBoundaryConditions();
        p1 = p;
        p1.correctBoundaryConditions();
        phi1 = phi;

        runTime.printExecutionTime(Info);
    }

    Info << "End\n"
         << endl;

    return 0;
}

} // End namespace Foam

// ************************************************************************* //
