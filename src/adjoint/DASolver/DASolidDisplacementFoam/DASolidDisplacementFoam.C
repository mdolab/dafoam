/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    This class is modified from OpenFOAM's source code
    applications/solvers/stressAnalysis/solidDisplacementFoam

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

#include "DASolidDisplacementFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DASolidDisplacementFoam, 0);
addToRunTimeSelectionTable(DASolver, DASolidDisplacementFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DASolidDisplacementFoam::DASolidDisplacementFoam(
    char* argsAll,
    PyObject* pyOptions)
    : DASolver(argsAll, pyOptions),
      rhoPtr_(nullptr),
      muPtr_(nullptr),
      lambdaPtr_(nullptr),
      EPtr_(nullptr),
      nuPtr_(nullptr),
      DPtr_(nullptr),
      gradDPtr_(nullptr)
{
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DASolidDisplacementFoam::initSolver()
{
    /*
    Description:
        Initialize variables for DASolver
    */

    Info << "Initializing fields for DASolidDisplacementFoam" << endl;
    Time& runTime = runTimePtr_();
    fvMesh& mesh = meshPtr_();
#include "createFieldsSolidDisplacement.H"
#include "createAdjoint.H"
}

label DASolidDisplacementFoam::solvePrimal()
{
    /*
    Description:
        Call the primal solver to get converged state variables
    */

#include "createRefsSolidDisplacement.H"

    Info << "\nCalculating displacement field\n"
         << endl;

    while (this->loop(runTime)) // using simple.loop() will have seg fault in parallel
    {

        if (printToScreen_)
        {
            Info << "Iteration = " << runTime.value() << nl << endl;
        }

        gradD = fvc::grad(D);
        volSymmTensorField sigmaD = mu * twoSymm(gradD) + (lambda * I) * tr(gradD);

        volVectorField divSigmaExp = fvc::div(sigmaD - (2 * mu + lambda) * gradD, "div(sigmaD)");

        fvVectorMatrix DEqn(
            fvm::d2dt2(D)
            == fvm::laplacian(2 * mu + lambda, D, "laplacian(DD,D)")
                + divSigmaExp);

        // get the solver performance info such as initial
        // and final residuals
        SolverPerformance<vector> solverD = DEqn.solve();

        DAUtility::primalResidualControl(solverD, printToScreen_, "D", daGlobalVarPtr_->primalMaxRes);

        // calculate all functions
        this->calcAllFunctions(printToScreen_);
        // print run time
        this->printElapsedTime(runTime, printToScreen_);

        runTime.write();
    }

#include "calculateStressSolidDisplacement.H"

    // write the mesh to files
    mesh.write();

    Info << "End\n"
         << endl;

    return this->checkPrimalFailure();
}

} // End namespace Foam

// ************************************************************************* //
