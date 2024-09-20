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
#include "createAdjointSolid.H"
    // initialize checkMesh
    daCheckMeshPtr_.reset(new DACheckMesh(daOptionPtr_(), runTime, mesh));

    daLinearEqnPtr_.reset(new DALinearEqn(mesh, daOptionPtr_()));

    this->setDAObjFuncList();
}

label DASolidDisplacementFoam::solvePrimal(
    const Vec xvVec,
    Vec wVec)
{
    /*
    Description:
        Call the primal solver to get converged state variables

    Input:
        xvVec: a vector that contains all volume mesh coordinates

    Output:
        wVec: state variable vector
    */

#include "createRefsSolidDisplacement.H"

    // change the run status
    daOptionPtr_->setOption<word>("runStatus", "solvePrimal");

    Info << "\nCalculating displacement field\n"
         << endl;

    // deform the mesh based on the xvVec
    this->pointVec2OFMesh(xvVec);

    // check mesh quality
    label meshOK = this->checkMesh();

    if (!meshOK)
    {
        this->writeFailedMesh();
        return 1;
    }

    label printInterval = daOptionPtr_->getOption<label>("printInterval");
    label printToScreen = 0;
    while (this->loop(runTime)) // using simple.loop() will have seg fault in parallel
    {
        DAUtility::primalMaxInitRes_ = -1e16;

        printToScreen = this->isPrintTime(runTime, printInterval);

        if (printToScreen)
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

        DAUtility::primalResidualControl(solverD, printToScreen, "D");

        if (this->validateStates())
        {
            // write data to files and quit
            runTime.writeNow();
            mesh.write();
            return 1;
        }

        if (printToScreen)
        {
            this->printAllObjFuncs();
            Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
                 << "  ClockTime = " << runTime.elapsedClockTime() << " s"
                 << nl << endl;
        }

        runTime.write();
    }

#include "calculateStressSolidDisplacement.H"

    this->calcPrimalResidualStatistics("print");

    // primal converged, assign the OpenFoam fields to the state vec wVec
    this->ofField2StateVec(wVec);

    // write the mesh to files
    mesh.write();

    Info << "End\n"
         << endl;

    return this->checkResidualTol();
}

} // End namespace Foam

// ************************************************************************* //
