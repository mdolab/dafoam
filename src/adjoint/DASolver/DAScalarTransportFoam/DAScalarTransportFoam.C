/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    This class is modified from OpenFOAM's source code
    applications/solvers/basic/scalarTransportFoam

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

#include "DAScalarTransportFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAScalarTransportFoam, 0);
addToRunTimeSelectionTable(DASolver, DAScalarTransportFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAScalarTransportFoam::DAScalarTransportFoam(
    char* argsAll,
    PyObject* pyOptions)
    : DASolver(argsAll, pyOptions),
      TPtr_(nullptr),
      UPtr_(nullptr),
      phiPtr_(nullptr),
      DTPtr_(nullptr)
{
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DAScalarTransportFoam::initSolver()
{
    /*
    Description:
        Initialize variables for DASolver
    */

    Info << "Initializing fields for DAScalarTransportFoam" << endl;
    Time& runTime = runTimePtr_();
    fvMesh& mesh = meshPtr_();
#include "createFieldsScalarTransport.H"
#include "createAdjoint.H"
}

label DAScalarTransportFoam::solvePrimal()
{
    /*
    Description:
        Call the primal solver to get converged state variables
    */

#include "createRefsScalarTransport.H"

    Info << "\nCalculating temperature distribution\n"
         << endl;

    scalar endTime = runTime.endTime().value();
    scalar deltaT = runTime.deltaT().value();
    label nInstances = round(endTime / deltaT);

    // main loop
    for (label iter = 1; iter <= nInstances; iter++)
    {
        ++runTime;
        
        printToScreen_ = this->isPrintTime(runTime, printIntervalUnsteady_);

        if (printToScreen_)
        {
            Info << "Time = " << runTime.timeName() << nl << endl;
        }

        fvScalarMatrix TEqn(
            fvm::ddt(T)
            + fvm::div(phi, T)
            - fvm::laplacian(DT, T));

        TEqn.relax();

        // get the solver performance info such as initial
        // and final residuals
        SolverPerformance<scalar> solverT = TEqn.solve();
        DAUtility::primalResidualControl(solverT, printToScreen_, "T", daGlobalVarPtr_->primalMaxRes);

        this->calcAllFunctions(printToScreen_);
        this->printElapsedTime(runTime, printToScreen_);

        runTime.write();
    }

    // need to save primalFinalTimeIndex_.
    primalFinalTimeIndex_ = runTime.timeIndex();

    // write the mesh to files
    mesh.write();

    Info << "End\n"
         << endl;

    return 0;
}

} // End namespace Foam

// ************************************************************************* //
