/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    This class is modified from OpenFOAM's source code
    applications/solvers/compressible/rhoSimpleFoam

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

#include "DAHisaFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAHisaFoam, 0);
addToRunTimeSelectionTable(DASolver, DAHisaFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAHisaFoam::DAHisaFoam(
    char* argsAll,
    PyObject* pyOptions)
    : DASolver(argsAll, pyOptions),
      hisaSolverPtr_(nullptr)
{
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DAHisaFoam::initSolver()
{
    /*
    Description:
        Initialize variables for DASolver
    */

    Info << "Initializing fields for DAHisaFoam" << endl;
    Time& runTime = runTimePtr_();
    fvMesh& mesh = meshPtr_();

    dictionary dict;
    dict.add("solver", "hisa");
    hisaSolverPtr_.reset(
        solverModule::New(Foam::polyMesh::defaultRegion, runTime, mesh, dict));

    solverModule& solver = hisaSolverPtr_();

    solver.initialise();

    // read the RAS model from constant/turbulenceProperties
    const word turbModelName(
        IOdictionary(
            IOobject(
                "turbulenceProperties",
                mesh.time().constant(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE,
                false))
            .subDict("RAS")
            .lookup("RASModel"));
    daTurbulenceModelPtr_.reset(DATurbulenceModel::New(turbModelName, mesh, daOptionPtr_()));

#include "createAdjoint.H"
}

label DAHisaFoam::solvePrimal()
{
    /*
    Description:
        Call the primal solver to get converged state variables
    */

    Info << "Running HiSA " << endl;

    solverModule& solver = hisaSolverPtr_();
    Time& runTime = runTimePtr_();
    runTime.setTime(0.0, 0);

    bool steadyState = solver.steadyState();

    if (!steadyState)
    {
#include "createTimeControls.H"

        scalar scale = solver.timeStepScaling(maxCo);
        //Generate a pretend Courant number so we can use setInitialDeltaT.H unmodified
        scalar CoNum = maxCo / scale;
#include "setInitialDeltaT.H"
    }

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    if (steadyState)
    {
        Info << "\nStarting pseudotime iteration loop\n"
             << endl;
    }
    else
    {
        Info << "\nStarting time loop\n"
             << endl;
    }

    //scalar prevCpuTime = runTime.elapsedCpuTime();
    while (runTime.run())
    {
        if (!steadyState)
        {
#include "createTimeControls.H"

            //Find out what to scale deltaT by
            scalar scale = solver.timeStepScaling(maxCo);
            //Generate a pretend Courant number so we can use setDeltaT.H unmodified
            scalar CoNum = maxCo / scale;
#include "setDeltaT.H"
        }

        solver.preTimeIncrement();

        runTime++;

        printToScreen_ = this->isPrintTime(runTime, printInterval_);

        if (printToScreen_)
        {
            Info << "Time = " << runTime.timeName() << nl << endl;
            this->calcPrimalResidualStatistics("print");
        }

        solver.beginTimeStep();

        // --- Outer corrector loop
        bool notFinished;
#if FOUNDATION >= 6
        while (true)
        {
            // loop() function was removed from the base class
            if (isA<pimpleControl>(solver.solnControl()))
            {
                notFinished = refCast<pimpleControl>(solver.solnControl()).loop();
            }
            else
            {
                notFinished = refCast<pseudotimeControl>(solver.solnControl()).loop();
            }
            if (!notFinished)
            {
                break;
            }
#else
        while ((notFinished = solver.solnControl().loop()))
        {
#endif
            solver.outerIteration();
            if (steadyState)
            {
                break;
            }
        }

        // calculate all functions
        this->calcAllFunctions(printToScreen_);
        // calculate yPlus
        daTurbulenceModelPtr_->printYPlus(printToScreen_);

        // print run time
        this->printElapsedTime(runTime, printToScreen_);

        if (steadyState && !notFinished)
        {
            runTime.writeNow();
            break;
        }
        else
        {
            runTime.write();
        }
    }

    // write the mesh to files
    meshPtr_->write();

    primalFinalTimeIndex_ = runTime.timeIndex();

    Info << "End\n"
         << endl;

    return 0;
}

} // End namespace Foam

// ************************************************************************* //
