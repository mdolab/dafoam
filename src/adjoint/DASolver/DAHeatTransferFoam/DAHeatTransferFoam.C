/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAHeatTransferFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAHeatTransferFoam, 0);
addToRunTimeSelectionTable(DASolver, DAHeatTransferFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAHeatTransferFoam::DAHeatTransferFoam(
    char* argsAll,
    PyObject* pyOptions)
    : DASolver(argsAll, pyOptions),
      TPtr_(nullptr),
      fvSourcePtr_(nullptr),
      kPtr_(nullptr)
{
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DAHeatTransferFoam::initSolver()
{
    /*
    Description:
        Initialize variables for DASolver
    */

    Info << "Initializing fields for DAHeatTransferFoam" << endl;
    Time& runTime = runTimePtr_();
    fvMesh& mesh = meshPtr_();
#include "createFieldsHeatTransfer.H"
#include "createAdjointSolid.H"
    // initialize checkMesh
    daCheckMeshPtr_.reset(new DACheckMesh(daOptionPtr_(), runTime, mesh));

    daLinearEqnPtr_.reset(new DALinearEqn(mesh, daOptionPtr_()));

    this->setDAObjFuncList();
}

label DAHeatTransferFoam::solvePrimal(
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

#include "createRefsHeatTransfer.H"

    // change the run status
    daOptionPtr_->setOption<word>("runStatus", "solvePrimal");

    Info << "\nCalculating temperature distribution\n"
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

    primalMinRes_ = 1e10;
    label printInterval = daOptionPtr_->getOption<label>("printIntervalUnsteady");
    label printToScreen = 0;
    // main loop
    while (this->loop(runTime)) // using simple.loop() will have seg fault in parallel
    {
        printToScreen = this->isPrintTime(runTime, printInterval);

        if (printToScreen)
        {
            Info << "Time = " << runTime.timeName() << nl << endl;
        }

        fvScalarMatrix TEqn(
            fvm::laplacian(k, T)
            + fvSource);

        // get the solver performance info such as initial
        // and final residuals
        SolverPerformance<scalar> solverT = TEqn.solve();
        this->primalResidualControl<scalar>(solverT, printToScreen, printInterval, "T");

        if (printToScreen)
        {

            this->printAllObjFuncs();

            Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
                 << "  ClockTime = " << runTime.elapsedClockTime() << " s"
                 << nl << endl;
        }

        runTime.write();

    }

    this->calcPrimalResidualStatistics("print");

    // primal converged, assign the OpenFoam fields to the state vec wVec
    this->ofField2StateVec(wVec);

    // write the mesh to files
    mesh.write();

    Info << "End\n"
         << endl;

    return 0;
}

} // End namespace Foam

// ************************************************************************* //
