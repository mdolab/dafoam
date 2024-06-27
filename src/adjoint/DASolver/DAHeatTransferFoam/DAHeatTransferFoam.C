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
      kPtr_(nullptr),
      daFvSourcePtr_(nullptr)
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

    // initialize fvSource and compute the source term
    const dictionary& allOptions = daOptionPtr_->getAllOptions();
    if (allOptions.subDict("fvSource").toc().size() != 0)
    {
        hasFvSource_ = 1;
        Info << "Initializing DASource" << endl;
        word sourceName = allOptions.subDict("fvSource").toc()[0];
        word fvSourceType = allOptions.subDict("fvSource").subDict(sourceName).getWord("type");
        daFvSourcePtr_.reset(DAFvSource::New(
            fvSourceType, mesh, daOptionPtr_(), daModelPtr_(), daIndexPtr_()));
    }
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

    label printInterval = daOptionPtr_->getOption<label>("printInterval");
    label printToScreen = 0;
    // main loop
    while (this->loop(runTime)) // using simple.loop() will have seg fault in parallel
    {
        DAUtility::primalMaxInitRes_ = -1e16;

        printToScreen = this->isPrintTime(runTime, printInterval);

        if (printToScreen)
        {
            Info << "Time = " << runTime.timeName() << nl << endl;
        }

        if (hasFvSource_)
        {
            volScalarField& fvSource = fvSourcePtr_();
            daFvSourcePtr_->calcFvSource(fvSource);
        }

        fvScalarMatrix TEqn(
            fvm::laplacian(k, T)
            + fvSource);

        // get the solver performance info such as initial
        // and final residuals
        SolverPerformance<scalar> solverT = TEqn.solve();
        DAUtility::primalResidualControl(solverT, printToScreen, "T");

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
