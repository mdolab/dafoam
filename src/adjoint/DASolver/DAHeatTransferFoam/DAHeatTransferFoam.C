/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

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
#include "createAdjoint.H"

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

label DAHeatTransferFoam::solvePrimal()
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

    Info << "\nCalculating temperature distribution\n"
         << endl;

    // main loop
    while (this->loop(runTime)) // using simple.loop() will have seg fault in parallel
    {

        if (printToScreen_)
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
        DAUtility::primalResidualControl(solverT, printToScreen_, "T", daGlobalVarPtr_->primalMaxRes);

        this->calcAllFunctions(printToScreen_);

        // print run time
        this->printElapsedTime(runTime, printToScreen_);

        runTime.write();
    }

    // write the mesh to files
    mesh.write();

    Info << "End\n"
         << endl;

    return 0;
}

} // End namespace Foam

// ************************************************************************* //
