/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

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
#include "createAdjointSolid.H"
    // initialize checkMesh
    daCheckMeshPtr_.reset(new DACheckMesh(daOptionPtr_(), runTime, mesh));

    daLinearEqnPtr_.reset(new DALinearEqn(mesh, daOptionPtr_()));

    this->setDAObjFuncList();

    mode_ = daOptionPtr_->getSubDictOption<word>("unsteadyAdjoint", "mode");

    if (mode_ == "hybridAdjoint")
    {

        nTimeInstances_ =
            daOptionPtr_->getSubDictOption<label>("unsteadyAdjoint", "nTimeInstances");

        periodicity_ =
            daOptionPtr_->getSubDictOption<scalar>("unsteadyAdjoint", "periodicity");

        if (periodicity_ <= 0)
        {
            FatalErrorIn("") << "periodicity <= 0!" << abort(FatalError);
        }
    }
    else if (mode_ == "timeAccurateAdjoint")
    {

        nTimeInstances_ =
            daOptionPtr_->getSubDictOption<label>("unsteadyAdjoint", "nTimeInstances");

        scalar endTime = runTimePtr_->endTime().value();
        scalar deltaT = runTimePtr_->deltaTValue();
        label maxNTimeInstances = round(endTime / deltaT) + 1;
        if (nTimeInstances_ != maxNTimeInstances)
        {
            FatalErrorIn("") << "nTimeInstances in timeAccurateAdjoint is not equal to "
                             << "the maximal possible value!" << abort(FatalError);
        }
    }

    if (mode_ == "hybridAdjoint" || mode_ == "timeAccurateAdjoint")
    {

        if (nTimeInstances_ <= 0)
        {
            FatalErrorIn("") << "nTimeInstances <= 0!" << abort(FatalError);
        }

        stateAllInstances_.setSize(nTimeInstances_);
        stateBoundaryAllInstances_.setSize(nTimeInstances_);
        objFuncsAllInstances_.setSize(nTimeInstances_);
        runTimeAllInstances_.setSize(nTimeInstances_);
        runTimeIndexAllInstances_.setSize(nTimeInstances_);

        forAll(stateAllInstances_, idxI)
        {
            stateAllInstances_[idxI].setSize(daIndexPtr_->nLocalAdjointStates);
            stateBoundaryAllInstances_[idxI].setSize(daIndexPtr_->nLocalAdjointBoundaryStates);
            runTimeAllInstances_[idxI] = 0.0;
            runTimeIndexAllInstances_[idxI] = 0;
        }
    }
}

label DAScalarTransportFoam::solvePrimal(
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

#include "createRefsScalarTransport.H"

    // change the run status
    daOptionPtr_->setOption<word>("runStatus", "solvePrimal");

    // first check if we need to change the boundary conditions based on
    // the primalBC dict in DAOption. NOTE: this will overwrite whatever
    // boundary conditions defined in the "0" folder
    dictionary bcDict = daOptionPtr_->getAllOptions().subDict("primalBC");
    if (bcDict.toc().size() != 0)
    {
        Info << "Setting up primal boundary conditions based on pyOptions: " << endl;
        daFieldPtr_->setPrimalBoundaryConditions();
    }

    Info << "\nCalculating temperature distribution\n"
         << endl;

    // deform the mesh based on the xvVec
    this->pointVec2OFMesh(xvVec);

    // check mesh quality
    label meshOK = this->checkMesh();

    if (!meshOK)
    {
        return 1;
    }

    // We need to set the mesh moving to false, otherwise we will get V0 not found error.
    // Need to dig into this issue later
    // NOTE: we have commented this out. Setting mesh.moving(false) has been done
    // right after mesh.movePoints() calls.
    //mesh.moving(false);

    primalMinRes_ = 1e10;
    label printInterval = daOptionPtr_->getOption<label>("printIntervalUnsteady");
    label printToScreen = 0;
    label timeInstanceI = 0;
    // for time accurate adjoints, we need to save states for Time = 0
    if (mode_ == "timeAccurateAdjoint")
    {
        this->saveTimeInstanceFieldTimeAccurate(timeInstanceI);
    }
    // main loop
    while (this->loop(runTime)) // using simple.loop() will have seg fault in parallel
    {
        printToScreen = this->isPrintTime(runTime, printInterval);

        if (printToScreen)
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
        this->primalResidualControl<scalar>(solverT, printToScreen, printInterval, "T");

        if (printToScreen)
        {

            this->printAllObjFuncs();

            if (daOptionPtr_->getOption<label>("debug"))
            {
                this->calcPrimalResidualStatistics("print");
            }

            Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
                 << "  ClockTime = " << runTime.elapsedClockTime() << " s"
                 << nl << endl;
        }

        runTime.write();

        if (mode_ == "hybridAdjoint")
        {
            this->saveTimeInstanceFieldHybrid(timeInstanceI);
        }

        if (mode_ == "timeAccurateAdjoint")
        {
            this->saveTimeInstanceFieldTimeAccurate(timeInstanceI);
        }
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
