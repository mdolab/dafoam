/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAPisoFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAPisoFoam, 0);
addToRunTimeSelectionTable(DASolver, DAPisoFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAPisoFoam::DAPisoFoam(
    char* argsAll,
    PyObject* pyOptions)
    : DASolver(argsAll, pyOptions),
      pisoPtr_(nullptr),
      pPtr_(nullptr),
      UPtr_(nullptr),
      phiPtr_(nullptr),
      laminarTransportPtr_(nullptr),
      turbulencePtr_(nullptr)
{
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DAPisoFoam::initSolver()
{
    /*
    Description:
        Initialize variables for DASolver
    */

    Info << "Initializing fields for DAPisoFoam" << endl;
    Time& runTime = runTimePtr_();
    fvMesh& mesh = meshPtr_();
#include "createPisoControlPython.H"
#include "createFieldsPiso.H"
#include "createAdjointIncompressible.H"
    // initialize checkMesh
    daCheckMeshPtr_.reset(new DACheckMesh(runTime, mesh));

    label active = daOptionPtr_->getSubDictOption<label>("hybridAdjoint", "active");

    nTimeInstances_ =
        daOptionPtr_->getSubDictOption<label>("hybridAdjoint", "nTimeInstances");

    periodicity_ =
        daOptionPtr_->getSubDictOption<scalar>("hybridAdjoint", "periodicity");

    if (!active)
    {
        FatalErrorIn("hybridAdjoint") << "active is False!" << abort(FatalError);
    }

    if (periodicity_ <= 0)
    {
        FatalErrorIn("hybridAdjoint") << "periodicity <= 0!" << abort(FatalError);
    }

    if (nTimeInstances_ <= 0)
    {
        FatalErrorIn("hybridAdjoint") << "nTimeInstances <= 0!" << abort(FatalError);
    }

    endTime_ = runTimePtr_->endTime().value();
    deltaT_ = runTimePtr_->deltaT().value();

    stateAllInstances_.setSize(nTimeInstances_);
    stateBounaryAllInstances_.setSize(nTimeInstances_);
    objFuncsAllInstances_.setSize(nTimeInstances_);

    forAll(stateAllInstances_, idxI)
    {
        stateAllInstances_[idxI].setSize(daIndexPtr_->nLocalAdjointStates);
        stateBounaryAllInstances_[idxI].setSize(daIndexPtr_->nLocalAdjointBoundaryStates);
    }
}

label DAPisoFoam::solvePrimal(
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

#include "createRefsPiso.H"

    // first check if we need to change the boundary conditions based on
    // the primalBC dict in DAOption. NOTE: this will overwrite whatever
    // boundary conditions defined in the "0" folder
    dictionary bcDict = daOptionPtr_->getAllOptions().subDict("primalBC");
    if (bcDict.toc().size() != 0)
    {
        Info << "Setting up primal boundary conditions based on pyOptions: " << endl;
        daFieldPtr_->setPrimalBoundaryConditions();
    }

    turbulencePtr_->validate();

    Info << "\nStarting time loop\n"
         << endl;

    // deform the mesh based on the xvVec
    this->pointVec2OFMesh(xvVec);

    // check mesh quality
    label meshOK = this->checkMesh();

    if (!meshOK)
    {
        return 1;
    }

    // create a file to store the objective values
    this->initializeObjFuncHistFilePtr("objFuncHist.txt");

    label nSolverIters = 1;
    primalMinRes_ = 1e10;
    label printInterval = daOptionPtr_->getOption<label>("printIntervalUnsteady");
    while (this->loop(runTime)) // using simple.loop() will have seg fault in parallel
    {

        if (nSolverIters % printInterval == 0 || nSolverIters == 1)
        {
            Info << "Time = " << runTime.timeName() << nl << endl;
        }

        // Pressure-velocity PISO corrector
        {
#include "UEqnPiso.H"

            // --- PISO loop
            while (piso.correct())
            {
#include "pEqnPiso.H"
            }
        }

        laminarTransport.correct();
        daTurbulenceModelPtr_->correct();

        if (nSolverIters % printInterval == 0 || nSolverIters == 1)
        {
            daTurbulenceModelPtr_->printYPlus();

            this->printAllObjFuncs();

            Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
                 << "  ClockTime = " << runTime.elapsedClockTime() << " s"
                 << nl << endl;
        }

        // we write the objective function to file at every step
        this->writeObjFuncHistFile();

        runTime.write();

        this->saveTimeInstanceField();

        nSolverIters++;
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

void DAPisoFoam::saveTimeInstanceField()
{
    /*
    Description:
        Save primal variable to time instance list for unsteady adjoint
        Here we save the last nTimeInstances snapshots
    */
    scalar t = runTimePtr_->timeOutputValue();
    scalar saveStart = endTime_ - periodicity_;
    scalar ToNT = periodicity_ / nTimeInstances_;

    if (t > saveStart)
    {
        scalar remaindTime = endTime_ - t;
        if (std::fmod(remaindTime, ToNT) < deltaT_)
        {
            Info << "Saving time instances at Time = " << t << endl;
            label instanceI = nTimeInstances_ - 1 - round(remaindTime / ToNT);
            daFieldPtr_->ofField2List(
                stateAllInstances_[instanceI],
                stateBounaryAllInstances_[instanceI]);
            
            // save objective functions
            forAll(daOptionPtr_->getAllOptions().subDict("objFunc").toc(), idxI)
            {
                word objFuncName = daOptionPtr_->getAllOptions().subDict("objFunc").toc()[idxI];
                scalar objFuncVal = this->getObjFuncValue(objFuncName);
                objFuncsAllInstances_[instanceI].set(objFuncName, objFuncVal);
            }
        }
    }
    return;
}

void DAPisoFoam::setTimeInstanceField(const label instanceI)
{
    /*
    Description:
        Assign primal variables based on the current time instance
        If unsteady adjoint solvers are used, this virtual function should be 
        implemented in a child class, otherwise, return error if called
    */
    daFieldPtr_->list2OFField(
        stateAllInstances_[instanceI],
        stateBounaryAllInstances_[instanceI]);
}

scalar DAPisoFoam::getTimeInstanceObjFunc(
    const label instanceI,
    const word objFuncName)
{
    /*
    Description:
        Return the value of objective function at the given time instance and name
    */

    return objFuncsAllInstances_[instanceI].getScalar(objFuncName);
}

} // End namespace Foam

// ************************************************************************* //
