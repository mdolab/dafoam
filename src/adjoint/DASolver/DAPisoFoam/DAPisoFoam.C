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
      turbulencePtr_(nullptr),
      daTurbulenceModelPtr_(nullptr),
      daIntmdVarPtr_(nullptr),
      daFvSourcePtr_(nullptr),
      fvSourcePtr_(nullptr)
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
    daCheckMeshPtr_.reset(new DACheckMesh(daOptionPtr_(), runTime, mesh));

    daLinearEqnPtr_.reset(new DALinearEqn(mesh, daOptionPtr_()));

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

    stateAllInstances_.setSize(nTimeInstances_);
    stateBounaryAllInstances_.setSize(nTimeInstances_);
    objFuncsAllInstances_.setSize(nTimeInstances_);
    runTimeAllInstances_.setSize(nTimeInstances_);
    runTimeIndexAllInstances_.setSize(nTimeInstances_);

    forAll(stateAllInstances_, idxI)
    {
        stateAllInstances_[idxI].setSize(daIndexPtr_->nLocalAdjointStates);
        stateBounaryAllInstances_[idxI].setSize(daIndexPtr_->nLocalAdjointBoundaryStates);
        runTimeAllInstances_[idxI] = 0.0;
        runTimeIndexAllInstances_[idxI] = 0.0;
    }

    // initialize fvSource and the source term
    const dictionary& allOptions = daOptionPtr_->getAllOptions();
    if (allOptions.subDict("fvSource").toc().size() != 0)
    {
        hasFvSource_ = 1;
        Info << "Computing fvSource" << endl;
        word sourceName = allOptions.subDict("fvSource").toc()[0];
        word fvSourceType = allOptions.subDict("fvSource").subDict(sourceName).getWord("type");
        daFvSourcePtr_.reset(DAFvSource::New(
            fvSourceType, mesh, daOptionPtr_(), daModelPtr_(), daIndexPtr_()));
        daFvSourcePtr_->calcFvSource(fvSource);
    }

    // initialize intermediate variable pointer for mean field calculation
    daIntmdVarPtr_.reset(new DAIntmdVar(mesh, daOptionPtr_()));

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

    // call correctNut, this is equivalent to turbulence->validate();
    daTurbulenceModelPtr_->updateIntermediateVariables();

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

    // We need to set the mesh moving to false, otherwise we will get V0 not found error.
    // Need to dig into this issue later
    mesh.moving(false);

    primalMinRes_ = 1e10;
    label printInterval = daOptionPtr_->getOption<label>("printIntervalUnsteady");
    label printToScreen = 0;
    label timeInstanceI = 0;
    while (this->loop(runTime)) // using simple.loop() will have seg fault in parallel
    {
        printToScreen = this->isPrintTime(runTime, printInterval);

        if (printToScreen)
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

        if (printToScreen)
        {
#include "CourantNo.H"

            daTurbulenceModelPtr_->printYPlus();

            this->printAllObjFuncs();

            Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
                 << "  ClockTime = " << runTime.elapsedClockTime() << " s"
                 << nl << endl;
        }

        daIntmdVarPtr_->update();

        runTime.write();

        this->saveTimeInstanceField(timeInstanceI);
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

void DAPisoFoam::saveTimeInstanceField(label& timeInstanceI)
{
    /*
    Description:
        Save primal variable to time instance list for unsteady adjoint
        Here we save the last nTimeInstances snapshots
    */

    scalar endTime = runTimePtr_->endTime().value();
    scalar t = runTimePtr_->timeOutputValue();
    scalar instanceStart =
        endTime - periodicity_ / nTimeInstances_ * (nTimeInstances_ - 1 - timeInstanceI);

    // the 2nd condition is for t=9.999999999999 scenario)
    if (t > instanceStart || fabs(t - endTime) < 1e-8)
    {
        Info << "Saving time instance " << timeInstanceI << " at Time = " << t << endl;

        // save fields
        daFieldPtr_->ofField2List(
            stateAllInstances_[timeInstanceI],
            stateBounaryAllInstances_[timeInstanceI]);

        // save objective functions
        forAll(daOptionPtr_->getAllOptions().subDict("objFunc").toc(), idxI)
        {
            word objFuncName = daOptionPtr_->getAllOptions().subDict("objFunc").toc()[idxI];
            scalar objFuncVal = this->getObjFuncValue(objFuncName);
            objFuncsAllInstances_[timeInstanceI].set(objFuncName, objFuncVal);
        }

        // save runTime
        runTimeAllInstances_[timeInstanceI] = t;
        runTimeIndexAllInstances_[timeInstanceI] = runTimePtr_->timeIndex();

        if (daOptionPtr_->getOption<label>("debug"))
        {
            this->calcPrimalResidualStatistics("print");
        }

        timeInstanceI++;
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

    Info << "Setting fields for time instance " << instanceI << endl;

    // set fields
    daFieldPtr_->list2OFField(
        stateAllInstances_[instanceI],
        stateBounaryAllInstances_[instanceI]);

    // set run time
    runTimePtr_->setTime(runTimeAllInstances_[instanceI], runTimeIndexAllInstances_[instanceI]);

    // We need to call correctBC multiple times to reproduce
    // the exact residual for mulitpoint, this is needed for some boundary conditions
    // and intermediate variables (e.g., U for inletOutlet, nut with wall functions)
    for (label i = 0; i < 10; i++)
    {
        daResidualPtr_->correctBoundaryConditions();
        daResidualPtr_->updateIntermediateVariables();
        daModelPtr_->correctBoundaryConditions();
        daModelPtr_->updateIntermediateVariables();
    }
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
