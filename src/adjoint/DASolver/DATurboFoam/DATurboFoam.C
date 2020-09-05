/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DATurboFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DATurboFoam, 0);
addToRunTimeSelectionTable(DASolver, DATurboFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DATurboFoam::DATurboFoam(
    char* argsAll,
    PyObject* pyOptions)
    : DASolver(argsAll, pyOptions),
      simplePtr_(nullptr),
      pThermoPtr_(nullptr),
      pPtr_(nullptr),
      rhoPtr_(nullptr),
      UPtr_(nullptr),
      URelPtr_(nullptr),
      phiPtr_(nullptr),
      pressureControlPtr_(nullptr),
      turbulencePtr_(nullptr),
      daTurbulenceModelPtr_(nullptr),
      MRFPtr_(nullptr),
      initialMass_(dimensionedScalar("initialMass", dimensionSet(1, 0, 0, 0, 0, 0, 0), 0.0))
{
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DATurboFoam::initSolver()
{
    /*
    Description:
        Initialize variables for DASolver
    */

    Info << "Initializing fields for DATurboFoam" << endl;
    Time& runTime = runTimePtr_();
    fvMesh& mesh = meshPtr_();
    argList& args = argsPtr_();
#include "createSimpleControlPython.H"
#include "createFieldsTurbo.H"
#include "createAdjointCompressible.H"
    // initialize checkMesh
    daCheckMeshPtr_.reset(new DACheckMesh(runTime, mesh));
}

label DATurboFoam::solvePrimal(
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

#include "createRefsTurbo.H"

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

    // set the rotating wall velocity after the mesh is updated
    this->setRotingWallVelocity();

    label nSolverIters = 1;
    primalMinRes_ = 1e10;
    label printInterval = daOptionPtr_->getOption<label>("printInterval");
    while (this->loop(runTime)) // using simple.loop() will have seg fault in parallel
    {
        if (nSolverIters % printInterval == 0 || nSolverIters == 1)
        {
            Info << "Time = " << runTime.timeName() << nl << endl;
        }

        p.storePrevIter();
        rho.storePrevIter();

        // Pressure-velocity SIMPLE corrector
#include "UEqnTurbo.H"
#include "pEqnTurbo.H"
#include "EEqnTurbo.H"

        daTurbulenceModelPtr_->correct();

        if (nSolverIters % printInterval == 0 || nSolverIters == 1)
        {
            daTurbulenceModelPtr_->printYPlus();

            this->printAllObjFuncs();

            Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
                 << "  ClockTime = " << runTime.elapsedClockTime() << " s"
                 << nl << endl;
        }

        runTime.write();

        nSolverIters++;
    }

    this->calcPrimalResidualStatistics("print");

    // primal converged, assign the OpenFoam fields to the state vec wVec
    this->ofField2StateVec(wVec);

    // write the mesh to files
    mesh.write();

    Info << "End\n"
         << endl;

    return this->checkResidualTol();
}

void DATurboFoam::setRotingWallVelocity()
{
    /*
    Description:
        Set velocity boundary condition for rotating walls
        This function should be called once for each primal solution.
        It should be called AFTER the mesh points are updated
    */

    volVectorField& U = const_cast<volVectorField&>(
        meshPtr_->thisDb().lookupObject<volVectorField>("U"));

    IOdictionary MRFProperties(
        IOobject(
            "MRFProperties",
            runTimePtr_->constant(),
            meshPtr_(),
            IOobject::MUST_READ,
            IOobject::NO_WRITE));

    wordList nonRotatingPatches;
    MRFProperties.subDict("MRF").readEntry<wordList>("nonRotatingPatches", nonRotatingPatches);

    vector origin;
    MRFProperties.subDict("MRF").readEntry<vector>("origin", origin);
    vector axis;
    MRFProperties.subDict("MRF").readEntry<vector>("axis", axis);
    scalar omega = MRFProperties.subDict("MRF").getScalar("omega");

    forAll(meshPtr_->boundaryMesh(), patchI)
    {
        word bcName = meshPtr_->boundaryMesh()[patchI].name();
        if (!DAUtility::isInList<word>(bcName, nonRotatingPatches))
        {
            Info << "Setting rotating wall velocity for " << bcName << endl;
            if (U.boundaryField()[patchI].size() > 0)
            {
                forAll(U.boundaryField()[patchI], faceI)
                {
                    vector patchCf = meshPtr_->Cf().boundaryField()[patchI][faceI];
                    U.boundaryFieldRef()[patchI][faceI] =
                        -omega * ((patchCf - origin) ^ (axis / mag(axis)));
                }
            }
        }
    }
}

} // End namespace Foam

// ************************************************************************* //
