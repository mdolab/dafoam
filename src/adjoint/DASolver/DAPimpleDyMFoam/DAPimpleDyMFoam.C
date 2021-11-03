/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAPimpleDyMFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAPimpleDyMFoam, 0);
addToRunTimeSelectionTable(DASolver, DAPimpleDyMFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAPimpleDyMFoam::DAPimpleDyMFoam(
    char* argsAll,
    PyObject* pyOptions)
    : DASolver(argsAll, pyOptions),
      daMotionPtr_(nullptr)
{
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DAPimpleDyMFoam::initSolver()
{
    /*
    Description:
        Initialize variables for DASolver
    */
    daOptionPtr_.reset(new DAOption(meshPtr_(), pyOptions_));
}

label DAPimpleDyMFoam::solvePrimal(
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
for(int i=0;i<argc;i++)
Info<<argv[i]<<endl;

//#include "setArgs.H"
#include "setRootCase.H"
#include "createTime.H"
#include "createDynamicFvMesh.H"
#include "initContinuityErrs.H"
#include "createDyMControls.H"
#include "createFieldsPimpleDyM.H"
#include "createUfIfPresent.H"
#include "CourantNo.H"
#include "setInitialDeltaT.H"

    daMotionPtr_.reset(DAMotion::New(mesh, daOptionPtr_()));

    turbulence->validate();

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info << "\nStarting time loop\n"
         << endl;

    while (runTime.run())
    {

#include "readDyMControls.H"
#include "CourantNo.H"
#include "setDeltaT.H"

        ++runTime;

        daMotionPtr_->correct();

        Info << "Time = " << runTime.timeName() << nl << endl;

        // --- Pressure-velocity PIMPLE corrector loop
        while (pimple.loop())
        {
            if (pimple.firstIter() || moveMeshOuterCorrectors)
            {
                mesh.update();

                if (mesh.changing())
                {
                    MRF.update();

                    if (correctPhi)
                    {
                        // Calculate absolute flux
                        // from the mapped surface velocity
                        phi = mesh.Sf() & Uf();

#include "correctPhiPimpleDyM.H"

                        // Make the flux relative to the mesh motion
                        fvc::makeRelative(phi, U);
                    }

                    if (checkMeshCourantNo)
                    {
#include "meshCourantNo.H"
                    }
                }
            }

#include "UEqnPimpleDyM.H"

            // --- Pressure corrector loop
            while (pimple.correct())
            {
#include "pEqnPimpleDyM.H"
            }

            if (pimple.turbCorr())
            {
                laminarTransport.correct();
                turbulence->correct();
            }
        }

        runTime.write();

        runTime.printExecutionTime(Info);
    }

    Info << "End\n"
         << endl;

    return 0;
}

} // End namespace Foam

// ************************************************************************* //
