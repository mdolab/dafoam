/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

    Description:
    Adjoint solver for buoyantSimpleFoam with buoyant, turbulent 
    flow of compressible fluids.

\*---------------------------------------------------------------------------*/

static char help[] = "Solves a linear system in parallel with KSP in OpenFOAM.\n\n";

#include <petscksp.h>
#include "fvCFD.H"
#include "fluidThermo.H"
#include "turbulentFluidThermoModel.H"
#include "radiationModel.H"
#include "simpleControl.H"
#include "fvOptions.H"
#include "AdjointIO.H"
#include "AdjointSolverRegistry.H"
#include "AdjointRASModel.H"
#include "AdjointIndexing.H"
#include "AdjointJacobianConnectivity.H"
#include "AdjointObjectiveFunction.H"
#include "AdjointDerivative.H"
#include "nearWallDist.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    #include "postProcess.H"

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControl.H"

    // Initialize the petsc solver. This needs to be called after the case
    // setup so that petsc uses the OpenFOAM MPI_COMM
    PetscInitialize(&argc,&argv,(char*)0,help);

    #include "createFields.H"
    #include "createFieldRefs.H"
    #include "createFvOptions.H"
    #include "initContinuityErrs.H"

    turbulence->validate();

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    while (simple.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        // Pressure-velocity SIMPLE corrector
        {
            #include "UEqn.H"
            #include "EEqn.H"
            #include "pEqn.H"
        }

        turbulence->correct();

        adjObj.printObjFuncValues();

        runTime.write();

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    Info<< "End\n" << endl;

    adjObj.writeObjFuncValues();

    // solve adjoint
    autoPtr<AdjointDerivative> adjDev
    (
        AdjointDerivative::New(mesh,adjIO,adjReg(),adjRAS(),adjIdx,adjCon(),adjObj,thermo)
    );
    adjDev->calcFlowResidualStatistics("print");
    if (adjIO.solveAdjoint)
    {
        adjDev->solve();
    }

    PetscFinalize();
    Info<<"Petsc Finalized"<<endl;

    return 0;
}


// ************************************************************************* //
