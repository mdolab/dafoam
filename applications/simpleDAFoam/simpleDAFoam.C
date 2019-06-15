/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Adjoint solver for simpleFoam

\*---------------------------------------------------------------------------*/

static char help[] = "Solves a linear system in parallel with KSP in OpenFOAM.\n\n";

#include <petscksp.h>
#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "simpleControl.H"
#include "fvOptions.H"
#include "AdjointIO.H"
#include "AdjointSolverRegistry.H"
#include "AdjointRASModel.H"
#include "AdjointIndexing.H"
#include "AdjointJacobianConnectivity.H"
#include "AdjointObjectiveFunction.H"
#include "AdjointDerivative.H"
#include "AdjointNewtonKrylov.H"
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
    #include "createFvOptions.H"
    #include "initContinuityErrs.H"

    turbulence->validate();

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    while (simple.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        // --- Pressure-velocity SIMPLE corrector
        {
            #include "UEqn.H"
            #include "pEqn.H"
        }

        laminarTransport.correct();
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
        AdjointDerivative::New(mesh,adjIO,adjReg(),adjRAS(),adjIdx,adjCon(),adjObj)
    );

    adjDev->calcFlowResidualStatistics("print");

    if(adjIO.useNKSolver and !adjIO.solveAdjoint)
    {

        AdjointNewtonKrylov adjNK(mesh,adjIO,adjReg,adjRAS(),adjObj,adjDev());
        adjNK.solve();

        adjDev->calcFlowResidualStatistics("print");

        adjDev->writeStates();
        adjRAS->writeTurbStates();

        Info<< "NK Finished!\n" << endl;
        
        return 0;
    }

    if (adjIO.solveAdjoint)
    {
        adjDev->solve();
    }

    PetscFinalize();
 
    return 0;
}


// ************************************************************************* //
