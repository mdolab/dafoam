/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Adjoint solver for Laplace equation.

\*---------------------------------------------------------------------------*/

static char help[] = "Solves a linear system in parallel with KSP in OpenFOAM.\n\n";

#include <petscksp.h>
#include "fvCFD.H"
#include "fvOptions.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "simpleControl.H"
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
    #include "setRootCase.H"

    #include "createTime.H"
    #include "createMesh.H"

    // Initialize the petsc solver. This needs to be called after the case
    // setup so that petsc uses the OpenFOAM MPI_COMM
    PetscInitialize(&argc,&argv,(char*)0,help);

    simpleControl simple(mesh);

    #include "createFields.H"
    #include "createFvOptions.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nCalculating temperature distribution\n" << endl;

    while (simple.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        while (simple.correctNonOrthogonal())
        {
            fvScalarMatrix TEqn
            (
                fvm::ddt(T) - fvm::laplacian(DT, T)
             ==
                fvOptions(T)
            );

            fvOptions.constrain(TEqn);
            TEqn.solve();
            fvOptions.correct(T);
        }

        adjObj.printObjFuncValues();

        #include "write.H"

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

    if (adjIO.solveAdjoint)
    {
        adjDev->solve();
    }

    PetscFinalize();

    return 0;
}


// ************************************************************************* //
