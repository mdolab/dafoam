/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.1

\*---------------------------------------------------------------------------*/
#include "RhoSimpleDAFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Constructors
RhoSimpleDAFoam::RhoSimpleDAFoam(char* argsAll)
{
    argsAll_ = argsAll;
}

RhoSimpleDAFoam::~RhoSimpleDAFoam()
{
}

void RhoSimpleDAFoam::run()
{

    #include "setArgs.H"
    #include "setRootCasePython.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControl.H"
    #include "createFields.H"
    #include "createAdjoint.H"
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
        #include "UEqn.H"
        #include "EEqn.H"
        #include "pEqn.H"

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

    if(adjIO.useNKSolver and !adjIO.solveAdjoint)
    {

        AdjointNewtonKrylov adjNK(mesh,adjIO,adjReg,adjRAS(),adjObj,adjDev());
        adjNK.solve();

        adjDev->calcFlowResidualStatistics("print");

        adjDev->writeStates();
        adjRAS->writeTurbStates();

        Info<< "NK Finished!\n" << endl;
        
        return;
    }

    if (adjIO.solveAdjoint)
    {
        adjDev->solve();
    }

    return;

}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
