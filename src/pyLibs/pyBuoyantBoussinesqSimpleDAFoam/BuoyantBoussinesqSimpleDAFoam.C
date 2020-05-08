/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.1

\*---------------------------------------------------------------------------*/
#include "BuoyantBoussinesqSimpleDAFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Constructors
BuoyantBoussinesqSimpleDAFoam::BuoyantBoussinesqSimpleDAFoam(char* argsAll)
{
    argsAll_ = argsAll;
}

BuoyantBoussinesqSimpleDAFoam::~BuoyantBoussinesqSimpleDAFoam()
{
}

void BuoyantBoussinesqSimpleDAFoam::run()
{
    #include "setArgs.H"
    #include "setRootCasePython.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControl.H"
    #include "createFields.H"
    #include "createAdjoint.H"
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
            #include "TEqn.H"
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
    
    if (adjIO.solveAdjoint)
    {
        adjDev->solve();
    }

    return;

}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
