/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.1

\*---------------------------------------------------------------------------*/
#include "SolidDisplacementDAFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Constructors
SolidDisplacementDAFoam::SolidDisplacementDAFoam(char* argsAll)
{
    argsAll_ = argsAll;
}

SolidDisplacementDAFoam::~SolidDisplacementDAFoam()
{
}

void SolidDisplacementDAFoam::run()
{
    #include "setArgs.H"
    #include "setRootCasePython.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControls.H"
    #include "createFields.H"
    #include "createAdjoint.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nCalculating displacement field\n" << endl;

    while (runTime.loop())
    {
        Info<< "Iteration: " << runTime.value() << nl << endl;

        #include "readSolidDisplacementFoamControls.H"

        int iCorr = 0;
        scalar initialResidual = 0;

        do
        {
            if (thermalStress)
            {
                volScalarField& T = Tptr();
                solve
                (
                    fvm::ddt(T) == fvm::laplacian(DT, T)
                );
            }

            {
                fvVectorMatrix DEqn
                (
                    fvm::d2dt2(D)
                 ==
                    fvm::laplacian(2*mu + lambda, D, "laplacian(DD,D)")
                  + divSigmaExp
                  + centrifugalForce
                );

                if (thermalStress)
                {
                    const volScalarField& T = Tptr();
                    DEqn += fvc::grad(threeKalpha*T);
                }

                //DEqn.setComponentReference(1, 0, vector::X, 0);
                //DEqn.setComponentReference(1, 0, vector::Z, 0);

                initialResidual = DEqn.solve().max().initialResidual();

                if (!compactNormalStress)
                {
                    divSigmaExp = fvc::div(DEqn.flux());
                }
            }

            {
                //volTensorField gradD(fvc::grad(D));
                gradD = fvc::grad(D);
                sigmaD = mu*twoSymm(gradD) + (lambda*I)*tr(gradD);

                if (compactNormalStress)
                {
                    divSigmaExp = fvc::div
                    (
                        sigmaD - (2*mu + lambda)*gradD,
                        "div(sigmaD)"
                    );
                }
                else
                {
                    divSigmaExp += fvc::div(sigmaD);
                }
            }

            adjObj.printObjFuncValues();

        } while (initialResidual > convergenceTolerance && ++iCorr < nCorr);

        #include "calculateStress.H"

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
