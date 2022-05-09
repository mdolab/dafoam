/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAPartDerivdRdFFD.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAPartDerivdRdFFD, 0);
addToRunTimeSelectionTable(DAPartDeriv, DAPartDerivdRdFFD, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAPartDerivdRdFFD::DAPartDerivdRdFFD(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex,
    const DAJacCon& daJacCon,
    const DAResidual& daResidual)
    : DAPartDeriv(modelType,
                  mesh,
                  daOption,
                  daModel,
                  daIndex,
                  daJacCon,
                  daResidual)
{
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void DAPartDerivdRdFFD::initializePartDerivMat(
    const dictionary& options,
    Mat jacMat)
{
    /*
    Description:
        Initialize jacMat
    
    Input:
        options.nDesignVars. The number of design variable for dRdFFD
    */

    label nDesignVars = options.getLabel("nDesignVars");

    // now initialize the memory for the jacobian itself
    label localSize = daIndex_.nLocalAdjointStates;

    // create dRdFFDT
    //MatCreate(PETSC_COMM_WORLD, jacMat);
    MatSetSizes(
        jacMat,
        localSize,
        PETSC_DECIDE,
        PETSC_DETERMINE,
        nDesignVars);
    MatSetFromOptions(jacMat);
    MatMPIAIJSetPreallocation(jacMat, nDesignVars, NULL, nDesignVars, NULL);
    MatSeqAIJSetPreallocation(jacMat, nDesignVars, NULL);
    //MatSetOption(jacMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(jacMat);
    MatZeroEntries(jacMat);
    Info << "Partial derivative matrix created. " << mesh_.time().elapsedClockTime() << " s" << endl;
}

void DAPartDerivdRdFFD::calcPartDerivMat(
    const dictionary& options,
    const Vec xvVec,
    const Vec wVec,
    Mat jacMat)
{
    /*
    Description:
        Compute jacMat. We use brute-force finite-difference
    
    Input:

        options.nDesignVars. The number of design variable for dRdFFD

        options.isPC: whether to compute the jacMat for preconditioner

        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector
    
    Output:
        jacMat: the partial derivative matrix dRdFFD to compute
    */

    label nDesignVars = options.getLabel("nDesignVars");

    DAResidual& daResidual = const_cast<DAResidual&>(daResidual_);

    // zero all the matrices
    MatZeroEntries(jacMat);

    // initialize residual vectors
    Vec resVecRef, resVec;
    VecDuplicate(wVec, &resVec);
    VecDuplicate(wVec, &resVecRef);
    VecZeroEntries(resVec);
    VecZeroEntries(resVecRef);

    dictionary mOptions;
    mOptions.set("updateState", 1);
    mOptions.set("updateMesh", 1);
    mOptions.set("setResVec", 1);
    mOptions.set("isPC", options.getLabel("isPC"));
    daResidual.masterFunction(mOptions, xvVec, wVec, resVecRef);

    scalar delta = daOption_.getSubDictOption<scalar>("adjPartDerivFDStep", "FFD");
    scalar rDelta = 1.0 / delta;
    PetscScalar rDeltaValue = 0.0;
    assignValueCheckAD(rDeltaValue, rDelta);

    Vec xvVecNew;
    VecDuplicate(xvVec, &xvVecNew);
    VecZeroEntries(xvVecNew);

    label printInterval = daOption_.getOption<label>("printInterval");
    for (label i = 0; i < nDesignVars; i++)
    {
        label eTime = mesh_.time().elapsedClockTime();
        // print progress
        if (i % printInterval == 0 or i == nDesignVars - 1)
        {
            Info << modelType_ << ": " << i << " of " << nDesignVars
                 << ", ExecutionTime: " << eTime << " s" << endl;
        }

        // perturb FFD
        VecZeroEntries(xvVecNew);
        MatGetColumnVector(dXvdFFDMat_, xvVecNew, i);
        VecAXPY(xvVecNew, 1.0, xvVec);

        // compute residual
        daResidual.masterFunction(mOptions, xvVecNew, wVec, resVec);

        // no need to reset FFD here

        // compute residual partial using finite-difference
        VecAXPY(resVec, -1.0, resVecRef);
        VecScale(resVec, rDeltaValue);

        // assign resVec to jacMat
        PetscInt Istart, Iend;
        VecGetOwnershipRange(resVec, &Istart, &Iend);

        const PetscScalar* resVecArray;
        VecGetArrayRead(resVec, &resVecArray);
        for (label j = Istart; j < Iend; j++)
        {
            label relIdx = j - Istart;
            PetscScalar val = resVecArray[relIdx];
            MatSetValue(jacMat, j, i, val, INSERT_VALUES);
        }
        VecRestoreArrayRead(resVec, &resVecArray);
    }

    // call the master function again to reset the xvVec and wVec to OpenFOAM fields and points
    daResidual.masterFunction(mOptions, xvVec, wVec, resVecRef);

    MatAssemblyBegin(jacMat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(jacMat, MAT_FINAL_ASSEMBLY);
}

} // End namespace Foam

// ************************************************************************* //
