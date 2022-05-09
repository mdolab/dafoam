/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAPartDerivdRdAOA.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAPartDerivdRdAOA, 0);
addToRunTimeSelectionTable(DAPartDeriv, DAPartDerivdRdAOA, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAPartDerivdRdAOA::DAPartDerivdRdAOA(
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

void DAPartDerivdRdAOA::initializePartDerivMat(
    const dictionary& options,
    Mat jacMat)
{
    /*
    Description:
        Initialize jacMat
    
    Input:
        options. this is not used
    */

    // now initialize the memory for the jacobian itself
    label localSize = daIndex_.nLocalAdjointStates;

    // create dRdAOAT
    //MatCreate(PETSC_COMM_WORLD, jacMat);
    MatSetSizes(
        jacMat,
        localSize,
        PETSC_DECIDE,
        PETSC_DETERMINE,
        1);
    MatSetFromOptions(jacMat);
    MatMPIAIJSetPreallocation(jacMat, 1, NULL, 1, NULL);
    MatSeqAIJSetPreallocation(jacMat, 1, NULL);
    //MatSetOption(jacMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(jacMat);
    MatZeroEntries(jacMat);
    Info << "Partial derivative matrix created. " << mesh_.time().elapsedClockTime() << " s" << endl;
}

void DAPartDerivdRdAOA::calcPartDerivMat(
    const dictionary& options,
    const Vec xvVec,
    const Vec wVec,
    Mat jacMat)
{
    /*
    Description:
        Compute jacMat. We use brute-force finite-difference
    
    Input:

        options.isPC: whether to compute the jacMat for preconditioner

        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector
    
    Output:
        jacMat: the partial derivative matrix dRdAOA to compute
    */

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
    mOptions.set("updateMesh", 0);
    mOptions.set("setResVec", 1);
    mOptions.set("isPC", options.getLabel("isPC"));
    daResidual.masterFunction(mOptions, xvVec, wVec, resVecRef);

    scalar delta = daOption_.getSubDictOption<scalar>("adjPartDerivFDStep", "AOA");
    scalar rDelta = 1.0 / delta;
    PetscScalar rDeltaValue = 0.0;
    assignValueCheckAD(rDeltaValue, rDelta);

    // perturb angle of attack
    this->perturbAOA(options, delta);

    // compute residual
    daResidual.masterFunction(mOptions, xvVec, wVec, resVec);

    // compute residual partial using finite-difference
    VecAXPY(resVec, -1.0, resVecRef);
    VecScale(resVec, rDeltaValue);

    // assign resVec to jacMat
    PetscInt Istart, Iend;
    VecGetOwnershipRange(resVec, &Istart, &Iend);

    const PetscScalar* resVecArray;
    VecGetArrayRead(resVec, &resVecArray);
    for (label i = Istart; i < Iend; i++)
    {
        label relIdx = i - Istart;
        PetscScalar val = resVecArray[relIdx];
        MatSetValue(jacMat, i, 0, val, INSERT_VALUES);
    }
    VecRestoreArrayRead(resVec, &resVecArray);

    // reset perturbation
    this->perturbAOA(options, -1.0 * delta);
    // call masterFunction again to reset the wVec to OpenFOAM field
    daResidual.masterFunction(mOptions, xvVec, wVec, resVec);

    MatAssemblyBegin(jacMat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(jacMat, MAT_FINAL_ASSEMBLY);

}

} // End namespace Foam

// ************************************************************************* //
