/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAPartDerivdRdACTD.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAPartDerivdRdACTD, 0);
addToRunTimeSelectionTable(DAPartDeriv, DAPartDerivdRdACTD, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAPartDerivdRdACTD::DAPartDerivdRdACTD(
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

void DAPartDerivdRdACTD::initializePartDerivMat(
    const dictionary& options,
    Mat jacMat)
{
    /*
    Description:
        Initialize jacMat
    */

    // now initialize the memory for the jacobian itself
    label localSize = daIndex_.nLocalAdjointStates;

    // create dRdACTDT
    //MatCreate(PETSC_COMM_WORLD, jacMat);
    MatSetSizes(
        jacMat,
        localSize,
        PETSC_DECIDE,
        PETSC_DETERMINE,
        nActDVs_);
    MatSetFromOptions(jacMat);
    MatMPIAIJSetPreallocation(jacMat, nActDVs_, NULL, nActDVs_, NULL);
    MatSeqAIJSetPreallocation(jacMat, nActDVs_, NULL);
    //MatSetOption(jacMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(jacMat);
    MatZeroEntries(jacMat);
    Info << "Partial derivative matrix created. " << mesh_.time().elapsedClockTime() << " s" << endl;
}

void DAPartDerivdRdACTD::calcPartDerivMat(
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
        jacMat: the partial derivative matrix dRdACTD to compute
    */

    word actuatorName = options.getWord("actuatorName");

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

    scalar delta = daOption_.getSubDictOption<scalar>("adjPartDerivFDStep", "ACTD");
    scalar rDelta = 1.0 / delta;
    PetscScalar rDeltaValue = 0.0;
    assignValueCheckAD(rDeltaValue, rDelta);

    Vec xvVecNew;
    VecDuplicate(xvVec, &xvVecNew);
    VecZeroEntries(xvVecNew);

    dictionary& diskModelSubDict = const_cast<dictionary&>(
        daOption_.getAllOptions().subDict("fvSource").subDict(actuatorName));

    scalarList center;
    diskModelSubDict.readEntry<scalarList>("center", center);
    scalar innerRadius = diskModelSubDict.getScalar("innerRadius");
    scalar outerRadius = diskModelSubDict.getScalar("outerRadius");
    scalar POD = diskModelSubDict.getScalar("POD");
    scalar scale = diskModelSubDict.getScalar("scale");
    scalar expM = diskModelSubDict.getScalar("expM");
    scalar expN = diskModelSubDict.getScalar("expN");
    scalar targetThrust = diskModelSubDict.getScalar("targetThrust");

    DAFvSource& fvSource = const_cast<DAFvSource&>(
        mesh_.thisDb().lookupObject<DAFvSource>("DAFvSource"));

    for (label i = 0; i < nActDVs_; i++)
    {

        // perturb ACTD
        if (i == 0)
        {
            // perturb x
            center[0] += delta;
            diskModelSubDict.set("center", center);
        }
        else if (i == 1)
        {
            // perturb y
            center[1] += delta;
            diskModelSubDict.set("center", center);
        }
        else if (i == 2)
        {
            // perturb z
            center[2] += delta;
            diskModelSubDict.set("center", center);
        }
        else if (i == 3)
        {
            // perturb innerRadius
            innerRadius += delta;
            diskModelSubDict.set("innerRadius", innerRadius);
        }
        else if (i == 4)
        {
            // perturb outerRadius
            outerRadius += delta;
            diskModelSubDict.set("outerRadius", outerRadius);
        }
        else if (i == 5)
        {
            // perturb scale
            scale += delta;
            diskModelSubDict.set("scale", scale);
        }
        else if (i == 6)
        {
            // perturb POD
            POD += delta;
            diskModelSubDict.set("POD", POD);
        }
        else if (i == 7)
        {
            // perturb expM
            expM += delta;
            diskModelSubDict.set("expM", expM);
        }
        else if (i == 8)
        {
            // perturb expN
            expN += delta;
            diskModelSubDict.set("expN", expN);
        }
        else if (i == 9)
        {
            targetThrust += delta;
            diskModelSubDict.set("targetThrust", targetThrust);
        }

        // we need to synchronize the DAOption to actuatorDVs
        fvSource.syncDAOptionToActuatorDVs();

        // Info<<daOption_.getAllOptions().subDict("fvSource")<<endl;

        // compute residual
        daResidual.masterFunction(mOptions, xvVecNew, wVec, resVec);

        // reset perturbation
        if (i == 0)
        {
            // reset x
            center[0] -= delta;
            diskModelSubDict.set("center", center);
        }
        else if (i == 1)
        {
            // reset y
            center[1] -= delta;
            diskModelSubDict.set("center", center);
        }
        else if (i == 2)
        {
            // reset z
            center[2] -= delta;
            diskModelSubDict.set("center", center);
        }
        else if (i == 3)
        {
            // reset innerRadius
            innerRadius -= delta;
            diskModelSubDict.set("innerRadius", innerRadius);
        }
        else if (i == 4)
        {
            // reset outerRadius
            outerRadius -= delta;
            diskModelSubDict.set("outerRadius", outerRadius);
        }
        else if (i == 5)
        {
            // reset scale
            scale -= delta;
            diskModelSubDict.set("scale", scale);
        }
        else if (i == 6)
        {
            // reset POD
            POD -= delta;
            diskModelSubDict.set("POD", POD);
        }
        else if (i == 7)
        {
            // reset expM
            expM -= delta;
            diskModelSubDict.set("expM", expM);
        }
        else if (i == 8)
        {
            // reset expN
            expN -= delta;
            diskModelSubDict.set("expN", expN);
        }
        else if (i == 9)
        {
            targetThrust -= delta;
            diskModelSubDict.set("targetThrust", targetThrust);
        }

        // we need to synchronize the DAOption to actuatorDVs
        fvSource.syncDAOptionToActuatorDVs();

        // Info<<daOption_.getAllOptions().subDict("fvSource")<<endl;

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

    label eTime = mesh_.time().elapsedClockTime();
    Info << modelType_ << " ExecutionTime: " << eTime << " s" << endl;

    // call the master function again to reset the xvVec and wVec to OpenFOAM fields and points
    daResidual.masterFunction(mOptions, xvVec, wVec, resVecRef);

    MatAssemblyBegin(jacMat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(jacMat, MAT_FINAL_ASSEMBLY);
}

} // End namespace Foam

// ************************************************************************* //
