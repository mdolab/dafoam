/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAPartDerivdRdW.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAPartDerivdRdW, 0);
addToRunTimeSelectionTable(DAPartDeriv, DAPartDerivdRdW, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAPartDerivdRdW::DAPartDerivdRdW(
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

void DAPartDerivdRdW::initializePartDerivMat(
    const dictionary& options,
    Mat jacMat)
{
    /*
    Description:
        Initialize jacMat
    
    Input:
        options.transposed. Whether to compute the transposed of dRdW
    */

    label transposed = options.getLabel("transposed");

    // now initialize the memory for the jacobian itself
    label localSize = daIndex_.nLocalAdjointStates;

    // create dRdWT
    //MatCreate(PETSC_COMM_WORLD, jacMat);
    MatSetSizes(
        jacMat,
        localSize,
        localSize,
        PETSC_DETERMINE,
        PETSC_DETERMINE);
    MatSetFromOptions(jacMat);
    daJacCon_.preallocatedRdW(jacMat, transposed);
    //MatSetOption(jacMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(jacMat);
    MatZeroEntries(jacMat);
    Info << "Partial derivative matrix created. " << mesh_.time().elapsedClockTime() << " s" << endl;
}

void DAPartDerivdRdW::calcPartDerivMat(
    const dictionary& options,
    const Vec xvVec,
    const Vec wVec,
    Mat jacMat)
{
    /*
    Description:
        Compute jacMat. We use coloring accelerated finite-difference
    
    Input:

        options.transposed. Whether to compute the transposed of dRdW

        options.isPC: whether to compute the jacMat for preconditioner

        options.lowerBound: any |value| that is smaller than lowerBound will be set to zero in dRdW

        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector
    
    Output:
        jacMat: the partial derivative matrix dRdW to compute
    */

    label transposed = options.getLabel("transposed");

    // initialize coloredColumn vector
    Vec coloredColumn;
    VecDuplicate(wVec, &coloredColumn);
    VecZeroEntries(coloredColumn);

    DAResidual& daResidual = const_cast<DAResidual&>(daResidual_);

    // zero all the matrices
    MatZeroEntries(jacMat);

    Vec wVecNew;
    VecDuplicate(wVec, &wVecNew);
    VecCopy(wVec, wVecNew);

    // initialize residual vectors
    Vec resVecRef, resVec;
    VecDuplicate(wVec, &resVec);
    VecDuplicate(wVec, &resVecRef);
    VecZeroEntries(resVec);
    VecZeroEntries(resVecRef);

    // set up state normalization vector
    Vec normStatePerturbVec;
    this->setNormStatePerturbVec(&normStatePerturbVec);

    dictionary mOptions;
    mOptions.set("updateState", 1);
    mOptions.set("updateMesh", 0);
    mOptions.set("setResVec", 1);
    mOptions.set("isPC", options.getLabel("isPC"));
    daResidual.masterFunction(mOptions, xvVec, wVec, resVecRef);

    scalar jacLowerBound = options.getScalar("lowerBound");

    scalar delta = daOption_.getSubDictOption<scalar>("adjPartDerivFDStep", "State");
    scalar rDelta = 1.0 / delta;
    PetscScalar rDeltaValue = 0.0;
    assignValueCheckAD(rDeltaValue, rDelta);

    label nColors = daJacCon_.getNJacConColors();

    word partDerivName = modelType_;
    if (transposed)
    {
        partDerivName += "T";
    }
    if (options.getLabel("isPC"))
    {
        partDerivName += "PC";
    }

    label printInterval = daOption_.getOption<label>("printInterval");
    for (label color = 0; color < nColors; color++)
    {
        label eTime = mesh_.time().elapsedClockTime();
        // print progress
        if (color % printInterval == 0 or color == nColors - 1)
        {
            Info << partDerivName << ": " << color << " of " << nColors
                 << ", ExecutionTime: " << eTime << " s" << endl;
        }

        // perturb states
        this->perturbStates(
            daJacCon_.getJacConColor(),
            normStatePerturbVec,
            color,
            delta,
            wVecNew);

        // compute residual
        daResidual.masterFunction(mOptions, xvVec, wVecNew, resVec);

        // reset state perburbation
        VecCopy(wVec, wVecNew);

        // compute residual partial using finite-difference
        VecAXPY(resVec, -1.0, resVecRef);
        VecScale(resVec, rDeltaValue);

        // compute the colored coloumn and assign resVec to jacMat
        daJacCon_.calcColoredColumns(color, coloredColumn);
        this->setPartDerivMat(resVec, coloredColumn, transposed, jacMat, jacLowerBound);
    }

    // call masterFunction again to reset the wVec to OpenFOAM field
    daResidual.masterFunction(mOptions, xvVec, wVec, resVecRef);

    MatAssemblyBegin(jacMat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(jacMat, MAT_FINAL_ASSEMBLY);

    if (daOption_.getOption<label>("debug"))
    {
        daIndex_.printMatChars(jacMat);
    }
}

} // End namespace Foam

// ************************************************************************* //
