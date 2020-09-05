/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAPartDerivdFdW.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAPartDerivdFdW, 0);
addToRunTimeSelectionTable(DAPartDeriv, DAPartDerivdFdW, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAPartDerivdFdW::DAPartDerivdFdW(
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

void DAPartDerivdFdW::initializePartDerivMat(
    const dictionary& options,
    Mat* jacMat)
{
    /*
    Description:
        Initialize jacMat
    
    Input:
        options.objFuncFaceSources:  The list of objFunc face indices

        options.objFuncCellSources:  The list of objFunc cell indices
    */

    labelList objFuncFaceSources;
    labelList objFuncCellSources;
    options.readEntry<labelList>("objFuncFaceSources", objFuncFaceSources);
    options.readEntry<labelList>("objFuncCellSources", objFuncCellSources);

    // nLocalObjFuncGeoElements: the number of objFunc discrete elements for local procs
    nLocalObjFuncGeoElements_ = objFuncFaceSources.size() + objFuncCellSources.size();

    // create dFdW
    MatCreate(PETSC_COMM_WORLD, jacMat);
    MatSetSizes(
        *jacMat,
        nLocalObjFuncGeoElements_,
        daIndex_.nLocalAdjointStates,
        PETSC_DETERMINE,
        PETSC_DETERMINE);
    MatSetFromOptions(*jacMat);
    MatMPIAIJSetPreallocation(*jacMat, 200, NULL, 200, NULL);
    MatSeqAIJSetPreallocation(*jacMat, 200, NULL);
    //MatSetOption(jacMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(*jacMat);
    MatZeroEntries(*jacMat);
    Info << "Partial deriative matrix created. " << mesh_.time().elapsedClockTime() << " s" << endl;
}

void DAPartDerivdFdW::calcPartDerivMat(
    const dictionary& options,
    const Vec xvVec,
    const Vec wVec,
    Mat jacMat)
{

    /*
    Description:
        Compute jacMat. We use coloring acclerated finite-difference for dFdW
    
    Input:

        options.objFuncSubDictPart: the objFunc subDict, obtained from DAOption

        options.objFuncName: the name of the objective

        options.objFuncPart: the part of the objective

        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector
    
    Output:
        jacMat: the partial derivative matrix dFdW to compute
    */

    label transposed = 0;

    // initialize coloredColumn vector
    Vec coloredColumn;
    VecCreate(PETSC_COMM_WORLD, &coloredColumn);
    VecSetSizes(coloredColumn, nLocalObjFuncGeoElements_, PETSC_DECIDE);
    VecSetFromOptions(coloredColumn);
    VecZeroEntries(coloredColumn);

    word objFuncName, objFuncPart;
    dictionary objFuncSubDictPart = options.subDict("objFuncSubDictPart");
    options.readEntry<word>("objFuncName", objFuncName);
    options.readEntry<word>("objFuncPart", objFuncPart);

    autoPtr<DAObjFunc> daObjFunc(
        DAObjFunc::New(
            mesh_,
            daOption_,
            daModel_,
            daIndex_,
            daResidual_,
            objFuncName,
            objFuncPart,
            objFuncSubDictPart)
            .ptr());

    // zero all the matrices
    MatZeroEntries(jacMat);

    Vec wVecNew;
    VecDuplicate(wVec, &wVecNew);
    VecCopy(wVec, wVecNew);

    // initialize f vectors
    Vec fVecRef, fVec;
    VecDuplicate(coloredColumn, &fVec);
    VecDuplicate(coloredColumn, &fVecRef);
    VecZeroEntries(fVec);
    VecZeroEntries(fVecRef);

    // set up state normalization vector
    Vec normStatePerturbVec;
    this->setNormStatePerturbVec(&normStatePerturbVec);

    dictionary mOptions;
    mOptions.set("updateState", 1);
    mOptions.set("updateMesh", 0);
    daObjFunc->masterFunction(mOptions, xvVec, wVec);
    const scalarList& objFuncFaceValues = daObjFunc->getObjFuncFaceValues();
    const scalarList& objFuncCellValues = daObjFunc->getObjFuncCellValues();
    daJacCon_.setObjFuncVec(objFuncFaceValues, objFuncCellValues, fVecRef);

    // NOTE: for some objectives, we need to scale dFdW so we first fetch their
    // scaling before perturbing W and after computing the reference objFunc
    scalar scalingKS = 1.0;
    if (objFuncSubDictPart.getWord("type") == "vonMisesStressKS")
    {
        scalar coeffKS = objFuncSubDictPart.getScalar("coeffKS");
	// expSumKS should be computed by calling the above masterFunction
	// based on unperturbed W
        scalingKS = 1.0 / daObjFunc->expSumKS / coeffKS;
    }

    scalar delta = daOption_.getSubDictOption<scalar>("adjPartDerivFDStep", "State");
    scalar rDelta = 1.0 / delta;

    label nColors = daJacCon_.getNJacConColors();

    label printInterval = daOption_.getOption<label>("printInterval");
    for (label color = 0; color < nColors; color++)
    {
        label eTime = mesh_.time().elapsedClockTime();
        // print progress
        if (color % printInterval == 0 or color == nColors - 1)
        {
            Info << modelType_ << ": " << color << " of " << nColors
                 << ", ExecutionTime: " << eTime << " s" << endl;
        }

        // perturb states
        this->perturbStates(
            daJacCon_.getJacConColor(),
            normStatePerturbVec,
            color,
            delta,
            wVecNew);

        // compute object
        daObjFunc->masterFunction(mOptions, xvVec, wVecNew);
        daJacCon_.setObjFuncVec(objFuncFaceValues, objFuncCellValues, fVec);

        // reset state perburbation
        VecCopy(wVec, wVecNew);

        // compute residual partial using finite-difference
        VecAXPY(fVec, -1.0, fVecRef);
        VecScale(fVec, rDelta);
        // NOTE: need to further scale fVec by scalingKS for KS objectives
	    // If no KS objectives are used, scalingKS=1
        VecScale(fVec, scalingKS);

        // compute the colored coloumn and assign resVec to jacMat
        daJacCon_.calcColoredColumns(color, coloredColumn);
        this->setPartDerivMat(fVec, coloredColumn, transposed, jacMat);
    }

    // call the master function again to reset wVec to OpenFOAM fields
    scalar fRef = daObjFunc->masterFunction(mOptions, xvVec, wVec);

    if (daOption_.getOption<label>("debug"))
    {
        Info << objFuncName << ": " << fRef << endl;
    }

    MatAssemblyBegin(jacMat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(jacMat, MAT_FINAL_ASSEMBLY);
}

} // End namespace Foam

// ************************************************************************* //
