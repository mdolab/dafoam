/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAPartDerivdFdBC.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAPartDerivdFdBC, 0);
addToRunTimeSelectionTable(DAPartDeriv, DAPartDerivdFdBC, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAPartDerivdFdBC::DAPartDerivdFdBC(
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

void DAPartDerivdFdBC::initializePartDerivMat(
    const dictionary& options,
    Mat jacMat)
{
    /*
    Description:
        Initialize jacMat
    
    Input:
        options. This is not used
    */

    // create dFdBC
    //MatCreate(PETSC_COMM_WORLD, jacMat);
    MatSetSizes(
        jacMat,
        PETSC_DECIDE,
        PETSC_DECIDE,
        1,
        1);
    MatSetFromOptions(jacMat);
    MatMPIAIJSetPreallocation(jacMat, 1, NULL, 1, NULL);
    MatSeqAIJSetPreallocation(jacMat, 1, NULL);
    //MatSetOption(jacMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(jacMat);
    MatZeroEntries(jacMat);
    Info << "Partial derivative matrix created. " << mesh_.time().elapsedClockTime() << " s" << endl;
}

void DAPartDerivdFdBC::calcPartDerivMat(
    const dictionary& options,
    const Vec xvVec,
    const Vec wVec,
    Mat jacMat)
{
    /*
    Description:
        Compute jacMat. Note for dFdBC, we have only one column so we can do brute 
        force finite-difference there is no need to do coloring
    
    Input:
        options.objFuncSubDictPart: the objFunc subDict, obtained from DAOption

        options.objFuncName: the name of the objective

        options.objFuncPart: the part of the objective

        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector
    
    Output:
        jacMat: the partial derivative matrix dFdBC to compute
    */

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

    dictionary mOptions;
    mOptions.set("updateState", 1);
    mOptions.set("updateMesh", 0);
    scalar fRef = daObjFunc->masterFunction(mOptions, xvVec, wVec);

    scalar delta = daOption_.getSubDictOption<scalar>("adjPartDerivFDStep", "BC");
    scalar rDelta = 1.0 / delta;

    // perturb BC
    this->perturbBC(options, delta);

    // compute object
    scalar fNew = daObjFunc->masterFunction(mOptions, xvVec, wVec);

    scalar partDeriv = (fNew - fRef) * rDelta;
    PetscScalar partDerivValue = 0.0;
    assignValueCheckAD(partDerivValue, partDeriv);

    MatSetValue(jacMat, 0, 0, partDerivValue, INSERT_VALUES);

    // reset perturbation
    this->perturbBC(options, -1.0 * delta);
    // call masterFunction again to reset the wVec to OpenFOAM field
    fRef = daObjFunc->masterFunction(mOptions, xvVec, wVec);

    if (daOption_.getOption<label>("debug"))
    {
        Info << objFuncName << ": " << fRef << endl;
    }

    MatAssemblyBegin(jacMat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(jacMat, MAT_FINAL_ASSEMBLY);
}

} // End namespace Foam

// ************************************************************************* //
