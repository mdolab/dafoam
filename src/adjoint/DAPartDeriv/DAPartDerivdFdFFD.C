/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAPartDerivdFdFFD.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAPartDerivdFdFFD, 0);
addToRunTimeSelectionTable(DAPartDeriv, DAPartDerivdFdFFD, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAPartDerivdFdFFD::DAPartDerivdFdFFD(
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

void DAPartDerivdFdFFD::initializePartDerivMat(
    const dictionary& options,
    Mat jacMat)
{
    /*
    Description:
        Initialize jacMat
    
    Input:
        options.nDesignVars: The number of design variable for dFdFFD
    */

    label nDesignVars = options.getLabel("nDesignVars");

    // create dFdFFD
    //MatCreate(PETSC_COMM_WORLD, jacMat);
    MatSetSizes(
        jacMat,
        PETSC_DECIDE,
        PETSC_DECIDE,
        1,
        nDesignVars);
    MatSetFromOptions(jacMat);
    MatMPIAIJSetPreallocation(jacMat, nDesignVars, NULL, nDesignVars, NULL);
    MatSeqAIJSetPreallocation(jacMat, nDesignVars, NULL);
    //MatSetOption(jacMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(jacMat);
    MatZeroEntries(jacMat);
    Info << "Partial derivative matrix created. " << mesh_.time().elapsedClockTime() << " s" << endl;
}

void DAPartDerivdFdFFD::calcPartDerivMat(
    const dictionary& options,
    const Vec xvVec,
    const Vec wVec,
    Mat jacMat)
{
    /*
    Description:
        Compute jacMat. Note for dFdFFD, we do brute force finite-difference
        there is no need to do coloring
    
    Input:
        options.nDesignVars: The number of design variable for dFdFFD

        options.objFuncSubDictPart: the objFunc subDict, obtained from DAOption

        options.objFuncName: the name of the objective

        options.objFuncPart: the part of the objective

        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector
    
    Output:
        jacMat: the partial derivative matrix dFdFFD to compute
    */

    label nDesignVars = options.getLabel("nDesignVars");

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
    mOptions.set("updateMesh", 1);
    scalar fRef = daObjFunc->masterFunction(mOptions, xvVec, wVec);

    scalar delta = daOption_.getSubDictOption<scalar>("adjPartDerivFDStep", "FFD");
    scalar rDelta = 1.0 / delta;

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

        // compute object
        scalar fNew = daObjFunc->masterFunction(mOptions, xvVecNew, wVec);

        // no need to reset FFD here

        scalar partDeriv = (fNew - fRef) * rDelta;
        PetscScalar partDerivValue = 0.0;
        assignValueCheckAD(partDerivValue, partDeriv);

        MatSetValue(jacMat, 0, i, partDerivValue, INSERT_VALUES);
    }

    // call the master function again to reset xvVec and wVec to OpenFOAM fields and points
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
