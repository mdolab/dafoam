/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAPartDerivdFdState.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAPartDerivdFdState, 0);
addToRunTimeSelectionTable(DAPartDeriv, DAPartDerivdFdState, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAPartDerivdFdState::DAPartDerivdFdState(
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

void DAPartDerivdFdState::initializePartDerivMat(
    const dictionary& options,
    Mat jacMat)
{
    /*
    Description:
        Initialize jacMat
    
    Input:
        options. This is not used
    */

    // create dFdState
    label nCells = mesh_.nCells();
    //MatCreate(PETSC_COMM_WORLD, jacMat);
    MatSetSizes(
        jacMat,
        nCells,
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

void DAPartDerivdFdState::calcPartDerivMat(
    const dictionary& options,
    const Vec xvVec,
    const Vec wVec,
    Mat jacMat)
{
    /*
    Description:
        Compute jacMat. We use the analytical method
    
    Input:
        options.objFuncSubDictPart: the objFunc subDict, obtained from DAOption

        options.stateName: the name of the state in designVar

        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector
    
    Output:
        jacMat: the partial derivative matrix dFdState to compute
    */

    // zero all the matrices

    MatZeroEntries(jacMat);

    dictionary objFuncSubDictPart = options.subDict("objFuncSubDictPart");
    word objFuncType = objFuncSubDictPart.getWord("type");
    word stateNameInDesignVar = options.getWord("stateName");

    if (objFuncType == "stateErrorNorm")
    {
        if (stateNameInDesignVar == "betaSA")
        {
            // if the stateName in designVar equals the state name defined in objFunc,
            // compute dFdState, otherwise, set dFdState = 0
            word stateNameInObjFunc = objFuncSubDictPart.getWord("stateName");
            if (stateNameInDesignVar == stateNameInObjFunc)
            {

                scalar scale = objFuncSubDictPart.getScalar("scale");
                word stateRefName = objFuncSubDictPart.getWord("stateRefName");
                const objectRegistry& db = mesh_.thisDb();
                const volScalarField& state = db.lookupObject<volScalarField>(stateNameInDesignVar);
                const volScalarField& stateRef = db.lookupObject<volScalarField>(stateRefName);

                forAll(mesh_.cells(), cellI)
                {
                    label globalCellI = daIndex_.getGlobalCellIndex(cellI);
                    scalar partDeriv = 2.0 * scale * (state[cellI] - stateRef[cellI]);
                    PetscScalar partDerivValue = 0.0;
                    assignValueCheckAD(partDerivValue, partDeriv);
                    MatSetValue(jacMat, globalCellI, 0, partDerivValue, INSERT_VALUES);
                }

            }
        }
        else
        {
            FatalErrorIn("") << "stateName: " << stateNameInDesignVar << " not supported!"
                             << abort(FatalError);
        }
    }
    else
    {
        FatalErrorIn("") << "objFuncType: " << objFuncType << " not supported!"
                         << abort(FatalError);
    }

    MatAssemblyBegin(jacMat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(jacMat, MAT_FINAL_ASSEMBLY);
}

} // End namespace Foam

// ************************************************************************* //
