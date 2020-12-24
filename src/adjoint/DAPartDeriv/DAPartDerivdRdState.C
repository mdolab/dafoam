/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAPartDerivdRdState.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAPartDerivdRdState, 0);
addToRunTimeSelectionTable(DAPartDeriv, DAPartDerivdRdState, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAPartDerivdRdState::DAPartDerivdRdState(
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

void DAPartDerivdRdState::initializePartDerivMat(
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
    label nCells = mesh_.nCells();

    // create dRdState
    //MatCreate(PETSC_COMM_WORLD, jacMat);
    MatSetSizes(
        jacMat,
        localSize,
        nCells,
        PETSC_DETERMINE,
        PETSC_DETERMINE);
    MatSetFromOptions(jacMat);
    MatMPIAIJSetPreallocation(jacMat, 1, NULL, 1, NULL);
    MatSeqAIJSetPreallocation(jacMat, 1, NULL);
    //MatSetOption(jacMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(jacMat);
    MatZeroEntries(jacMat);
    Info << "Partial derivative matrix created. " << mesh_.time().elapsedClockTime() << " s" << endl;
}

void DAPartDerivdRdState::calcPartDerivMat(
    const dictionary& options,
    const Vec xvVec,
    const Vec wVec,
    Mat jacMat)
{
    /*
    Description:
        Compute jacMat. We use analytical method
    
    Input:

        options.stateName: the name of the state in designVar

        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector
    
    Output:
        jacMat: the partial derivative matrix dRdState to compute
    */

    // zero all the matrices
    MatZeroEntries(jacMat);

    word stateNameInDesignVar = options.getWord("stateName");

    if (stateNameInDesignVar == "betaSA")
    {

        scalarList prodTerm(mesh_.nCells());
        daModel_.getTurbProdTerm(prodTerm);

        // dRdBetaSA only has diagonal component for the turbulence residual
        forAll(mesh_.cells(), cellI)
        {
            label globalIndex = daIndex_.getGlobalAdjointStateIndex("nuTilda", cellI);
            label globalCellIndex = daIndex_.getGlobalCellIndex(cellI);
            // note the dR/dBetaSA is the negative of the prod term because
            // the prod term shows up on the right hand side of the residual equation
            scalar val = -prodTerm[cellI];
            MatSetValue(jacMat, globalIndex, globalCellIndex, val, INSERT_VALUES);
        }
    }
    else
    {
        FatalErrorIn("") << "stateName: " << stateNameInDesignVar << "not supported!"
                         << abort(FatalError);
    }

    MatAssemblyBegin(jacMat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(jacMat, MAT_FINAL_ASSEMBLY);
}

} // End namespace Foam

// ************************************************************************* //
