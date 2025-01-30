/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DATimeOp.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defineTypeNameAndDebug(DATimeOp, 0);
defineRunTimeSelectionTable(DATimeOp, dictionary);

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DATimeOp::DATimeOp(
    const word timeOpType)
    : timeOpType_(timeOpType)
{
}

// * * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * //

autoPtr<DATimeOp> DATimeOp::New(
    const word timeOpType)
{
    // standard setup for runtime selectable classes

    dictionaryConstructorTable::iterator cstrIter =
        dictionaryConstructorTablePtr_->find(timeOpType);

    // if the solver name is not found in any child class, print an error
    if (cstrIter == dictionaryConstructorTablePtr_->end())
    {
        FatalErrorIn(
            "DATimeOp::New"
            "("
            "    const word,"
            "    fvMesh&,"
            "    const DAOption&,"
            ")")
            << "Unknown DATimeOp type "
            << timeOpType << nl << nl
            << "Valid DATimeOp types:" << endl
            << dictionaryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    // child class found
    return autoPtr<DATimeOp>(
        cstrIter()(timeOpType));
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
