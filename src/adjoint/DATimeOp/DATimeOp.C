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
    const word timeOpType,
    const dictionary options)
    : timeOpType_(timeOpType),
      options_(options)
{
}

// * * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * //

autoPtr<DATimeOp> DATimeOp::New(
    const word timeOpType,
    const dictionary options)
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
            "    const dictionary,"
            ")")
            << "Unknown DATimeOp type "
            << timeOpType << nl << nl
            << "Valid DATimeOp types:" << endl
            << dictionaryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    // child class found
    return autoPtr<DATimeOp>(
        cstrIter()(timeOpType, options));
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
