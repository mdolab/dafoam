/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v5

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
    auto* ctorPtr = dictionaryConstructorTable(timeOpType);

    if (!ctorPtr)
    {
        FatalErrorInLookup(
            "DATimeOp",
            timeOpType,
            *dictionaryConstructorTablePtr_)
            << exit(FatalError);
    }

    return autoPtr<DATimeOp>(ctorPtr(timeOpType, options));
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
