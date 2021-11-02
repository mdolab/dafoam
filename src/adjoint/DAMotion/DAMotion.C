/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAMotion.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defineTypeNameAndDebug(DAMotion, 0);
defineRunTimeSelectionTable(DAMotion, dictionary);

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAMotion::DAMotion(
    const dynamicFvMesh& mesh,
    const DAOption& daOption)
    : mesh_(mesh),
      daOption_(daOption)
{
}

// * * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * //

autoPtr<DAMotion> DAMotion::New(
    const dynamicFvMesh& mesh,
    const DAOption& daOption)
{
    // standard setup for runtime selectable classes

    // look up the solver name
    word modelType = daOption.getAllOptions().subDict("rigidBodyMotion").getWord("mode");

    if (daOption.getAllOptions().lookupOrDefault<label>("debug", 0))
    {
        Info << "Selecting type: " << modelType << " for DAMotion. " << endl;
    }

    dictionaryConstructorTable::iterator cstrIter =
        dictionaryConstructorTablePtr_->find(modelType);

    // if the solver name is not found in any child class, print an error
    if (cstrIter == dictionaryConstructorTablePtr_->end())
    {
        FatalErrorIn(
            "DAMotion::New"
            "("
            "    const dynamicFvMesh&,"
            "    const DAOption&,"
            ")")
            << "Unknown DAMotion type "
            << modelType << nl << nl
            << "Valid DAMotion types:" << endl
            << dictionaryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    // child class found
    return autoPtr<DAMotion>(
        cstrIter()(mesh,
                   daOption));
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
