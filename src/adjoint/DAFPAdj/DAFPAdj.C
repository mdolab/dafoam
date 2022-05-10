/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAFPAdj.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defineTypeNameAndDebug(DAFPAdj, 0);
defineRunTimeSelectionTable(DAFPAdj, dictionary);

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFPAdj::DAFPAdj(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex,
    const DAResidual& daResidual)
    : mesh_(mesh),
      daOption_(daOption),
      daModel_(daModel),
      daIndex_(daIndex),
      daResidual_(daResidual),
      daField_(mesh, daOption, daModel, daIndex)
{
}

// * * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * //

autoPtr<DAFPAdj> DAFPAdj::New(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex,
    const DAResidual& daResidual)
{
    // standard setup for runtime selectable classes

    if (daOption.getAllOptions().lookupOrDefault<label>("debug", 0))
    {
        Info << "Selecting " << modelType << " for DAFPAdj" << endl;
    }

    dictionaryConstructorTable::iterator cstrIter =
        dictionaryConstructorTablePtr_->find(modelType);

    // if the solver name is not found in any child class, print an error
    if (cstrIter == dictionaryConstructorTablePtr_->end())
    {
        FatalErrorIn(
            "DAFPAdj::New"
            "("
            "    const word,"
            "    const fvMesh&,"
            "    const DAOption&,"
            "    const DAModel&,"
            "    const DAIndex&,"
            "    const DAResidual&,"
            ")")
            << "Unknown DAFPAdj type "
            << modelType << nl << nl
            << "Valid DAFPAdj types:" << endl
            << dictionaryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    // child class found
    return autoPtr<DAFPAdj>(
        cstrIter()(modelType,
                   mesh,
                   daOption,
                   daModel,
                   daIndex,
                   daResidual));
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
