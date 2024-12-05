/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAInput.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defineTypeNameAndDebug(DAInput, 0);
defineRunTimeSelectionTable(DAInput, dictionary);

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAInput::DAInput(
    const word inputName,
    fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : inputName_(inputName),
      mesh_(mesh),
      daOption_(daOption),
      daModel_(daModel),
      daIndex_(daIndex)
{
    // initialize stateInfo_
    word solverName = daOption_.getOption<word>("solverName");
    autoPtr<DAStateInfo> daStateInfo(DAStateInfo::New(solverName, mesh, daOption, daModel));
    stateInfo_ = daStateInfo->getStateInfo();
}

// * * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * //

autoPtr<DAInput> DAInput::New(
    const word inputName,
    fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
{
    // standard setup for runtime selectable classes

    if (daOption.getAllOptions().lookupOrDefault<label>("debug", 0))
    {
        Info << "Selecting input: " << inputName << " for DAInput." << endl;
    }

    dictionaryConstructorTable::iterator cstrIter =
        dictionaryConstructorTablePtr_->find(inputName);

    // if the solver name is not found in any child class, print an error
    if (cstrIter == dictionaryConstructorTablePtr_->end())
    {
        FatalErrorIn(
            "DAInput::New"
            "("
            "    const word,"
            "    fvMesh&,"
            "    const DAOption&,"
            "    const DAModel&,"
            "    const DAIndex&"
            ")")
            << "Unknown DAInput type "
            << inputName << nl << nl
            << "Valid DAInput types:" << endl
            << dictionaryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    // child class found
    return autoPtr<DAInput>(
        cstrIter()(inputName,
                   mesh,
                   daOption,
                   daModel,
                   daIndex));
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
