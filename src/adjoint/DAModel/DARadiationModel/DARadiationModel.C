/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DARadiationModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defineTypeNameAndDebug(DARadiationModel, 0);
defineRunTimeSelectionTable(DARadiationModel, dictionary);

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DARadiationModel::DARadiationModel(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption)
    : regIOobject(
        IOobject(
            "DARadiationModel",
            mesh.time().timeName(),
            mesh, // register to mesh
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            true // always register object
            )),
      mesh_(mesh),
      daOption_(daOption)
{
}

// * * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * //

autoPtr<DARadiationModel> DARadiationModel::New(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption)
{
    if (daOption.getAllOptions().lookupOrDefault<label>("debug", 0))
    {
        Info << "Selecting " << modelType << " for DARadiationModel" << endl;
    }

    dictionaryConstructorTable::iterator cstrIter =
        dictionaryConstructorTablePtr_->find(modelType);

    if (cstrIter == dictionaryConstructorTablePtr_->end())
    {
        FatalErrorIn(
            "DARadiationModel::New"
            "("
            "    const word,"
            "    const fvMesh&,"
            "    const DAOption&"
            ")")
            << "Unknown DARadiationModel type "
            << modelType << nl << nl
            << "Valid DARadiationModel types:" << endl
            << dictionaryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    return autoPtr<DARadiationModel>(
        cstrIter()(modelType, mesh, daOption));
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

// this is a virtual function for regIOobject
bool DARadiationModel::writeData(Ostream& os) const
{
    // do nothing
    return true;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
