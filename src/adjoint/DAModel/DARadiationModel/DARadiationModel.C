/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v5

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

    auto* ctorPtr = dictionaryConstructorTable(modelType);

    if (!ctorPtr)
    {
        FatalErrorInLookup(
            "DARadiationModel",
            modelType,
            *dictionaryConstructorTablePtr_)
            << exit(FatalError);
    }

    return autoPtr<DARadiationModel>(
        ctorPtr(modelType, mesh, daOption));
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
