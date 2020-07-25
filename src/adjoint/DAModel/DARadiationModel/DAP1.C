/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAP1.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAP1, 0);
addToRunTimeSelectionTable(DARadiationModel, DAP1, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAP1::DAP1(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption)
    : DARadiationModel(modelType, mesh, daOption)
{
}

/// add the model residual connectivity to stateCon
void DAP1::addModelResidualCon(HashTable<List<List<word>>>& allCon) const
{
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
