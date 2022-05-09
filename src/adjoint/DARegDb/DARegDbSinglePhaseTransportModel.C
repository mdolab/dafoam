/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DARegDbSinglePhaseTransportModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Constructors
DARegDbSinglePhaseTransportModel::DARegDbSinglePhaseTransportModel(
    const fvMesh& mesh,
    const singlePhaseTransportModel& singlePhaseTransportModel)
    : regIOobject(
        IOobject(
            "DARegDbSinglePhaseTransportModel", // always use DARegDbSinglePhaseTransportModel for the db name
            mesh.time().timeName(),
            mesh, // register to mesh
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            true // always register object
            )),
      mesh_(mesh),
      singlePhaseTransportModel_(singlePhaseTransportModel)
{
}

DARegDbSinglePhaseTransportModel::~DARegDbSinglePhaseTransportModel()
{
}

// this is a virtual function for regIOobject
bool DARegDbSinglePhaseTransportModel::writeData(Ostream& os) const
{
    // do nothing
    return true;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
