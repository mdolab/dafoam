/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DARegDbNearWallDist.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Constructors
DARegDbNearWallDist::DARegDbNearWallDist(
    const fvMesh& mesh,
    const nearWallDist& nearWallDist)
    : regIOobject(
        IOobject(
            "DARegDbNearWallDist", // always use DARegDbNearWallDist for the db name
            mesh.time().timeName(),
            mesh, // register to mesh
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            true // always register object
            )),
      mesh_(mesh),
      nearWallDist_(nearWallDist)
{
}

DARegDbNearWallDist::~DARegDbNearWallDist()
{
}

// this is a virtual function for regIOobject
bool DARegDbNearWallDist::writeData(Ostream& os) const
{
    // do nothing
    return true;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
