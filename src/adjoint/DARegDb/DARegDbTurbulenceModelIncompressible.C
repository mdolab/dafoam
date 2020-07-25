/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DARegDbTurbulenceModelIncompressible.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Constructors
DARegDbTurbulenceModelIncompressible::DARegDbTurbulenceModelIncompressible(
    const fvMesh& mesh,
    const incompressible::turbulenceModel& turbulenceModel)
    : regIOobject(
        IOobject(
            "DARegDbTurbulenceModelIncompressible", 
            mesh.time().timeName(),
            mesh, // register to mesh
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            true // always register object
            )),
      mesh_(mesh),
      turbulenceModel_(turbulenceModel)
{
}

DARegDbTurbulenceModelIncompressible::~DARegDbTurbulenceModelIncompressible()
{
}

// this is a virtual function for regIOobject
bool DARegDbTurbulenceModelIncompressible::writeData(Ostream& os) const
{
    // do nothing
    return true;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
