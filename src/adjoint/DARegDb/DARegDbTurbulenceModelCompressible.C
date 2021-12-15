/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DARegDbTurbulenceModelCompressible.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Constructors
DARegDbTurbulenceModelCompressible::DARegDbTurbulenceModelCompressible(
    const fvMesh& mesh,
    const compressible::turbulenceModel& turbulenceModel)
    : regIOobject(
        IOobject(
            "DARegDbTurbulenceModelCompressible", 
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

DARegDbTurbulenceModelCompressible::~DARegDbTurbulenceModelCompressible()
{
}

// this is a virtual function for regIOobject
bool DARegDbTurbulenceModelCompressible::writeData(Ostream& os) const
{
    // do nothing
    return true;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
