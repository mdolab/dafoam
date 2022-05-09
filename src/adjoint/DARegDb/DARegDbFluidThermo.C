/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DARegDbFluidThermo.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Constructors
DARegDbFluidThermo::DARegDbFluidThermo(
    const fvMesh& mesh,
    const fluidThermo& fluidThermo)
    : regIOobject(
        IOobject(
            "DARegDbFluidThermo", // always use DARegDbFluidThermo for the db name
            mesh.time().timeName(),
            mesh, // register to mesh
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            true // always register object
            )),
      mesh_(mesh),
      fluidThermo_(fluidThermo)
{
}

DARegDbFluidThermo::~DARegDbFluidThermo()
{
}

// this is a virtual function for regIOobject
bool DARegDbFluidThermo::writeData(Ostream& os) const
{
    // do nothing
    return true;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
