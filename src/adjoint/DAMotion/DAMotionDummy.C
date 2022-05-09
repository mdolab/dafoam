/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAMotionDummy.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAMotionDummy, 0);
addToRunTimeSelectionTable(DAMotion, DAMotionDummy, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAMotionDummy::DAMotionDummy(
    const dynamicFvMesh& mesh,
    const DAOption& daOption)
    : DAMotion(
        mesh,
        daOption)
{
}

void DAMotionDummy::correct()
{
    // do nothing here
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
