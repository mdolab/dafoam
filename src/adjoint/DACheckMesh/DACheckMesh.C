/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DACheckMesh.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Constructors
DACheckMesh::DACheckMesh(
    const Time& runTime1,
    const fvMesh& mesh1)
    : runTime(runTime1),
      mesh(mesh1)
{
}

DACheckMesh::~DACheckMesh()
{
}

label DACheckMesh::run() const
{
    /*
    Description:
        Run checkMesh and return meshOK
    
    Output:
        meshOK: 1 means quality passes
    */

    label meshOK=1;

    Info << "Checking mesh quality for time = " << runTime.timeName() << endl;

    label nFailedChecks = checkGeometry(mesh);

    if (nFailedChecks)
    {
        Info << "\nFailed " << nFailedChecks << " mesh checks.\n"
             << endl;
        meshOK = 0;
    }
    else
    {
        Info << "\nMesh OK.\n"
             << endl;
    }

    return meshOK;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //