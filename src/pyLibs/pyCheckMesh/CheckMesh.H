/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2.0

    Description:
        Python wrapper library for checkMesh

\*---------------------------------------------------------------------------*/

#ifndef CheckMesh_H
#define CheckMesh_H

#include "argList.H"
#include "timeSelector.H"
#include "Time.H"
#include "fvMesh.H"
#include "globalMeshData.H"
#include "surfaceWriter.H"
#include "vtkSetWriter.H"
#include "IOdictionary.H"

#include "checkTools.H"
#include "checkTopology.H"
#include "checkGeometry.H"
#include "checkMeshQuality.H"
#include "writeFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class CheckMesh Declaration
\*---------------------------------------------------------------------------*/

class CheckMesh
{

private:

    /// Disallow default bitwise copy construct
    CheckMesh(const CheckMesh&);

    /// Disallow default bitwise assignment
    void operator=(const CheckMesh&);

    /// all the arguments
    char* argsAll_;

    /// mesh checks pass or not
    label meshOK_;

public:

    // Constructors

    /// Construct from components
    CheckMesh(char* argsAll);

    /// Destructor
    virtual ~CheckMesh();

    /// run meshCheck and return meshOK
    label run();

};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

