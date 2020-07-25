/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Check mesh quality, modified from the checkMesh utility

\*---------------------------------------------------------------------------*/

#ifndef DACheckMesh_H
#define DACheckMesh_H

#include "Time.H"
#include "fvMesh.H"
#include "IOdictionary.H"
#include "checkGeometry.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DACheckMesh Declaration
\*---------------------------------------------------------------------------*/

class DACheckMesh
{

private:
    /// Disallow default bitwise copy construct
    DACheckMesh(const DACheckMesh&);

    /// Disallow default bitwise assignment
    void operator=(const DACheckMesh&);

    /// Time object
    const Time& runTime;
    
    /// fvMesh object
    const fvMesh& mesh;

public:
    // Constructors

    /// Construct from components
    DACheckMesh(
        const Time& runTime1,
        const fvMesh& mesh1);

    /// Destructor
    virtual ~DACheckMesh();

    /// run meshCheck and return meshOK
    label run() const;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
