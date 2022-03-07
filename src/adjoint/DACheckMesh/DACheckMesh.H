/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Check mesh quality, modified from the checkMesh utility

\*---------------------------------------------------------------------------*/

#ifndef DACheckMesh_H
#define DACheckMesh_H

#include "DAOption.H"
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

    /// DAOption object
    const DAOption& daOption_;

    /// Time object
    const Time& runTime;

    /// fvMesh object
    const fvMesh& mesh;

    /// max aspect ratio
    scalar maxAspectRatio_ = 1000.0;

    /// max face non orthogonal angle in degree
    scalar maxNonOrth_ = 70.0;

    /// max skewness
    scalar maxSkewness_ = 4.0;

    /// max number of incorrectly oriented faces
    label maxIncorrectlyOrientedFaces_ = 0;

    /// surface writer pointer for failed mesh
    autoPtr<surfaceWriter> surfWriter;

    /// set writer pointer for failed mesh
    autoPtr<writer<scalar>> setWriter;

public:
    // Constructors

    /// Construct from components
    DACheckMesh(
        const DAOption& daOption,
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
