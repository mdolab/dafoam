/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for dummy (no) motion

\*---------------------------------------------------------------------------*/

#ifndef DAMotionDummy_H
#define DAMotionDummy_H

#include "DAMotion.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAMotionDummy Declaration
\*---------------------------------------------------------------------------*/

class DAMotionDummy
    : public DAMotion
{

protected:

public:
    TypeName("dummy");
    // Constructors

    //- Construct from components
    DAMotionDummy(
        const dynamicFvMesh& mesh,
        const DAOption& daOption);

    //- Destructor
    virtual ~DAMotionDummy()
    {
    }

    /// update the cell displacement
    virtual void correct();

};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
