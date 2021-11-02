/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for translation motion

\*---------------------------------------------------------------------------*/

#ifndef DAMotionTranslation_H
#define DAMotionTranslation_H

#include "DAMotion.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAMotionTranslation Declaration
\*---------------------------------------------------------------------------*/

class DAMotionTranslation
    : public DAMotion
{

protected:

    scalar amplitude_;

    scalar frequency_;

    scalar phase_;

    vector direction_ = vector::zero;

    wordList patchNames_;

public:
    TypeName("translation");
    // Constructors

    //- Construct from components
    DAMotionTranslation(
        const dynamicFvMesh& mesh,
        const DAOption& daOption);

    //- Destructor
    virtual ~DAMotionTranslation()
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
