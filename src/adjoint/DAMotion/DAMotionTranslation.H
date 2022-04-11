/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for translation oscillation motion

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
    /// amplitude of the translation oscillation
    scalar amplitude_;

    /// frequency of the translation oscillation
    scalar frequency_;

    /// phase of the translation oscillation
    scalar phase_;

    /// direction of the translation oscillation
    vector direction_ = vector::zero;

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
