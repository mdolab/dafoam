/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for coupled translation motion (FSI)
        with the Mass-spring-damper model
        
        M * a + C * V + K * y = F

        where M is the mass, a is the acceleration, V is the velocity, 
        C is the damping coefficient, K is the spring constant, y is 
        the displacement, and F is the external force

        To solve this equation, we can break it into two ODEs

        y1 = y;   y2 = dy/dt

        dy1/dt = V = y2
        dy2/dt = a = (F - C * y2 - K * y1) / M

        We can then use the RK2 method to solve the above two ODEs

\*---------------------------------------------------------------------------*/

#ifndef DAMotionTranslationCoupled_H
#define DAMotionTranslationCoupled_H

#include "DAMotion.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAMotionTranslationCoupled Declaration
\*---------------------------------------------------------------------------*/

class DAMotionTranslationCoupled
    : public DAMotion
{

protected:
    /// mass of the object
    scalar M_;

    /// spring damping coefficient
    scalar C_;

    /// spring stiffness constant
    scalar K_;
    
    /// scaling factor for force
    scalar forceScale_;

    /// spring oscillation direction
    vector direction_ = vector::zero;

    /// location for the current step
    scalar y_;

    /// velocity for the current step
    scalar V_;

    /// location from the previous step
    scalar y0_;

    /// velocity from the previous step
    scalar V0_;

    /// the simulation time based on dT and it should be same as the runtime
    scalar t_;

public:
    TypeName("translationCoupled");
    // Constructors

    //- Construct from components
    DAMotionTranslationCoupled(
        const dynamicFvMesh& mesh,
        const DAOption& daOption);

    //- Destructor
    virtual ~DAMotionTranslationCoupled()
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
