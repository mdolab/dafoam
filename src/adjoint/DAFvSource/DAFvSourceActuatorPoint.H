/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for ActuatorPoint source

\*---------------------------------------------------------------------------*/

#ifndef DAFvSourceActuatorPoint_H
#define DAFvSourceActuatorPoint_H

#include "DAFvSource.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAFvSourceActuatorPoint Declaration
\*---------------------------------------------------------------------------*/

class DAFvSourceActuatorPoint
    : public DAFvSource
{

protected:

    /// print interval 
    label printIntervalUnsteady_;

public:
    TypeName("actuatorPoint");
    // Constructors

    //- Construct from components
    DAFvSourceActuatorPoint(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAFvSourceActuatorPoint()
    {
    }

    /// compute the FvSource term
    virtual void calcFvSource(volVectorField& fvSource);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
