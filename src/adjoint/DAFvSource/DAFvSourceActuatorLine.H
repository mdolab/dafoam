/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for ActuatorLine source

\*---------------------------------------------------------------------------*/

#ifndef DAFvSourceActuatorLine_H
#define DAFvSourceActuatorLine_H

#include "DAFvSource.H"
#include "addToRunTimeSelectionTable.H"
#include "DAGlobalVar.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAFvSourceActuatorLine Declaration
\*---------------------------------------------------------------------------*/

class DAFvSourceActuatorLine
    : public DAFvSource
{

protected:
    /// print interval
    label printIntervalUnsteady_;

public:
    TypeName("actuatorLine");
    // Constructors

    //- Construct from components
    DAFvSourceActuatorLine(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAFvSourceActuatorLine()
    {
    }

    /// compute the FvSource term
    virtual void calcFvSource(volVectorField& fvSource);

    /// Initialize the values for all types of fvSource in DAGlobalVar, including actuatorDiskPars, heatSourcePars, etc
    virtual void initFvSourcePars();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
