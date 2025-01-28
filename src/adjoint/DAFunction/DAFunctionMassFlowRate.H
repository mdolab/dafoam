/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for mass flow rate

\*---------------------------------------------------------------------------*/

#ifndef DAFunctionMassFlowRate_H
#define DAFunctionMassFlowRate_H

#include "DAFunction.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAFunctionMassFlowRate Declaration
\*---------------------------------------------------------------------------*/

class DAFunctionMassFlowRate
    : public DAFunction
{

protected:
    /// DATurbulenceModel object
    DATurbulenceModel& daTurb_;

public:
    TypeName("massFlowRate");
    // Constructors

    //- Construct from components
    DAFunctionMassFlowRate(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const word functionName);

    //- Destructor
    virtual ~DAFunctionMassFlowRate()
    {
    }

    /// calculate the value of objective function
    virtual scalar calcFunction();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
