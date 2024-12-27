/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for force 

\*---------------------------------------------------------------------------*/

#ifndef DAFunctionForce_H
#define DAFunctionForce_H

#include "DAFunction.H"
#include "addToRunTimeSelectionTable.H"
#include "mixedFvPatchFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAFunctionForce Declaration
\*---------------------------------------------------------------------------*/

class DAFunctionForce
    : public DAFunction
{

protected:
    /// the direction of the force
    vector forceDir_;

    /// DATurbulenceModel object
    const DATurbulenceModel& daTurb_;

public:
    TypeName("force");
    // Constructors

    //- Construct from components
    DAFunctionForce(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const word functionName,
        const word functionPart,
        const dictionary& functionDict);

    //- Destructor
    virtual ~DAFunctionForce()
    {
    }

    /// calculate the value of objective function
    virtual void calcFunction(scalar& functionValue);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
