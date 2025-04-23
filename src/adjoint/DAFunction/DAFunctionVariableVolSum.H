/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for variable volume sum

\*---------------------------------------------------------------------------*/

#ifndef DAFunctionVariableVolSum_H
#define DAFunctionVariableVolSum_H

#include "DAFunction.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAFunctionVariableVolSum Declaration
\*---------------------------------------------------------------------------*/

class DAFunctionVariableVolSum
    : public DAFunction
{

protected:
    /// name of the variable
    word varName_;

    /// type of the variable either vector or scalar
    word varType_;

    /// if vector which element/index?
    label index_;

    /// whether to take a square of the variable
    label isSquare_;

    /// whether to normalize the volume sum with the total volume
    label divByTotalVol_;

    /// whether to multiply the variable by the volume
    label multiplyVol_;

public:
    TypeName("variableVolSum");
    // Constructors

    //- Construct from components
    DAFunctionVariableVolSum(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const word functionName);

    //- Destructor
    virtual ~DAFunctionVariableVolSum()
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
