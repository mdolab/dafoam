/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v5

    Description:
        Child class for the max value of a field 

\*---------------------------------------------------------------------------*/

#ifndef DAFunctionFieldMax_H
#define DAFunctionFieldMax_H

#include "DAFunction.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAFunctionFieldMax Declaration
\*---------------------------------------------------------------------------*/

class DAFunctionFieldMax
    : public DAFunction
{

protected:
    /// name of the variable
    word fieldName_;

    /// type of the variable
    word fieldType_;

    /// if it is a vector, prescribe the indices
    labelList indices_;

public:
    TypeName("fieldMax");
    // Constructors

    //- Construct from components
    DAFunctionFieldMax(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const word functionName);

    //- Destructor
    virtual ~DAFunctionFieldMax()
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
