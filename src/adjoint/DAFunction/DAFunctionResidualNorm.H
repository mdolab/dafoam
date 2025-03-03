/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for residual norm

\*---------------------------------------------------------------------------*/

#ifndef DAFunctionResidualNorm_H
#define DAFunctionResidualNorm_H

#include "DAFunction.H"
#include "addToRunTimeSelectionTable.H"
#include "DAResidual.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAFunctionResidualNorm Declaration
\*---------------------------------------------------------------------------*/

class DAFunctionResidualNorm
    : public DAFunction
{

protected:
    /// weights of the residuals
    HashTable<scalar> resWeight_;

    /// the stateInfo_ list from DAStateInfo object
    HashTable<wordList> stateInfo_;

public:
    TypeName("residualNorm");
    // Constructors

    //- Construct from components
    DAFunctionResidualNorm(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const word functionName);

    //- Destructor
    virtual ~DAFunctionResidualNorm()
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
