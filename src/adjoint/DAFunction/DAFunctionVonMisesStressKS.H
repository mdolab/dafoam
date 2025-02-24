/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for max von Mises stress

\*---------------------------------------------------------------------------*/

#ifndef DAFunctionVonMisesStressKS_H
#define DAFunctionVonMisesStressKS_H

#include "DAFunction.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAFunctionVonMisesStressKS Declaration
\*---------------------------------------------------------------------------*/

class DAFunctionVonMisesStressKS
    : public DAFunction
{

protected:
    /// coefficient for the KS function
    scalar coeffKS_;

public:
    TypeName("vonMisesStressKS");
    // Constructors

    //- Construct from components
    DAFunctionVonMisesStressKS(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const word functionName);

    //- Destructor
    virtual ~DAFunctionVonMisesStressKS()
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
