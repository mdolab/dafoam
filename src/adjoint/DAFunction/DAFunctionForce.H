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
#include "DAGlobalVar.H"

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

    /// if dynamically adjusting the angle what mode to use
    word dirMode_;

    /// flowAxisIndex_ for the alpha design variable with tan(U_normal/U_flow)
    label flowAxisIndex_;

    /// normalAxisIndex_ for the alpha design variable with tan(U_normal/U_flow)
    label normalAxisIndex_;

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
        const word functionName);

    //- Destructor
    virtual ~DAFunctionForce()
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
