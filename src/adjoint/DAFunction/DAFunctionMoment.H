/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for moment

\*---------------------------------------------------------------------------*/

#ifndef DAFunctionMoment_H
#define DAFunctionMoment_H

#include "DAFunction.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAFunctionMoment Declaration
\*---------------------------------------------------------------------------*/

class DAFunctionMoment
    : public DAFunction
{

protected:
    /// the direction of the moment
    vector momentDir_;

    /// the center of rotation for moment
    vector momentCenter_;

    /// DATurbulenceModel object
    const DATurbulenceModel& daTurb_;

public:
    TypeName("moment");
    // Constructors

    //- Construct from components
    DAFunctionMoment(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const word functionName);

    //- Destructor
    virtual ~DAFunctionMoment()
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
