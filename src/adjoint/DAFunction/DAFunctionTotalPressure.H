/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for total pressure

\*---------------------------------------------------------------------------*/

#ifndef DAFunctionTotalPressure_H
#define DAFunctionTotalPressure_H

#include "DAFunction.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAFunctionTotalPressure Declaration
\*---------------------------------------------------------------------------*/

class DAFunctionTotalPressure
    : public DAFunction
{

protected:
    /// DATurbulenceModel object
    DATurbulenceModel& daTurb_;

    /// density
    volScalarField rho_;

    /// the area of all total pressure patches
    scalar areaSum_ = -9999.0;

public:
    TypeName("totalPressure");
    // Constructors

    //- Construct from components
    DAFunctionTotalPressure(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const word functionName);

    //- Destructor
    virtual ~DAFunctionTotalPressure()
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
