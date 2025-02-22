/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for total pressure ratio

\*---------------------------------------------------------------------------*/

#ifndef DAFunctionTotalPressureRatio_H
#define DAFunctionTotalPressureRatio_H

#include "DAFunction.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAFunctionTotalPressureRatio Declaration
\*---------------------------------------------------------------------------*/

class DAFunctionTotalPressureRatio
    : public DAFunction
{

protected:
    /// the area of all inlet patches
    scalar areaSumInlet_ = -9999.0;

    /// the area of all outlet patches
    scalar areaSumOutlet_ = -9999.0;

    /// names of outlet patches
    wordList outletPatches_;

    /// names of inlet patches
    wordList inletPatches_;

    scalar Cp_;

    scalar gamma_;

public:
    TypeName("totalPressureRatio");
    // Constructors

    //- Construct from components
    DAFunctionTotalPressureRatio(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const word functionName);

    //- Destructor
    virtual ~DAFunctionTotalPressureRatio()
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
