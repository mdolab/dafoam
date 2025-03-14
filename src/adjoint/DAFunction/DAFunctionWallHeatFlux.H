/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for heat flux

\*---------------------------------------------------------------------------*/

#ifndef DAFunctionWallHeatFlux_H
#define DAFunctionWallHeatFlux_H

#include "DAFunction.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAFunctionWallHeatFlux Declaration
\*---------------------------------------------------------------------------*/

class DAFunctionWallHeatFlux
    : public DAFunction
{

protected:
    /// Cp used in incompressible heatFlux calculation
    scalar Cp_ = -9999.0;

    /// thermal conductivity for solid heatFlux calculation
    scalar k_ = -9999.0;

    /// wall heat flux field
    volScalarField wallHeatFlux_;

    /// the area of all heat flux patches
    scalar areaSum_ = -9999.0;

    /// if calculating flux per unit area or total, which mode to use
    bool calcMode_;

    /// if calculating wallHeatFlux by OpenFOAMs snGrad() or DAFOAM's custom (daCustom) formulation
    word distanceMode_;

public:
    TypeName("wallHeatFlux");
    // Constructors

    //- Construct from components
    DAFunctionWallHeatFlux(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const word functionName);

    //- Destructor
    virtual ~DAFunctionWallHeatFlux()
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
