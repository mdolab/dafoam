/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for heat flux

\*---------------------------------------------------------------------------*/

#ifndef DAObjFuncWallHeatFlux_H
#define DAObjFuncWallHeatFlux_H

#include "DAObjFunc.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAObjFuncWallHeatFlux Declaration
\*---------------------------------------------------------------------------*/

class DAObjFuncWallHeatFlux
    : public DAObjFunc
{

protected:
#ifdef CompressibleFlow
    /// thermophysical property
    fluidThermo& thermo_;
#endif

    /// Cp used in solid and incompressible heatFlux calculation
    scalar Cp_ = -9999.0;

    /// density used in solid and incompressible heatFlux calculation
    scalar rho_ = -9999.0;

    /// solid thermal diffusivity
    scalar DT_ = -9999.0;

#ifndef SolidDASolver
    /// DATurbulenceModel object
    DATurbulenceModel& daTurb_;
#endif

    /// wall heat flux field
    volScalarField wallHeatFlux_;

    /// the area of all heat flux patches
    scalar areaSum_ = -9999.0;

public:
    TypeName("wallHeatFlux");
    // Constructors

    //- Construct from components
    DAObjFuncWallHeatFlux(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAResidual& daResidual,
        const word objFuncName,
        const word objFuncPart,
        const dictionary& objFuncDict);

    //- Destructor
    virtual ~DAObjFuncWallHeatFlux()
    {
    }

    /// calculate the value of objective function
    virtual void calcObjFunc(
        const labelList& objFuncFaceSources,
        const labelList& objFuncCellSources,
        scalarList& objFuncFaceValues,
        scalarList& objFuncCellValues,
        scalar& objFuncValue);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
