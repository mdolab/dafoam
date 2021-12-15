/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

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

#ifdef IncompressibleFlow
    scalar Cp_ = -9999.0;
#endif

    /// DATurbulenceModel object
    DATurbulenceModel& daTurb_;

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
