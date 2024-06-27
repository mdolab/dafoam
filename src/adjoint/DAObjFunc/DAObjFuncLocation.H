/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for location of a select patch, e.g., maxRadius

\*---------------------------------------------------------------------------*/

#ifndef DAObjFuncLocation_H
#define DAObjFuncLocation_H

#include "DAObjFunc.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAObjFuncLocation Declaration
\*---------------------------------------------------------------------------*/

class DAObjFuncLocation
    : public DAObjFunc
{

protected:
    /// coefficient for the KS function
    scalar coeffKS_ = 1.0;

    /// location mode to use
    word mode_;

    /// axis for radius computation
    vector axis_ = {1, 0, 0};

    /// center for radius computation
    vector center_ = {0, 0, 0};

    /// maxRadius patchI
    label maxRPatchI_ = -1;

    /// maxRadius patchI
    label maxRFaceI_ = -1;

    /// whether to snap the center to a cell in the mesh if yes the center will move with the mesh
    label snapCenter2Cell_ = 0;

    /// the cell index for the center if snapCenter2Cell_ = 1
    label snappedCenterCellI_ = -1;

public:
    TypeName("location");
    // Constructors

    //- Construct from components
    DAObjFuncLocation(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAResidual& daResidual,
        const word objFuncName,
        const word objFuncPart,
        const dictionary& objFuncDict);

    //- Destructor
    virtual ~DAObjFuncLocation()
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
