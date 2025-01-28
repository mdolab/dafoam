/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for location of a select patch, e.g., maxRadius

\*---------------------------------------------------------------------------*/

#ifndef DAFunctionLocation_H
#define DAFunctionLocation_H

#include "DAFunction.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAFunctionLocation Declaration
\*---------------------------------------------------------------------------*/

class DAFunctionLocation
    : public DAFunction
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
    DAFunctionLocation(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const word functionName);

    //- Destructor
    virtual ~DAFunctionLocation()
    {
    }

    /// calculate the value of objective function
    virtual scalar calcFunction();

    void findGlobalSnappedCenter(
        label snappedCenterCellI,
        vector& center);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
