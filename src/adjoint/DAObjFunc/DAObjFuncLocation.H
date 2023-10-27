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

    vector center_ = {0, 0, 0};

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
