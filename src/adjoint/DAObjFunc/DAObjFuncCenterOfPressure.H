/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for center of pressure

\*---------------------------------------------------------------------------*/

#ifndef DAObjFuncCenterOfPressure_H
#define DAObjFuncCenterOfPressure_H

#include "DAObjFunc.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAObjFuncCenterOfPressure Declaration
\*---------------------------------------------------------------------------*/

class DAObjFuncCenterOfPressure
    : public DAObjFunc
{

protected:

    /// the direction to project the center of pressure onto
    vector axis_;

    /// the direction to project the force onto
    vector forceAxis_;

    /// The point to compute center of pressure about
    vector center_;

public:
    TypeName("centerOfPressure");
    // Constructors

    //- Construct from components
    DAObjFuncCenterOfPressure(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAResidual& daResidual,
        const word objFuncName,
        const word objFuncPart,
        const dictionary& objFuncDict);

    //- Destructor
    virtual ~DAObjFuncCenterOfPressure()
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
