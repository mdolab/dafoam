/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for moment

\*---------------------------------------------------------------------------*/

#ifndef DAObjFuncCenterPressure_H
#define DAObjFuncCenterPressure_H

#include "DAObjFunc.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAObjFuncCenterPressure Declaration
\*---------------------------------------------------------------------------*/

class DAObjFuncCenterPressure
    : public DAObjFunc
{

protected:
    /// the direction of the moment
    vector pressureDir_;

    /// the center of rotation for moment
    vector pressureOrigin_;

    /// DATurbulenceModel object
    //const DATurbulenceModel& daTurb_;

public:
    TypeName("centerPressure");
    // Constructors

    //- Construct from components
    DAObjFuncCenterPressure(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAResidual& daResidual,
        const word objFuncName,
        const word objFuncPart,
        const dictionary& objFuncDict);

    //- Destructor
    virtual ~DAObjFuncCenterPressure()
    {
    }

    /// calculate the value of objective function
    virtual void calcObjFunc(
        const labelList& objFuncFaceSources,
        //const labelList& objFuncCellSources,
        //scalarList& objFuncFaceValues,
        //scalarList& objFuncCellValues,
        scalar& objFuncValue);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
