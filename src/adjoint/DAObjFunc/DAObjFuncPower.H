/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for moment

\*---------------------------------------------------------------------------*/

#ifndef DAObjFuncPower_H
#define DAObjFuncPower_H

#include "DAObjFunc.H"
#include "addToRunTimeSelectionTable.H"
#include "IOMRFZoneListDF.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAObjFuncPower Declaration
\*---------------------------------------------------------------------------*/

class DAObjFuncPower
    : public DAObjFunc
{

protected:
    /// the direction of the moment
    vector momentDir_;

    /// the center of rotation for moment
    vector momentCenter_;

    /// DATurbulenceModel object
    const DATurbulenceModel& daTurb_;

public:
    TypeName("power");
    // Constructors

    //- Construct from components
    DAObjFuncPower(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAResidual& daResidual,
        const word objFuncName,
        const word objFuncPart,
        const dictionary& objFuncDict);

    //- Destructor
    virtual ~DAObjFuncPower()
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
