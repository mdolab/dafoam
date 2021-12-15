/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for moment

\*---------------------------------------------------------------------------*/

#ifndef DAObjFuncMoment_H
#define DAObjFuncMoment_H

#include "DAObjFunc.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAObjFuncMoment Declaration
\*---------------------------------------------------------------------------*/

class DAObjFuncMoment
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
    TypeName("moment");
    // Constructors

    //- Construct from components
    DAObjFuncMoment(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAResidual& daResidual,
        const word objFuncName,
        const word objFuncPart,
        const dictionary& objFuncDict);

    //- Destructor
    virtual ~DAObjFuncMoment()
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
