/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for total pressure

\*---------------------------------------------------------------------------*/

#ifndef DAObjFuncTotalPressure_H
#define DAObjFuncTotalPressure_H

#include "DAObjFunc.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAObjFuncTotalPressure Declaration
\*---------------------------------------------------------------------------*/

class DAObjFuncTotalPressure
    : public DAObjFunc
{

protected:

    /// DATurbulenceModel object
    DATurbulenceModel& daTurb_;

    /// density 
    volScalarField rho_;

    /// the area of all total pressure patches
    scalar areaSum_ = -9999.0;

public:
    TypeName("totalPressure");
    // Constructors

    //- Construct from components
    DAObjFuncTotalPressure(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAResidual& daResidual,
        const word objFuncName,
        const word objFuncPart,
        const dictionary& objFuncDict);

    //- Destructor
    virtual ~DAObjFuncTotalPressure()
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
