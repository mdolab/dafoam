/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for total pressure ratio

\*---------------------------------------------------------------------------*/

#ifndef DAObjFuncTotalPressureRatio_H
#define DAObjFuncTotalPressureRatio_H

#include "DAObjFunc.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAObjFuncTotalPressureRatio Declaration
\*---------------------------------------------------------------------------*/

class DAObjFuncTotalPressureRatio
    : public DAObjFunc
{

protected:

    /// the area of all inlet patches
    scalar areaSumInlet_ = -9999.0;

    /// the area of all outlet patches
    scalar areaSumOutlet_ = -9999.0;

    /// names of outlet patches
    wordList outletPatches_;

    /// names of inlet patches
    wordList inletPatches_;

    scalar TPInRef_ = 0.0;

    scalar TPOutRef_ = 0.0;

    scalar Cp_;

    scalar gamma_;

public:
    TypeName("totalPressureRatio");
    // Constructors

    //- Construct from components
    DAObjFuncTotalPressureRatio(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAResidual& daResidual,
        const word objFuncName,
        const word objFuncPart,
        const dictionary& objFuncDict);

    //- Destructor
    virtual ~DAObjFuncTotalPressureRatio()
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
