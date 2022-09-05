/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for state error L2 norm sum[(state-stateRef)^2]

\*---------------------------------------------------------------------------*/

#ifndef DAObjFuncFieldInversion_H
#define DAObjFuncFieldInversion_H

#include "DAObjFunc.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAObjFuncFieldInversion Declaration
\*---------------------------------------------------------------------------*/

class DAObjFuncFieldInversion
    : public DAObjFunc
{

protected:

    /// name of the state variable to compute the error norm
    word data_;

    word stateName_;

    word stateRefName_; 

    scalar scale_; 

    /// weighted objective function (especially useful when performing weighted sum multi-objective optimisation)
    bool weightedSum_;  // switch 
    scalar weight_; 

    /// DATurbulenceModel object
    const DATurbulenceModel& daTurb_;

public:
    TypeName("fieldInversion");
    // Constructors

    //- Construct from components
    DAObjFuncFieldInversion(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAResidual& daResidual,
        const word objFuncName,
        const word objFuncPart,
        const dictionary& objFuncDict);

    //- Destructor
    virtual ~DAObjFuncFieldInversion()
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
