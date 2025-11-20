/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for state error L2 norm sum[(state-stateRef)^2]

\*---------------------------------------------------------------------------*/

#ifndef DAObjFuncfvSource_H
#define DAObjFuncfvSource_H

#include "DAObjFunc.H"
#include "addToRunTimeSelectionTable.H"
#include "fvCFD.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAObjFuncfvSource Declaration
\*---------------------------------------------------------------------------*/

class DAObjFuncfvSource
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
    TypeName("sourceTerm");
    // Constructors

    //- Construct from components
    DAObjFuncfvSource(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAResidual& daResidual,
        const word objFuncName,
        const word objFuncPart,
        const dictionary& objFuncDict);

    //- Destructor
    virtual ~DAObjFuncfvSource()
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
