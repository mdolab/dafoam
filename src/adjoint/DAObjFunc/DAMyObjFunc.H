/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for state error L2 norm sum[(state-stateRef)^2]

\*---------------------------------------------------------------------------*/

#ifndef DAMyObjFunc_H
#define DAMyObjFunc_H

#include "DAObjFunc.H"
#include "addToRunTimeSelectionTable.H"
#include <vector>
#include <tuple>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAMyObjFunc Declaration
\*---------------------------------------------------------------------------*/

class DAMyObjFunc
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
    TypeName("naiveAverage");
    // Constructors

    //- Construct from components
    DAMyObjFunc(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAResidual& daResidual,
        const word objFuncName,
        const word objFuncPart,
        const dictionary& objFuncDict);

    //- Destructor
    virtual ~DAMyObjFunc()
    {
    }

    /// calculate the value of objective function
    virtual void calcObjFunc(
        const labelList& objFuncFaceSources,
        const labelList& objFuncCellSources,
        scalarList& objFuncFaceValues,
        scalarList& objFuncCellValues,
        scalar& objFuncValue);
        
   List<List<label>> points;
   scalarList uBoxAvg;
   scalarList vBoxAvg;
   scalarList wBoxAvg;

};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
