/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for state error L2 norm sum[(state-stateRef)^2]

\*---------------------------------------------------------------------------*/

#ifndef DAObjFunc3D_H
#define DAObjFunc3D_H

#include "DAObjFunc.H"
#include "addToRunTimeSelectionTable.H"
#include <vector>
#include <tuple>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAObjFunc3D Declaration
\*---------------------------------------------------------------------------*/

class DAObjFunc3D
    : public DAObjFunc
{

protected:

    /// name of the state variable to compute the error norm
    word data_;

    word stateName_;

    word stateRefName_; 

    scalar scale_; 
    
    autoPtr<volVectorField> ubDataPtr_;
    
    /// weighted objective function (especially useful when performing weighted sum multi-objective optimisation)
    bool weightedSum_;  // switch 
    scalar weight_; 

    /// DATurbulenceModel object
    const DATurbulenceModel& daTurb_;

public:
    TypeName("average3D");
    // Constructors

    //- Construct from components
    DAObjFunc3D(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAResidual& daResidual,
        const word objFuncName,
        const word objFuncPart,
        const dictionary& objFuncDict);

    //- Destructor
    virtual ~DAObjFunc3D()
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
