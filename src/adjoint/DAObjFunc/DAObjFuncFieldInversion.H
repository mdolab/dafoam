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
    word stateName_;

    /// name of the reference state variable to compute the error norm
    word stateRefName_;

    /// state variable type: can be either scalar or vector
    word stateType_;

    /// type of data available for training: can be surface or volume
    word varTypeFieldInversion_;

    /// scaling factor for the objective function
    scalar scale_; 

    /// list of pactches for computing varaibles such as skin friction if using surface data
    wordList patchNames_; 
    
    /// direction vector if using aerodynamic coefficient data 
    vector forceDir_;

    /// vector to decide the component of wallShearStress vector used for calculating surfaceFriction
    vector wssDir_;

    /// the lift, or drag coefficient from data
    scalar aeroCoeffRef_; 

    /// the Cell coordinates of location where pRef value should be taken from
    bool nonZeroPRefFlag_; 
    vector pRefCoords_;
    scalar pRef_;

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
