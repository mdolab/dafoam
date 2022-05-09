/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for variable volume sum

\*---------------------------------------------------------------------------*/

#ifndef DAObjFuncVariableVolSum_H
#define DAObjFuncVariableVolSum_H

#include "DAObjFunc.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAObjFuncVariableVolSum Declaration
\*---------------------------------------------------------------------------*/

class DAObjFuncVariableVolSum
    : public DAObjFunc
{

protected:

    /// name of the variable
    word varName_;

    /// type of the variable either vector or scalar
    word varType_;

    /// if vector which element?
    label component_;

    /// whether to take a square of the variable
    label isSquare_;

public:
    TypeName("variableVolSum");
    // Constructors

    //- Construct from components
    DAObjFuncVariableVolSum(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAResidual& daResidual,
        const word objFuncName,
        const word objFuncPart,
        const dictionary& objFuncDict);

    //- Destructor
    virtual ~DAObjFuncVariableVolSum()
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
