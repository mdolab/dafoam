/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for approximated max variable on patches using the KS function

\*---------------------------------------------------------------------------*/

#ifndef DAObjFuncPatchMaxKS_H
#define DAObjFuncPatchMaxKS_H

#include "DAObjFunc.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAObjFuncPatchMaxKS Declaration
\*---------------------------------------------------------------------------*/

class DAObjFuncPatchMaxKS
    : public DAObjFunc
{

protected:

    /// name of the variable
    word varName_;

    /// type of the variable either vector or scalar
    word varType_;

    /// if vector which element?
    label component_;

    /// the KS coefficient
    scalar coeffKS_;

public:
    TypeName("patchMaxKS");
    // Constructors

    //- Construct from components
    DAObjFuncPatchMaxKS(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAResidual& daResidual,
        const word objFuncName,
        const word objFuncPart,
        const dictionary& objFuncDict);

    //- Destructor
    virtual ~DAObjFuncPatchMaxKS()
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
