/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for mesh quality with KS function

\*---------------------------------------------------------------------------*/

#ifndef DAObjFuncMeshQualityKS_H
#define DAObjFuncMeshQualityKS_H

#include "DAObjFunc.H"
#include "polyMeshTools.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAObjFuncMeshQualityKS Declaration
\*---------------------------------------------------------------------------*/

class DAObjFuncMeshQualityKS
    : public DAObjFunc
{

protected:

    /// coefficient for the KS function
    scalar coeffKS_;

    /// which mesh quality metric to use
    word metric_;


public:
    TypeName("meshQualityKS");
    // Constructors

    //- Construct from components
    DAObjFuncMeshQualityKS(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAResidual& daResidual,
        const word objFuncName,
        const word objFuncPart,
        const dictionary& objFuncDict);

    //- Destructor
    virtual ~DAObjFuncMeshQualityKS()
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
