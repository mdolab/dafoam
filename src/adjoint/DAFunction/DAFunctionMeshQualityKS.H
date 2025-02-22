/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for mesh quality with KS function

\*---------------------------------------------------------------------------*/

#ifndef DAFunctionMeshQualityKS_H
#define DAFunctionMeshQualityKS_H

#include "DAFunction.H"
#include "polyMeshTools.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAFunctionMeshQualityKS Declaration
\*---------------------------------------------------------------------------*/

class DAFunctionMeshQualityKS
    : public DAFunction
{

protected:
    /// coefficient for the KS function
    scalar coeffKS_;

    /// which mesh quality metric to use
    word metric_;

    /// whether to include the processor patches for mesh quality calculation
    label includeProcPatches_;

    /// a list of included faces for the mesh quality calculation
    labelList includeFaceList_;

public:
    TypeName("meshQualityKS");
    // Constructors

    //- Construct from components
    DAFunctionMeshQualityKS(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const word functionName);

    //- Destructor
    virtual ~DAFunctionMeshQualityKS()
    {
    }

    /// calculate the value of objective function
    virtual scalar calcFunction();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
