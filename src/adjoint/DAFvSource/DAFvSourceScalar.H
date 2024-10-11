/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for passive scalar 

\*---------------------------------------------------------------------------*/

#ifndef DAFvSourceScalar_H
#define DAFvSourceScalar_H

#include "DAFvSource.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAFvSourceScalar Declaration
\*---------------------------------------------------------------------------*/

class DAFvSourceScalar
    : public DAFvSource
{

protected:
    /// HashTable that contains lists of cell indices that are within the scalar source space
    HashTable<labelList> fvSourceCellIndices_;

    HashTable<scalarList> min_;
    HashTable<scalarList> max_;
    //HashTable<scalar> cylinderRadius_;
    HashTable<scalar> power_;
    HashTable<scalar> cylinderEps_;

    /// print interval for primal and adjoint
    label printInterval_;

    /// whether to snap the center to a cell in the mesh if yes the center will move with the mesh
    HashTable<label> snapCenter2Cell_;

    /// the cell index for the center if snapCenter2Cell_ = 1
    HashTable<label> snappedCenterCellI_;

public:
    TypeName("passiveScalar");
    // Constructors

    //- Construct from components
    DAFvSourceScalar(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAFvSourceScalar()
    {
    }

    /// compute the FvSource term
    virtual void calcFvSource(volScalarField& fvSource);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
