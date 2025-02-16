/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for HeatSource disk source

\*---------------------------------------------------------------------------*/

#ifndef DAFvSourceHeatSource_H
#define DAFvSourceHeatSource_H

#include "DAFvSource.H"
#include "addToRunTimeSelectionTable.H"
#include "DAGlobalVar.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAFvSourceHeatSource Declaration
\*---------------------------------------------------------------------------*/

class DAFvSourceHeatSource
    : public DAFvSource
{

protected:
    /// HashTable that contains lists of cell indices that are within the actuator disk space
    HashTable<labelList> fvSourceCellIndices_;

    HashTable<scalarList> cylinderP1_;
    HashTable<scalarList> cylinderP2_;
    HashTable<scalar> cylinderRadius_;
    HashTable<scalar> power_;
    HashTable<scalar> cylinderEps_;

    /// print interval for primal and adjoint
    label printInterval_;

    /// whether to snap the center to a cell in the mesh if yes the center will move with the mesh
    HashTable<label> snapCenter2Cell_;

    /// the cell index for the center if snapCenter2Cell_ = 1
    HashTable<label> snappedCenterCellI_;

public:
    TypeName("heatSource");
    // Constructors

    //- Construct from components
    DAFvSourceHeatSource(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAFvSourceHeatSource()
    {
    }

    /// compute the FvSource term
    virtual void calcFvSource(volScalarField& fvSource);

    /// Initialize the values for all types of fvSource in DAGlobalVar, including actuatorDiskPars, heatSourcePars, etc
    virtual void initFvSourcePars();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
