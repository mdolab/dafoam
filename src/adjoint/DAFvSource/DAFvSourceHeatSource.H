/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for HeatSource disk source

\*---------------------------------------------------------------------------*/

#ifndef DAFvSourceHeatSource_H
#define DAFvSourceHeatSource_H

#include "DAFvSource.H"
#include "addToRunTimeSelectionTable.H"

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

    /// calculate DAFvSourceHeatSource::fvSourceCellIndices_
    void calcFvSourceCellIndices(const word sourceName, HashTable<labelList>& fvSourceCellIndices);

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
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
