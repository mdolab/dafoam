/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for force transfer in fluid-structure interaction cases

\*---------------------------------------------------------------------------*/

#ifndef DAOutputForceCoupling_H
#define DAOutputForceCoupling_H

#include "DAOutput.H"
#include "addToRunTimeSelectionTable.H"
#include "volPointInterpolation.H"
#include "SortableList.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAOutputForceCoupling Declaration
\*---------------------------------------------------------------------------*/

class DAOutputForceCoupling
    : public DAOutput
{

protected:
    /// list of patch names for the thermal var
    wordList patches_;

    /// the total face number for all the patches_ owned by this processor
    label size_;

    /// the reference pressure
    scalar pRef_;

public:
    TypeName("forceCouplingOutput");
    // Constructors

    //- Construct from components
    DAOutputForceCoupling(
        const word outputName,
        const word outputType,
        fvMesh& mesh,
        const DAOption& daOption,
        DAModel& daModel,
        const DAIndex& daIndex,
        DAResidual& daResidual,
        UPtrList<DAFunction>& daFunctionList);

    //- Destructor
    virtual ~DAOutputForceCoupling()
    {
    }

    virtual void run(scalarList& output);

    virtual label size()
    {
        return size_;
    }

    virtual label distributed()
    {
        return 1;
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
