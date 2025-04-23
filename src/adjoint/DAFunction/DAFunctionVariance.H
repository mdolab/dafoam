/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for variance

\*---------------------------------------------------------------------------*/

#ifndef DAFunctionVariance_H
#define DAFunctionVariance_H

#include "DAFunction.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAFunctionVariance Declaration
\*---------------------------------------------------------------------------*/

class DAFunctionVariance
    : public DAFunction
{

protected:
    /// name of the variable
    word varName_;

    /// data mode options are probePoint surface or field
    word mode_;

    /// type of the variable either vector or scalar
    word varType_;

    /// components/indices of the vector variable
    List<label> indices_;

    /// cell index of the probe points
    labelList probeCellIndex_;

    /// coordinates of the probe points
    List<List<scalar>> probePointCoords_;

    /// what probe point method to use
    word probePointMethod_;

    /// names of the surfaces for ref data
    List<word> surfaceNames_;

    /// probe value
    List<List<scalar>> refValue_;

    /// total number of reference points
    label nRefPoints_ = 0;

    /// upper bound to be considered as non-data
    scalar varUpperBound_;

    /// whether we find the reference data
    label isRefData_;

    /// whether the ref data is time dependent if yes we need data in all time folders otherwise get it from the 0 folder
    label timeDependentRefData_;

    /// DATurbulenceModel object
    DATurbulenceModel& daTurb_;

    /// Cp used in incompressible heatFlux calculation
    scalar Cp_ = -9999.0;

public:
    TypeName("variance");
    // Constructors

    //- Construct from components
    DAFunctionVariance(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const word functionName);

    //- Destructor
    virtual ~DAFunctionVariance()
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
