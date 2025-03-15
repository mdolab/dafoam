/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for thermal variables in CHT.

        Assign the thermal BC values to all of the faces on the conjugate heat 
        transfer patches. 

        We assume the conjugate coupling patches are of mixed type, so we need to assign
        refValue = neighbour near wall temperature
        refGrad = 0
        valueFraction = neighKDeltaCoeffs / ( neighKDeltaCoeffs + myKDeltaCoeffs)

        NOTE: we have two separate variables saved in the thermal array. 
        One is the neighbour near wall temperature and the other is the neighbour kappa/d. 
        So the size of thermal array is 2 * nCouplingFaces
        NOTE: this function can be called by either fluid or solid domain!

        This conjugate heat transfer coupling uses the OpenFOAM's implementation in
        turbulentTemperatureCoupledBaffleMixed.C

\*---------------------------------------------------------------------------*/

#ifndef DAInputThermalCoupling_H
#define DAInputThermalCoupling_H

#include "DAInput.H"
#include "addToRunTimeSelectionTable.H"
#include "mixedFvPatchFields.H"
#include "thermodynamicConstants.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAInputThermalCoupling Declaration
\*---------------------------------------------------------------------------*/

class DAInputThermalCoupling
    : public DAInput
{

protected:
    /// list of patch names for the thermal var
    wordList patches_;

    /// the total face number for all the patches_ owned by this processor
    label size_;

    /// whether this is flow or solid
    word discipline_;

    /// if calculating wallHeatFlux by OpenFOAMs snGrad() or DAFOAM's custom (daCustom) formulation
    word distanceMode_;

public:
    TypeName("thermalCouplingInput");
    // Constructors

    //- Construct from components
    DAInputThermalCoupling(
        const word inputName,
        const word inputType,
        fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAInputThermalCoupling()
    {
    }

    virtual void run(const scalarList& input);

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
