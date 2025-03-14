/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for thermal transfer variables for CHT

        Compute the thermal variables for all of the faces on the conjugate heat 
        transfer patches.

        NOTE: we have two separate variables to assign to the thermal array. 
        One is the near wall temperature and the other is kappa/d. 
        So the size of thermal array is 2 * nCouplingFaces

        NOTE: this function can be called by either fluid or solid domain!

        This conjugate heat transfer coupling uses the OpenFOAM's implementation in
        turbulentTemperatureCoupledBaffleMixed.C

\*---------------------------------------------------------------------------*/

#ifndef DAOutputThermalCoupling_H
#define DAOutputThermalCoupling_H

#include "DAOutput.H"
#include "addToRunTimeSelectionTable.H"
#include "mixedFvPatchFields.H"
#include "thermodynamicConstants.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAOutputThermalCoupling Declaration
\*---------------------------------------------------------------------------*/

class DAOutputThermalCoupling
    : public DAOutput
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
    TypeName("thermalCouplingOutput");
    // Constructors

    //- Construct from components
    DAOutputThermalCoupling(
        const word outputName,
        const word outputType,
        fvMesh& mesh,
        const DAOption& daOption,
        DAModel& daModel,
        const DAIndex& daIndex,
        DAResidual& daResidual,
        UPtrList<DAFunction>& daFunctionList);

    //- Destructor
    virtual ~DAOutputThermalCoupling()
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
