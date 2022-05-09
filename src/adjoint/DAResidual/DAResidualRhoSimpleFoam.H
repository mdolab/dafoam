/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for DARhoSimpleFoam

\*---------------------------------------------------------------------------*/

#ifndef DAResidualRhoSimpleFoam_H
#define DAResidualRhoSimpleFoam_H

#include "DAResidual.H"
#include "addToRunTimeSelectionTable.H"
#include "simpleControl.H"
#include "adjustPhi.H"
#include "thermodynamicConstants.H"
#include "pressureControl.H"
#include "constrainPressure.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAResidualRhoSimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class DAResidualRhoSimpleFoam
    : public DAResidual
{

protected:
    /// \name These are state variables, state residuals, and partial derivatives
    //@{
    volVectorField& U_;
    volVectorField URes_;

    volScalarField& p_;
    volScalarField pRes_;

    volScalarField& T_;
    volScalarField TRes_;

    surfaceScalarField& phi_;
    surfaceScalarField phiRes_;
    //@}

    /// fvSource term
    volVectorField& fvSource_;

    /// fvSource term for the energy equation
    volScalarField& fvSourceEnergy_;

    /// fvOptions term
    fv::options& fvOptions_;

    /// thermophysical property
    fluidThermo& thermo_;

    /// \name These are intermediate variables
    //@{
    volScalarField& he_;
    volScalarField& rho_;
    volScalarField& alphat_;
    volScalarField& psi_;
    //@}

    /// DATurbulenceModel object
    DATurbulenceModel& daTurb_;

    /// simpleControl object which will be initialized in this class
    simpleControl simple_;

    /// pressureControl object to set ref pressure
    pressureControl pressureControl_;

    /// Multiple Reference Frame 
    IOMRFZoneListDF& MRF_;

    /// \name These are constants to update the intermediate variables
    //@{
    scalar molWeight_;
    scalar Cp_;
    //@}

    /// whether to have fvSource term
    label hasFvSource_ = 0;

public:
    TypeName("DARhoSimpleFoam");
    // Constructors

    //- Construct from components
    DAResidualRhoSimpleFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAResidualRhoSimpleFoam()
    {
    }

    // Members

    /// clear the members
    virtual void clear();

    /// compute residual
    virtual void calcResiduals(const dictionary& options);

    /// update any intermediate variables that are dependent on state variables and are used in calcResiduals
    virtual void updateIntermediateVariables();

    /// update the boundary condition for all the states in the selected solver
    virtual void correctBoundaryConditions();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
