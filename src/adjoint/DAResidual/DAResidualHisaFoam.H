/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for DAHisaFoam

\*---------------------------------------------------------------------------*/

#ifndef DAResidualHisaFoam_H
#define DAResidualHisaFoam_H

#include "DAResidual.H"
#include "addToRunTimeSelectionTable.H"
#include "simpleControl.H"
#include "adjustPhi.H"
#include "thermodynamicConstants.H"
#include "pressureControl.H"
#include "constrainPressure.H"
#include "fluxScheme.H"
#include "solverModule.H"
#include "orthogonalSnGrad.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAResidualHisaFoam Declaration
\*---------------------------------------------------------------------------*/

class DAResidualHisaFoam
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
    //@}

    // fluxes
    surfaceScalarField& phi_;
    surfaceVectorField& phiUp_;
    surfaceScalarField& phiEp_;
    surfaceVectorField& Up_;
    volScalarField& rho_;
    volVectorField& rhoU_;
    volScalarField& rhoE_;
    volScalarField& psi_;

    fluxScheme& fluxScheme_;

    solverModule& solverModule_;

    /// thermophysical property
    fluidThermo& thermo_;

    volScalarField& he_;

    compressible::turbulenceModel& turbulence_;

    /// \name These are constants to update the intermediate variables
    //@{
    scalar molWeight_;
    scalar Cp_;
    //@}

public:
    TypeName("DAHisaFoam");
    // Constructors

    //- Construct from components
    DAResidualHisaFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAResidualHisaFoam()
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

    /// remove the off diagonal elements for the tensor field
    void removeOffDiagonals(volTensorField&);

    /// calculate the flux using laxFriedrichs scheme for PC mat
    void calcFluxLaxFriedrichs(
        surfaceScalarField& phi,
        surfaceVectorField& phiUp,
        surfaceScalarField& phiEp,
        surfaceVectorField& Up);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
