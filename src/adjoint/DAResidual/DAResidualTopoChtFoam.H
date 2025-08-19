/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for DATopoChtFoam

\*---------------------------------------------------------------------------*/

#ifndef DAResidualTopoChtFoam_H
#define DAResidualTopoChtFoam_H

#include "DAResidual.H"
#include "addToRunTimeSelectionTable.H"
#include "simpleControl.H"
#include "adjustPhi.H"
#include "constrainPressure.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAResidualTopoChtFoam Declaration
\*---------------------------------------------------------------------------*/

class DAResidualTopoChtFoam
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

    volScalarField& diffusivityFluid_;

    volScalarField& diffusivity_;

    /// alpha porosity term
    volScalarField& alphaPorosity_;

    /// TO design variable
    volScalarField& eta_;

    /// fvSource term
    volVectorField& fvSource_;

    /// DATurbulenceModel object
    DATurbulenceModel& daTurb_;

    /// simpleControl object which will be initialized in this class
    simpleControl simple_;

    /// Multiple Reference Frame
    IOMRFZoneListDF& MRF_;

    /// whether to have fvSource term
    label hasFvSource_ = 0;

    scalar Pr_;

    scalar Prt_;

    scalar alphaPorosityFluid_;
    scalar alphaPorositySolid_;
    scalar diffusivitySolid_;
    scalar pCoeff_;

public:
    TypeName("DATopoChtFoam");
    // Constructors

    //- Construct from components
    DAResidualTopoChtFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAResidualTopoChtFoam()
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
