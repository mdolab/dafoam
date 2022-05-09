/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for DASimpleFoam

\*---------------------------------------------------------------------------*/

#ifndef DAResidualSimpleFoam_H
#define DAResidualSimpleFoam_H

#include "DAResidual.H"
#include "addToRunTimeSelectionTable.H"
#include "simpleControl.H"
#include "adjustPhi.H"
#include "constrainPressure.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAResidualSimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class DAResidualSimpleFoam
    : public DAResidual
{
protected:
    /// \name These are state variables, state residuals, and partial derivatives
    //@{
    volVectorField& U_;
    volVectorField URes_;

    volScalarField& p_;
    volScalarField pRes_;

    surfaceScalarField& phi_;
    surfaceScalarField phiRes_;
    //@}

    /// alpha porosity term
    volScalarField& alphaPorosity_;

    /// fvSource term
    volVectorField& fvSource_;

    /// fvOptions term
    fv::options& fvOptions_;

    /// DATurbulenceModel object
    DATurbulenceModel& daTurb_;

    /// simpleControl object which will be initialized in this class
    simpleControl simple_;

    /// Multiple Reference Frame 
    IOMRFZoneListDF& MRF_;

    /// whether to have fvSource term
    label hasFvSource_ = 0;

public:
    TypeName("DASimpleFoam");
    // Constructors

    //- Construct from components
    DAResidualSimpleFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAResidualSimpleFoam()
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
