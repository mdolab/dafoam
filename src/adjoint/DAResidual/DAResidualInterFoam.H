/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for DAInterFoam

\*---------------------------------------------------------------------------*/

#ifndef DAResidualInterFoam_H
#define DAResidualInterFoam_H

#include "DAResidual.H"
#include "addToRunTimeSelectionTable.H"
#include "pimpleControl.H"
#include "adjustPhi.H"
#include "immiscibleIncompressibleTwoPhaseMixture.H"
#include "constrainPressure.H"
#include "EulerDdtScheme.H"
#include "gaussConvectionScheme.H"
#include "upwind.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAResidualInterFoam Declaration
\*---------------------------------------------------------------------------*/

class DAResidualInterFoam
    : public DAResidual
{
protected:
    /// \name These are state variables, state residuals, and partial derivatives
    //@{
    volVectorField& U_;
    volVectorField URes_;

    volScalarField& p_rgh_;
    volScalarField p_rghRes_;

    volScalarField& alpha1_;
    volScalarField alpha1Res_;

    surfaceScalarField& phi_;
    surfaceScalarField phiRes_;
    //@}

    volScalarField& rho_;
    surfaceScalarField& rhoPhi_;
    volScalarField& gh_;
    surfaceScalarField& ghf_;
    surfaceScalarField& alphaPhi10_;

    immiscibleIncompressibleTwoPhaseMixture& mixture_;

    incompressible::turbulenceModel& turbulence_;

    /// DATurbulenceModel object
    DATurbulenceModel& daTurb_;

    /// pimpleControl object which will be initialized in this class
    pimpleControl pimple_;

public:
    TypeName("DAInterFoam");
    // Constructors

    //- Construct from components
    DAResidualInterFoam(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAResidualInterFoam()
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

    virtual void calcPCMatWithFvMatrix(Mat PCMat);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
