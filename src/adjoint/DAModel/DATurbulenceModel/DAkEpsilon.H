/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
    Child class for the kEpsilon model

\*---------------------------------------------------------------------------*/

#ifndef DAkEpsilon_H
#define DAkEpsilon_H

#include "DATurbulenceModel.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DAkEpsilon Declaration
\*---------------------------------------------------------------------------*/

class DAkEpsilon
    : public DATurbulenceModel
{

protected:
    /// \name SST parameters
    //@{
    dimensionedScalar Cmu_;
    dimensionedScalar C1_;
    dimensionedScalar C2_;
    dimensionedScalar C3_;
    dimensionedScalar sigmak_;
    dimensionedScalar sigmaEps_;
    //@}

    /// \name SST functions
    //@{
    tmp<fvScalarMatrix> kSource() const;
    tmp<fvScalarMatrix> epsilonSource() const;
    //- Return the effective diffusivity for k
    tmp<volScalarField> DkEff() const;
    //- Return the effective diffusivity for epsilon
    tmp<volScalarField> DepsilonEff() const;
    //- Return the turbulence kinetic energy
    virtual tmp<volScalarField> k() const
    {
        return k_;
    }

    //- Return the turbulence kinetic energy dissipation rate
    virtual tmp<volScalarField> epsilon() const
    {
        return epsilon_;
    }
    //@}

    /// \name Augmented variables for adjoint residuals
    //@{
    volScalarField& epsilon_;
    volScalarField epsilonRes_;
    volScalarField& k_;
    volScalarField kRes_;
    //@}

    /// cell-center epsilon values near the wall, this is to fix the issue that the
    /// epsilonWallFunction will try to modify epsilon values for the cells near walls
    /// this will cause issue for FD-based partial derivatives, so here we basically
    /// implement a zeroGradient BC for near wall epsilon.
    scalarList epsilonNearWall_;

    /// whether to solve for turb states
    label solveTurbState_ = 0;

    /// time step interval to print residual
    label printInterval_;

public:
    TypeName("kEpsilon");
    // Constructors

    //- Construct from components
    DAkEpsilon(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption);

    //- Destructor
    virtual ~DAkEpsilon()
    {
    }

    // Member functions

    /// update the turbulence state for DAStateInfo::regStates_
    virtual void correctModelStates(wordList& modelStates) const;

    /// update nut based on other turbulence variables and update the BCs
    virtual void correctNut();

    /// update turbulence variable boundary values
    virtual void correctBoundaryConditions();

    /// update any intermediate variables that are dependent on state variables and are used in calcResiduals
    virtual void updateIntermediateVariables();

    /// update the original variable connectivity for the adjoint state residuals in stateCon
    virtual void correctStateResidualModelCon(List<List<word>>& stateCon) const;

    /// add the model residual connectivity to stateCon
    virtual void addModelResidualCon(HashTable<List<List<word>>>& allCon) const;

    /// compute the turbulence residuals
    virtual void calcResiduals(const dictionary& options);

    /// solve the residual equations and update the state
    virtual void correct();

    /// save near wall epsilon values to epsilonNearWall_
    void saveEpsilonNearWall();

    /// set epsilonNearWall_ to near wall epsilon values
    void setEpsilonNearWall();

    /// specially treatment to correct epsilon BC
    void correctEpsilonBoundaryConditions();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
