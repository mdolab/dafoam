/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
    Child class for the kOmega model

\*---------------------------------------------------------------------------*/

#ifndef DAkOmega_H
#define DAkOmega_H

#include "DATurbulenceModel.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DAkOmega Declaration
\*---------------------------------------------------------------------------*/

class DAkOmega
    : public DATurbulenceModel
{

protected:
    /// \name kOmega parameters
    //@{
    dimensionedScalar Cmu_;
    dimensionedScalar beta_;
    dimensionedScalar gamma_;
    dimensionedScalar alphaK_;
    dimensionedScalar alphaOmega_;
    //@}

    /// \name kOmega functions
    //@{
    //- Return the effective diffusivity for k
        tmp<volScalarField> DkEff() const
        {
            return tmp<volScalarField>
            (
                new volScalarField
                (
                    "DkEff",
                    alphaK_*this->nut_ + this->nu()
                )
            );
        }

        //- Return the effective diffusivity for omega
        tmp<volScalarField> DomegaEff() const
        {
            return tmp<volScalarField>
            (
                new volScalarField
                (
                    "DomegaEff",
                    alphaOmega_*this->nut_ + this->nu()
                )
            );
        }
    //@}

    /// \name Augmented variables for adjoint residuals
    //@{
    volScalarField& omega_;
    volScalarField omegaRes_;
    volScalarField& k_;
    volScalarField kRes_;
    //@}

    /// cell-center omega values near the wall, this is to fix the issue that the
    /// omegaWallFunction will try to modify omega values for the cells near walls
    /// this will cause issue for FD-based partial derivatives, so here we basically
    /// implement a zeroGradient BC for near wall omega.
    scalarList omegaNearWall_;

    /// whether to solve for turb states
    label solveTurbState_ = 0;

    /// time step interval to print residual
    label printInterval_;

public:
    TypeName("kOmega");
    // Constructors

    //- Construct from components
    DAkOmega(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption);

    //- Destructor
    virtual ~DAkOmega()
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

    /// save near wall omega values to omegaNearWall_
    void saveOmegaNearWall();

    /// set omegaNearWall_ to near wall omega values
    void setOmegaNearWall();

    /// specially treatment to correct epsilon BC
    void correctOmegaBoundaryConditions();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
