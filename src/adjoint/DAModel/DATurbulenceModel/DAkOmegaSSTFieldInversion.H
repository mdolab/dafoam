/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
    Child class for the kOmegaSST model with a betaFieldInversion field 
    multiplying to the omega transport equation. This betaFieldInversion term
    can then be trained to improve the kOmegaSST model. 

    This file is modified from OpenFOAM's source code
    src/TurbulenceModels/turbulenceModels/RAS/kOmega/kOmega.H

    OpenFOAM: The Open Source CFD Toolbox

    Copyright (C): 2011-2016 OpenFOAM Foundation

    OpenFOAM License:

        OpenFOAM is free software: you can redistribute it and/or modify it
        under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.
    
        OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
        ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
        FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
        for more details.
    
        You should have received a copy of the GNU General Public License
        along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#ifndef DAkOmegaSSTFieldInversion_H
#define DAkOmegaSSTFieldInversion_H

#include "DATurbulenceModel.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DAkOmegaSSTFieldInversion Declaration
\*---------------------------------------------------------------------------*/

class DAkOmegaSSTFieldInversion
    : public DATurbulenceModel
{

protected:
    /// \name SST parameters
    //@{
    dimensionedScalar alphaK1_;
    dimensionedScalar alphaK2_;

    dimensionedScalar alphaOmega1_;
    dimensionedScalar alphaOmega2_;

    dimensionedScalar gamma1_;
    dimensionedScalar gamma2_;

    dimensionedScalar beta1_;
    dimensionedScalar beta2_;

    dimensionedScalar betaStar_;

    dimensionedScalar a1_;
    dimensionedScalar b1_;
    dimensionedScalar c1_;

    Switch F3_;
    //@}

    /// \name SST functions
    //@{
    tmp<volScalarField> F1(const volScalarField& CDkOmega) const;
    tmp<volScalarField> F2() const;
    tmp<volScalarField> F3() const;
    tmp<volScalarField> F23() const;

    tmp<volScalarField> blend(
        const volScalarField& F1,
        const dimensionedScalar& psi1,
        const dimensionedScalar& psi2) const
    {
        return F1 * (psi1 - psi2) + psi2;
    }

    tmp<volScalarField::Internal> blend(
        const volScalarField::Internal& F1,
        const dimensionedScalar& psi1,
        const dimensionedScalar& psi2) const
    {
        return F1 * (psi1 - psi2) + psi2;
    }

    tmp<volScalarField> alphaK(const volScalarField& F1) const
    {
        return blend(F1, alphaK1_, alphaK2_);
    }

    tmp<volScalarField> alphaOmega(const volScalarField& F1) const
    {
        return blend(F1, alphaOmega1_, alphaOmega2_);
    }

    tmp<volScalarField::Internal> beta(
        const volScalarField::Internal& F1) const
    {
        return blend(F1, beta1_, beta2_);
    }

    tmp<volScalarField::Internal> gamma(
        const volScalarField::Internal& F1) const
    {
        return blend(F1, gamma1_, gamma2_);
    }

    //- Return the effective diffusivity for k
    tmp<volScalarField> DkEff(const volScalarField& F1) const
    {
        return tmp<volScalarField>(
            new volScalarField("DkEff", alphaK(F1) * nut_ + this->nu()));
    }

    //- Return the effective diffusivity for omega
    tmp<volScalarField> DomegaEff(const volScalarField& F1) const
    {
        return tmp<volScalarField>(
            new volScalarField(
                "DomegaEff",
                alphaOmega(F1) * nut_ + this->nu()));
    }

    //- Return k production rate
    tmp<volScalarField::Internal> Pk(
        const volScalarField::Internal& G) const;

    //- Return epsilon/k which for standard RAS is betaStar*omega
    tmp<volScalarField::Internal> epsilonByk(
        const volScalarField& F1,
        const volTensorField& gradU) const;

    //- Return G/nu
    tmp<volScalarField::Internal> GbyNu(
        const volScalarField::Internal& GbyNu0,
        const volScalarField::Internal& F2,
        const volScalarField::Internal& S2) const;

    tmp<fvScalarMatrix> kSource() const;

    tmp<fvScalarMatrix> omegaSource() const;

    tmp<fvScalarMatrix> Qsas(
        const volScalarField::Internal& S2,
        const volScalarField::Internal& gamma,
        const volScalarField::Internal& beta) const;
    //@}

    /// \name Augmented variables for adjoint residuals
    //@{
    volScalarField& omega_;
    volScalarField omegaRes_;
    volScalarField omegaResRef_;
    volScalarField omegaResPartDeriv_;
    volScalarField omegaRef_;
    volScalarField& k_;
    volScalarField kRes_;
    volScalarField kResRef_;
    volScalarField kResPartDeriv_;
    volScalarField kRef_;
    //@}

    /// A beta field multiplying to the production term
    volScalarField& betaFieldInversion_; 

    /// a surface friction 'field' when using skin friction data for field inversion
    volScalarField surfaceFriction_;

    /// the reference field for surfaceFriction
    volScalarField surfaceFrictionData_; 

    /// the reference field for surfacePressure
    volScalarField pData_; 
    
    /// reference field (e.g. velocity for DNS)
    volVectorField UData_; 

    /// the reference pressure field data
    volScalarField USingleComponentData_;
    
    /// Name of turbulent transport equation to modify: can be k, omega, or all
    /// if k OR omega is selected the field inversion modification is: ProdTerm(k or omega)*betaFieldInversion
    /// if all is selected the modification is: ProdTerm(k)*betaField, ProdTerm(omega)*(1.0/betaField)
    /// in the second case this is because nut is proportional to k and inversely proportional to omega
    //word eqnToModify_;

    /// Weight of betaFieldInversion for the turbulent transport equations if both are modified
    //scalar betaFieldInversionWeight_; 

    /// 3D wall distance
    const volScalarField& y_;

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
    TypeName("kOmegaSSTFieldInversion");
    // Constructors

    //- Construct from components
    DAkOmegaSSTFieldInversion(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption);

    //- Destructor
    virtual ~DAkOmegaSSTFieldInversion()
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
