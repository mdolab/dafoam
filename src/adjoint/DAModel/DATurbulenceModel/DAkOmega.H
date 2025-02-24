/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
    Child class for the kOmega model

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
        return tmp<volScalarField>(
            new volScalarField(
                "DkEff",
                alphaK_ * this->nut_ + this->nu()));
    }

    //- Return the effective diffusivity for omega
    tmp<volScalarField> DomegaEff() const
    {
        return tmp<volScalarField>(
            new volScalarField(
                "DomegaEff",
                alphaOmega_ * this->nut_ + this->nu()));
    }
    //@}

    /// \name Augmented variables for adjoint residuals
    //@{
    volScalarField& omega_;
    volScalarField omegaRes_;
    volScalarField& k_;
    volScalarField kRes_;
    //@}

    /// we need to make the G field a class variable and register it to the mesh.db
    /// such that the omegaWallFunction BC can find it
    autoPtr<volScalarField> GPtr_;

    /// beta field for field inversion
    volScalarField betaFIK_;
    volScalarField betaFIOmega_;

    /// cell-center omega values near the wall, this is to fix the issue that the
    /// omegaWallFunction will try to modify omega values for the cells near walls
    /// this will cause issue for FD-based partial derivatives, so here we basically
    /// implement a zeroGradient BC for near wall omega.
    scalarList omegaNearWall_;

    /// whether to solve for turb states
    label solveTurbState_ = 0;

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
    virtual void correct(label printToScreen);

    /// save near wall omega values to omegaNearWall_
    void saveOmegaNearWall();

    /// set omegaNearWall_ to near wall omega values
    void setOmegaNearWall();

    /// specially treatment to correct epsilon BC
    void correctOmegaBoundaryConditions();

    /// return the diag(), upper(), and lower() scalarFields from the turbulence model's fvMatrix
    virtual void getFvMatrixFields(
        const word varName,
        scalarField& diag,
        scalarField& upper,
        scalarField& lower);

    /// return the value of the destruction term from the turbulence model
    virtual void getTurbProdOverDestruct(volScalarField& PoD) const;

    /// return the value of the convective over production term from the turbulence model
    virtual void getTurbConvOverProd(volScalarField& CoP) const;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
