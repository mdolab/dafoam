/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
    Child class for the SpalartAllmarasFv3 model used in OpenFOAM 2.4 and before

    This file is modified from OpenFOAM's source code
    src/TurbulenceModels/turbulenceModels/RAS/SpalartAllmaras/SpalartAllmaras.H

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

#ifndef DASpalartAllmarasFv3_H
#define DASpalartAllmarasFv3_H

#include "DATurbulenceModel.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DASpalartAllmarasFv3 Declaration
\*---------------------------------------------------------------------------*/

class DASpalartAllmarasFv3
    : public DATurbulenceModel
{

protected:
    /// \name SA parameters
    //@{
    dimensionedScalar sigmaNut_;
    dimensionedScalar kappa_;
    dimensionedScalar Cb1_;
    dimensionedScalar Cb2_;
    dimensionedScalar Cw1_;
    dimensionedScalar Cw2_;
    dimensionedScalar Cw3_;
    dimensionedScalar Cv1_;
    dimensionedScalar Cv2_;
    //@}

    /// \name SA functions
    //@{
    tmp<volScalarField> chi() const;

    tmp<volScalarField> fv1(const volScalarField& chi) const;

    tmp<volScalarField> fv2(
        const volScalarField& chi,
        const volScalarField& fv1) const;

    tmp<volScalarField> fv3(
        const volScalarField& chi,
        const volScalarField& fv1) const;

    tmp<volScalarField> fw(const volScalarField& Stilda) const;

    //@}

    /// \name Augmented variables for residual computation
    //@{
    volScalarField& nuTilda_;
    volScalarField nuTildaRes_;
    //@}

    /// pseudoNuTilda_ and pseudoNuTildaEqn_ for solving adjoint equation
    volScalarField pseudoNuTilda_;
    fvScalarMatrix pseudoNuTildaEqn_;

    /// 3D wall distance
    const volScalarField& y_;

    /// beta field for field inversion
    volScalarField betaFINuTilda_;

    /// whether to solve for turb states
    label solveTurbState_ = 0;

    // fvSolutions parameters
    scalar relaxNuTildaEqn_ = 1.0;
    dictionary solverDictNuTilda_;

public:
    TypeName("SpalartAllmarasFv3");
    // Constructors

    //- Construct from components
    DASpalartAllmarasFv3(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption);

    //- Destructor
    virtual ~DASpalartAllmarasFv3()
    {
    }

    // Member functions

    /// Return the effective diffusivity for nuTilda
    tmp<volScalarField> DnuTildaEff() const;

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

    /// Inverse transpose product, A_nuTilda^(-T)
    virtual void invTranProdNuTildaEqn(
        const volScalarField& mySource,
        volScalarField& pseudoNuTilda);

    /// construct pseudoNuTildaEqn_, which is nuTildaEqn with swapped upper and lower arrays
    virtual void constructPseudoNuTildaEqn();

    /// solve pseudoNuTildaEqn_ with overwritten rhs
    virtual void rhsSolvePseudoNuTildaEqn(const volScalarField& nuTildaSource);

    /// calculate the turbulence residual using LDU matrix
    virtual void calcLduResidualTurb(volScalarField& nuTildaRes);

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
