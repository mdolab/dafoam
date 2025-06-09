/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    This file is modified from OpenFOAM's source code
    src/TurbulenceModels/turbulenceModels/RAS/SpalartAllmaras/SpalartAllmaras.C

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

#include "DASpalartAllmaras.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DASpalartAllmaras, 0);
addToRunTimeSelectionTable(DATurbulenceModel, DASpalartAllmaras, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DASpalartAllmaras::DASpalartAllmaras(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption)
    : DATurbulenceModel(modelType, mesh, daOption),
      // SA parameters
      sigmaNut_(dimensioned<scalar>::lookupOrAddToDict(
          "sigmaNut",
          this->coeffDict_,
          0.66666)),
      kappa_(dimensioned<scalar>::lookupOrAddToDict(
          "kappa",
          this->coeffDict_,
          0.41)),

      Cb1_(dimensioned<scalar>::lookupOrAddToDict(
          "Cb1",
          this->coeffDict_,
          0.1355)),
      Cb2_(dimensioned<scalar>::lookupOrAddToDict(
          "Cb2",
          this->coeffDict_,
          0.622)),
      Cw1_(Cb1_ / sqr(kappa_) + (1.0 + Cb2_) / sigmaNut_),
      Cw2_(dimensioned<scalar>::lookupOrAddToDict(
          "Cw2",
          this->coeffDict_,
          0.3)),
      Cw3_(dimensioned<scalar>::lookupOrAddToDict(
          "Cw3",
          this->coeffDict_,
          2.0)),
      Cv1_(dimensioned<scalar>::lookupOrAddToDict(
          "Cv1",
          this->coeffDict_,
          7.1)),
      Cs_(dimensioned<scalar>::lookupOrAddToDict(
          "Cs",
          this->coeffDict_,
          0.3)),
      // Augmented variables
      nuTilda_(const_cast<volScalarField&>(
          mesh.thisDb().lookupObject<volScalarField>("nuTilda"))),
      nuTildaRes_(
          IOobject(
              "nuTildaRes",
              mesh.time().timeName(),
              mesh,
              IOobject::NO_READ,
              IOobject::NO_WRITE),
          mesh,
          dimensionedScalar("nuTildaRes", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
          zeroGradientFvPatchField<scalar>::typeName),
      y_(mesh.thisDb().lookupObject<volScalarField>("yWall")),
      betaFINuTilda_(
          IOobject(
              "betaFINuTilda",
              mesh.time().timeName(),
              mesh,
              IOobject::READ_IF_PRESENT,
              IOobject::AUTO_WRITE),
          mesh,
          dimensionedScalar("betaFINuTilda", dimensionSet(0, 0, 0, 0, 0, 0, 0), 1.0),
          "zeroGradient"),
      psiNuTildaPC_(nullptr),
      dPsiNuTilda_("dPsiNuTilda", nuTilda_)
{
    // we need to reset the nuTildaRes's dimension based on the model type
    if (turbModelType_ == "incompressible")
    {
        nuTildaRes_.dimensions().reset(dimensionSet(0, 2, -2, 0, 0, 0, 0));
    }

    if (turbModelType_ == "compressible")
    {
        nuTildaRes_.dimensions().reset(dimensionSet(1, -1, -2, 0, 0, 0, 0));
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// SA member functions. these functions are copied from
// src/TurbulenceModels/turbulenceModels/RAS/SpalartAllmaras/SpalartAllmaras.C
tmp<volScalarField> DASpalartAllmaras::chi() const
{
    return nuTilda_ / this->nu();
}

tmp<volScalarField> DASpalartAllmaras::fv1(
    const volScalarField& chi) const
{
    const volScalarField chi3(pow3(chi));
    return chi3 / (chi3 + pow3(Cv1_));
}

tmp<volScalarField> DASpalartAllmaras::fv2(
    const volScalarField& chi,
    const volScalarField& fv1) const
{
    return 1.0 - chi / (1.0 + chi * fv1);
}

tmp<volScalarField> DASpalartAllmaras::Stilda(
    const volScalarField& chi,
    const volScalarField& fv1) const
{
    volScalarField Omega(::sqrt(2.0) * mag(skew(fvc::grad(U_))));

    return (
        max(
            Omega
                + fv2(chi, fv1) * nuTilda_ / sqr(kappa_ * y_),
            Cs_ * Omega));
}

tmp<volScalarField> DASpalartAllmaras::fw(
    const volScalarField& Stilda) const
{
    volScalarField r(
        min(
            nuTilda_
                / (max(
                       Stilda,
                       dimensionedScalar("SMALL", Stilda.dimensions(), SMALL))
                   * sqr(kappa_ * y_)),
            scalar(10.0)));
    r.boundaryFieldRef() == 0.0;

    const volScalarField g(r + Cw2_ * (pow6(r) - r));

    return g * pow((1.0 + pow6(Cw3_)) / (pow6(g) + pow6(Cw3_)), 1.0 / 6.0);
}

tmp<volScalarField> DASpalartAllmaras::DnuTildaEff() const
{
    return tmp<volScalarField>(
        new volScalarField("DnuTildaEff", (nuTilda_ + this->nu()) / sigmaNut_));
}

// Augmented functions
void DASpalartAllmaras::correctModelStates(wordList& modelStates) const
{
    /*
    Description:
        Update the name in modelStates based on the selected physical model at runtime

    Example:
        In DAStateInfo, if the modelStates reads:
        
        modelStates = {"nut"}
        
        then for the SA model, calling correctModelStates(modelStates) will give:
    
        modelStates={"nuTilda"}
        
        while calling correctModelStates(modelStates) for the SST model will give 
        
        modelStates={"k","omega"}
        
        We don't udpate the names for the radiation model becasue users are 
        supposed to set modelStates={"G"}
    */

    // replace nut with nuTilda
    forAll(modelStates, idxI)
    {
        word stateName = modelStates[idxI];
        if (stateName == "nut")
        {
            modelStates[idxI] = "nuTilda";
        }
    }
}

void DASpalartAllmaras::correctNut()
{
    /*
    Description:
        Update nut based on other turbulence variables and update the BCs
        Also update alphat if is present
    */

    const volScalarField chi(this->chi());
    const volScalarField fv1(this->fv1(chi));
    nut_ = nuTilda_ * fv1;

    nut_.correctBoundaryConditions();

    // this is basically BasicTurbulenceModel::correctNut();
    this->correctAlphat();

    return;
}

void DASpalartAllmaras::correctBoundaryConditions()
{
    /*
    Description:
        Update turbulence variable boundary values
    */

    // correct the BCs for the perturbed fields
    nuTilda_.correctBoundaryConditions();
}

void DASpalartAllmaras::updateIntermediateVariables()
{
    /*
    Description:
        Update nut based on nuTilda. Note: we need to update nut and its BC since we 
        may have perturbed other turbulence vars that affect the nut values
    */

    this->correctNut();
}

void DASpalartAllmaras::correctStateResidualModelCon(List<List<word>>& stateCon) const
{
    /*
    Description:
        Update the original variable connectivity for the adjoint state 
        residuals in stateCon. Basically, we modify/add state variables based on the
        original model variables defined in stateCon.

    Input:
    
        stateResCon: the connectivity levels for a state residual, defined in Foam::DAJacCon

    Example:
        If stateCon reads:
        stateCon=
        {
            {"U", "p", "nut"},
            {"p"}
        }
    
        For the SA turbulence model, calling this function for will get a new stateCon
        stateCon=
        {
            {"U", "p", "nuTilda"},
            {"p"}
        }
    
        For the SST turbulence model, calling this function will give
        stateCon=
        {
            {"U", "p", "k", "omega"},
            {"p", "U"}
        }
        ***NOTE***: we add a extra level of U connectivity because nut is 
        related to grad(U), k, and omega in SST!
    */

    forAll(stateCon, idxI)
    {
        forAll(stateCon[idxI], idxJ)
        {
            word conStateName = stateCon[idxI][idxJ];
            if (conStateName == "nut")
            {
                stateCon[idxI][idxJ] = "nuTilda";
            }
        }
    }
}

void DASpalartAllmaras::addModelResidualCon(HashTable<List<List<word>>>& allCon) const
{
    /*
    Description:
        Add the connectivity levels for all physical model residuals to allCon

    Input:
        allCon: the connectivity levels for all state residual, defined in DAJacCon

    Example:
        If stateCon reads:
        allCon=
        {
            "URes":
            {
               {"U", "p", "nut"},
               {"p"}
            }
        }
    
        For the SA turbulence model, calling this function for will get a new stateCon,
        something like this:
        allCon=
        {
            "URes":
            {
               {"U", "p", "nuTilda"},
               {"p"}
            },
            "nuTildaRes": 
            {
                {"U", "phi", "nuTilda"},
                {"U"}
            }
        }

    */

    word pName;

    if (mesh_.thisDb().foundObject<volScalarField>("p"))
    {
        pName = "p";
    }
    else if (mesh_.thisDb().foundObject<volScalarField>("p_rgh"))
    {
        pName = "p_rgh";
    }
    else
    {
        FatalErrorIn(
            "Neither p nor p_rgh was found in mesh.thisDb()!"
            "addModelResidualCon failed to setup turbulence residuals!")
            << exit(FatalError);
    }

    // NOTE: for compressible flow, it depends on rho so we need to add T and p
    if (turbModelType_ == "incompressible")
    {
        allCon.set(
            "nuTildaRes",
            {
                {"U", "nuTilda", "phi"}, // lv0
                {"U", "nuTilda"}, // lv1
                {"nuTilda"} // lv2
            });
    }
    else if (turbModelType_ == "compressible")
    {
        allCon.set(
            "nuTildaRes",
            {
                {"U", "T", pName, "nuTilda", "phi"}, // lv0
                {"U", "T", pName, "nuTilda"}, // lv1
                {"T", pName, "nuTilda"} // lv2
            });
    }
}

void DASpalartAllmaras::correct(label printToScreen)
{
    /*
    Descroption:
        Solve the residual equations and update the state. This function will be called 
        by the DASolver. It is needed because we want to control the output frequency
        of the residual convergence every 100 steps. If using the correct from turbulence
        it will output residual every step which will be too much of information.
    */

    // We set the flag solveTurbState_ to 1 such that in the calcResiduals function
    // we will solve and update nuTilda
    solveTurbState_ = 1;
    dictionary dummyOptions;
    dummyOptions.set("printToScreen", printToScreen);
    this->calcResiduals(dummyOptions);
    // after it, we reset solveTurbState_ = 0 such that calcResiduals will not
    // update nuTilda when calling from the adjoint class, i.e., solveAdjoint from DASolver.
    solveTurbState_ = 0;
}

void DASpalartAllmaras::calcResiduals(const dictionary& options)
{
    /*
    Descroption:
        If solveTurbState_ == 1, this function solve and update nuTilda, and 
        is the same as calling turbulence.correct(). If solveTurbState_ == 0,
        this function compute residuals for turbulence variables, e.g., nuTildaRes_

    Input:
        options.isPC: 1 means computing residuals for preconditioner matrix.
        This essentially use the first order scheme for div(phi,nuTilda)

        p_, U_, phi_, etc: State variables in OpenFOAM
    
    Output:
        nuTildaRes_: If solveTurbState_ == 0, update the residual field variable

        nuTilda_: If solveTurbState_ == 1, update nuTilda
    */

    // Copy and modify based on the "correct" function

    word divNuTildaScheme = "div(phi,nuTilda)";

    label isPC = 0;

    label printToScreen = options.lookupOrDefault("printToScreen", 0);

    if (!solveTurbState_)
    {
        isPC = options.getLabel("isPC");

        if (isPC)
        {
            divNuTildaScheme = "div(pc)";
        }
    }

    const volScalarField chi(this->chi());
    const volScalarField fv1(this->fv1(chi));

    const volScalarField Stilda(this->Stilda(chi, fv1));

    volScalarField rho = this->rho();

    tmp<fvScalarMatrix> nuTildaEqn(
        fvm::ddt(phase_, rho, nuTilda_)
            + fvm::div(phaseRhoPhi_, nuTilda_, divNuTildaScheme)
            - fvm::laplacian(phase_ * rho * DnuTildaEff(), nuTilda_)
            - Cb2_ / sigmaNut_ * phase_ * rho * magSqr(fvc::grad(nuTilda_))
        == Cb1_ * phase_ * rho * Stilda * nuTilda_ * betaFINuTilda_
            - fvm::Sp(Cw1_ * phase_ * rho * fw(Stilda) * nuTilda_ / sqr(y_), nuTilda_));

    nuTildaEqn.ref().relax();

    if (solveTurbState_)
    {
        // get the solver performance info such as initial
        // and final residuals
        SolverPerformance<scalar> solverNuTilda = solve(nuTildaEqn);

        DAUtility::primalResidualControl(solverNuTilda, printToScreen, "nuTilda", daGlobalVar_.primalMaxRes);

        DAUtility::boundVar(allOptions_, nuTilda_, printToScreen);
        nuTilda_.correctBoundaryConditions();

        // ***************** NOTE*****************
        // In the original SA, it is correctNut(fv1) and fv1 is not
        // updated based on the latest nuTilda. We use correctNut which
        // recompute fv1 with the latest nuTilda
        this->correctNut();
    }
    else
    {
        // calculate residuals
        nuTildaRes_ = nuTildaEqn.ref() & nuTilda_;
        // need to normalize residuals
        normalizeResiduals(nuTildaRes);
    }

    return;
}

void DASpalartAllmaras::getFvMatrixFields(
    const word varName,
    scalarField& diag,
    scalarField& upper,
    scalarField& lower)
{
    /* 
    Description:
        return the diag(), upper(), and lower() scalarFields from the turbulence model's fvMatrix
        this will be use to compute the preconditioner matrix
    */

    if (varName != "nuTilda")
    {
        FatalErrorIn(
            "varName not valid. It has to be nuTilda")
            << exit(FatalError);
    }

    const volScalarField chi(this->chi());
    const volScalarField fv1(this->fv1(chi));

    const volScalarField Stilda(this->Stilda(chi, fv1));

    volScalarField rho = this->rho();

    fvScalarMatrix nuTildaEqn(
        fvm::ddt(phase_, rho, nuTilda_)
            + fvm::div(phaseRhoPhi_, nuTilda_, "div(pc)")
            - fvm::laplacian(phase_ * rho * DnuTildaEff(), nuTilda_)
            - Cb2_ / sigmaNut_ * phase_ * rho * magSqr(fvc::grad(nuTilda_))
        == Cb1_ * phase_ * rho * Stilda * nuTilda_ * betaFINuTilda_
            - fvm::Sp(Cw1_ * phase_ * rho * fw(Stilda) * nuTilda_ / sqr(y_), nuTilda_));

    nuTildaEqn.relax();

    diag = nuTildaEqn.D();
    upper = nuTildaEqn.upper();
    lower = nuTildaEqn.lower();
}

void DASpalartAllmaras::solveAdjointFP(
    const word varName,
    const scalarList& rhs,
    scalarList& dPsi)
{
    if (varName != "nuTilda")
    {
        FatalErrorIn(
            "varName not valid. It has to be nuTilda")
            << exit(FatalError);
    }

    const fvSolution& myFvSolution = mesh_.thisDb().lookupObject<fvSolution>("fvSolution");
    dictionary solverDictNuTilda = myFvSolution.subDict("solvers").subDict("nuTilda");

    // solve the fvMatrixT field with given rhs and solution
    if (psiNuTildaPC_.empty())
    {
        dPsiNuTilda_ == nuTilda_;
        const volScalarField chi(this->chi());
        const volScalarField fv1(this->fv1(chi));

        const volScalarField Stilda(this->Stilda(chi, fv1));

        volScalarField rho = this->rho();

        // for the PC, we need only fvm:: terms
        psiNuTildaPC_.reset(new fvScalarMatrix(
            fvm::ddt(phase_, rho, dPsiNuTilda_)
            + fvm::div(phaseRhoPhi_, dPsiNuTilda_, "div(phi,nuTilda)")
            - fvm::laplacian(phase_ * rho * DnuTildaEff(), dPsiNuTilda_)
            - Cb2_ / sigmaNut_ * phase_ * rho * magSqr(fvc::grad(dPsiNuTilda_))
            + fvm::Sp(Cw1_ * phase_ * rho * fw(Stilda) * dPsiNuTilda_ / sqr(y_), dPsiNuTilda_)));

        psiNuTildaPC_->relax();

        DAUtility::swapLists<scalar>(psiNuTildaPC_->upper(), psiNuTildaPC_->lower());

        // make sure the boundary contribution to source hrs is zero
        forAll(dPsiNuTilda_.boundaryField(), patchI)
        {
            forAll(dPsiNuTilda_.boundaryField()[patchI], faceI)
            {
                psiNuTildaPC_->boundaryCoeffs()[patchI][faceI] = 0.0;
            }
        }
    }

    // force to use zeros as the initial guess
    forAll(dPsiNuTilda_, cellI)
    {
        dPsiNuTilda_[cellI] = 0.0;
    }

    // set the rhs
    forAll(psiNuTildaPC_->source(), cellI)
    {
        psiNuTildaPC_->source()[cellI] = rhs[cellI];
    }

    // solve
    psiNuTildaPC_->solve(solverDictNuTilda);

    // return the solution
    forAll(dPsiNuTilda_, cellI)
    {
        dPsi[cellI] = dPsiNuTilda_[cellI];
    }
}

void DASpalartAllmaras::getTurbProdOverDestruct(volScalarField& PoD) const
{
    /*
    Description:
        Return the value of the production over destruction term from the turbulence model 
    */

    const volScalarField chi(this->chi());
    const volScalarField fv1(this->fv1(chi));

    const volScalarField Stilda(this->Stilda(chi, fv1));

    volScalarField rho = this->rho();

    volScalarField P = Cb1_ * phase_ * rho * Stilda * nuTilda_;
    volScalarField D = Cw1_ * phase_ * rho * fw(Stilda) * sqr(nuTilda_ / y_);

    forAll(P, cellI)
    {
        PoD[cellI] = P[cellI] / (D[cellI] + P[cellI] + 1e-16);
    }
}

void DASpalartAllmaras::getTurbConvOverProd(volScalarField& CoP) const
{
    /*
    Description:
        Return the value of the convective over production term from the turbulence model 
    */

    const volScalarField chi(this->chi());
    const volScalarField fv1(this->fv1(chi));

    const volScalarField Stilda(this->Stilda(chi, fv1));

    volScalarField rho = this->rho();

    volScalarField P = Cb1_ * phase_ * rho * Stilda * nuTilda_;
    volScalarField C = fvc::div(phaseRhoPhi_, nuTilda_);

    forAll(P, cellI)
    {
        CoP[cellI] = C[cellI] / (P[cellI] + C[cellI] + 1e-16);
    }
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
