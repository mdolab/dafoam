/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    This file is modified from OpenFOAM's source code
    src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.H

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

#include "DAkEpsilon.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAkEpsilon, 0);
addToRunTimeSelectionTable(DATurbulenceModel, DAkEpsilon, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAkEpsilon::DAkEpsilon(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption)
    : DATurbulenceModel(modelType, mesh, daOption),
      // KE parameters
      Cmu_(dimensioned<scalar>::lookupOrAddToDict(
          "Cmu",
          this->coeffDict_,
          0.09)),
      C1_(dimensioned<scalar>::lookupOrAddToDict(
          "C1",
          this->coeffDict_,
          1.44)),
      C2_(dimensioned<scalar>::lookupOrAddToDict(
          "C2",
          this->coeffDict_,
          1.92)),
      C3_(dimensioned<scalar>::lookupOrAddToDict(
          "C3",
          this->coeffDict_,
          0)),
      sigmak_(dimensioned<scalar>::lookupOrAddToDict(
          "sigmak",
          this->coeffDict_,
          1.0)),
      sigmaEps_(dimensioned<scalar>::lookupOrAddToDict(
          "sigmaEps",
          this->coeffDict_,
          1.3)),
      // Augmented variables
      epsilon_(const_cast<volScalarField&>(
          mesh_.thisDb().lookupObject<volScalarField>("epsilon"))),
      epsilonRes_(
          IOobject(
              "epsilonRes",
              mesh.time().timeName(),
              mesh,
              IOobject::NO_READ,
              IOobject::NO_WRITE),
          mesh,
          dimensionedScalar("epsilonRes", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
          zeroGradientFvPatchField<scalar>::typeName),
      k_(const_cast<volScalarField&>(
          mesh_.thisDb().lookupObject<volScalarField>("k"))),
      kRes_(
          IOobject(
              "kRes",
              mesh.time().timeName(),
              mesh,
              IOobject::NO_READ,
              IOobject::NO_WRITE),
          mesh,
          dimensionedScalar("kRes", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
          zeroGradientFvPatchField<scalar>::typeName),
      GPtr_(nullptr),
      betaFIK_(
          IOobject(
              "betaFIK",
              mesh.time().timeName(),
              mesh,
              IOobject::READ_IF_PRESENT,
              IOobject::AUTO_WRITE),
          mesh,
          dimensionedScalar("betaFIK", dimensionSet(0, 0, 0, 0, 0, 0, 0), 1.0),
          "zeroGradient"),
      betaFIEpsilon_(
          IOobject(
              "betaFIEpsilon",
              mesh.time().timeName(),
              mesh,
              IOobject::READ_IF_PRESENT,
              IOobject::AUTO_WRITE),
          mesh,
          dimensionedScalar("betaFIEpsilon", dimensionSet(0, 0, 0, 0, 0, 0, 0), 1.0),
          "zeroGradient")
{

    // we need to reset the nuTildaRes's dimension based on the model type
    if (turbModelType_ == "incompressible")
    {
        epsilonRes_.dimensions().reset(dimensionSet(0, 2, -4, 0, 0, 0, 0));
        kRes_.dimensions().reset(dimensionSet(0, 2, -3, 0, 0, 0, 0));
    }

    if (turbModelType_ == "compressible")
    {
        epsilonRes_.dimensions().reset(dimensionSet(1, -1, -4, 0, 0, 0, 0));
        kRes_.dimensions().reset(dimensionSet(1, -1, -3, 0, 0, 0, 0));
    }

    // calculate the size of epsilonWallFunction faces
    label nWallFaces = 0;
    forAll(epsilon_.boundaryField(), patchI)
    {
        if (epsilon_.boundaryField()[patchI].type() == "epsilonWallFunction"
            and epsilon_.boundaryField()[patchI].size() > 0)
        {
            forAll(epsilon_.boundaryField()[patchI], faceI)
            {
                nWallFaces++;
                //Info<<"patchI: "<<patchI<<" faceI: "<<faceI<<endl;
            }
        }
    }

    // initialize epsilonNearWall
    epsilonNearWall_.setSize(nWallFaces);

    // initialize the G field
    tmp<volTensorField> tgradU = fvc::grad(U_);
    GPtr_.reset(new volScalarField::Internal(
        "kEpsilon:G",
        nut_.v() * (dev(twoSymm(tgradU().v())) && tgradU().v())));
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// KE member functions
tmp<fvScalarMatrix> DAkEpsilon::kSource() const
{
    return tmp<fvScalarMatrix>(
        new fvScalarMatrix(
            k_,
            dimVolume * this->rhoDimensions() * k_.dimensions()
                / dimTime));
}

tmp<fvScalarMatrix> DAkEpsilon::epsilonSource() const
{
    return tmp<fvScalarMatrix>(
        new fvScalarMatrix(
            epsilon_,
            dimVolume * this->rhoDimensions() * epsilon_.dimensions()
                / dimTime));
}

tmp<volScalarField> DAkEpsilon::DkEff() const
{
    return tmp<volScalarField>(
        new volScalarField(
            "DkEff",
            (nut_ / sigmak_ + this->nu())));
}

tmp<volScalarField> DAkEpsilon::DepsilonEff() const
{
    return tmp<volScalarField>(
        new volScalarField(
            "DepsilonEff",
            (nut_ / sigmaEps_ + this->nu())));
}

// Augmented functions
void DAkEpsilon::correctModelStates(wordList& modelStates) const
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
        
        modelStates={"k","epsilon"}
        
        We don't udpate the names for the radiation model because users are 
        supposed to set modelStates={"G"}
    */

    // For SST model, we need to replace nut with k, epsilon

    forAll(modelStates, idxI)
    {
        word stateName = modelStates[idxI];
        if (stateName == "nut")
        {
            modelStates[idxI] = "epsilon";
            modelStates.append("k");
        }
    }
}

void DAkEpsilon::correctNut()
{
    /*
    Description:
        Update nut based on other turbulence variables and update the BCs
        Also update alphat if is present
    */

    nut_ = Cmu_ * sqr(k_) / epsilon_;

    nut_.correctBoundaryConditions(); // nutkWallFunction: update wall face nut based on k

    // this is basically BasicTurbulenceModel::correctNut();
    this->correctAlphat();

    return;
}

void DAkEpsilon::correctBoundaryConditions()
{
    /*
    Description:
        Update turbulence variable boundary values
    */

    // correct the BCs for the perturbed fields
    // kqWallFunction is a zero-gradient BC
    k_.correctBoundaryConditions();
}

void DAkEpsilon::correctEpsilonBoundaryConditions()
{
    /*
    Description:
        this is a special treatment for epsilon BC because we cant directly call epsilon.
        correctBoundaryConditions() because it will modify the internal epsilon and G that 
        are right next to walls. This will mess up adjoint Jacobians
        To solve this issue,
        1. we store the near wall epsilon before calling epsilon.correctBoundaryConditions()
        2. call epsilon.correctBoundaryConditions()
        3. Assign the stored near wall epsilon back
        4. Apply a zeroGradient BC for epsilon at the wall patches
        *********** NOTE *************
        this treatment will obviously downgrade the accuracy of adjoint derivative since it is 
        not 100% consistent with what is used for the flow solver; however, our observation 
        shows that the impact is not very large.
    */

    // save the perturbed epsilon at the wall
    this->saveEpsilonNearWall();
    // correct epsilon boundary conditions, this includes updating wall face and near wall epsilon values,
    // updating the inter-proc BCs
    epsilon_.correctBoundaryConditions();
    // reset the corrected epsilon near wall cell to its perturbed value
    this->setEpsilonNearWall();
}

void DAkEpsilon::saveEpsilonNearWall()
{
    /*
    Description:
        Save the current near wall epsilon values to epsilonNearWall_
    */
    label counterI = 0;
    forAll(epsilon_.boundaryField(), patchI)
    {
        if (epsilon_.boundaryField()[patchI].type() == "epsilonWallFunction"
            and epsilon_.boundaryField()[patchI].size() > 0)
        {
            const UList<label>& faceCells = mesh_.boundaryMesh()[patchI].faceCells();
            forAll(faceCells, faceI)
            {
                //Info<<"Near Wall cellI: "<<faceCells[faceI]<<endl;
                epsilonNearWall_[counterI] = epsilon_[faceCells[faceI]];
                counterI++;
            }
        }
    }
    return;
}

void DAkEpsilon::setEpsilonNearWall()
{
    /*
    Description:
        Set the current near wall epsilon values from epsilonNearWall_
        Here we also apply a zeroGradient BC to the wall faces
    */
    label counterI = 0;
    forAll(epsilon_.boundaryField(), patchI)
    {
        if (epsilon_.boundaryField()[patchI].type() == "epsilonWallFunction"
            && epsilon_.boundaryField()[patchI].size() > 0)
        {
            const UList<label>& faceCells = mesh_.boundaryMesh()[patchI].faceCells();
            forAll(faceCells, faceI)
            {
                epsilon_[faceCells[faceI]] = epsilonNearWall_[counterI];
                // zeroGradient BC
                epsilon_.boundaryFieldRef()[patchI][faceI] = epsilon_[faceCells[faceI]];
                counterI++;
            }
        }
    }
    return;
}

void DAkEpsilon::updateIntermediateVariables()
{
    /*
    Description:
        Update nut based on nuTilda. Note: we need to update nut and its BC since we 
        may have perturbed other turbulence vars that affect the nut values
    */

    this->correctNut();
}

void DAkEpsilon::correctStateResidualModelCon(List<List<word>>& stateCon) const
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
                stateCon[idxI][idxJ] = "epsilon"; // replace nut with epsilon
                stateCon[idxI].append("k"); // also add k for that level
            }
        }
    }
}

void DAkEpsilon::addModelResidualCon(HashTable<List<List<word>>>& allCon) const
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
            "epsilonRes",
            {
                {"U", "epsilon", "k", "phi"}, // lv0
                {"U", "epsilon", "k"}, // lv1
                {"U", "epsilon", "k"} // lv2
            });
        allCon.set(
            "kRes",
            {
                {"U", "epsilon", "k", "phi"}, // lv0
                {"U", "epsilon", "k"}, // lv1
                {"U", "epsilon", "k"} // lv2
            });
    }
    else if (turbModelType_ == "compressible")
    {
        allCon.set(
            "epsilonRes",
            {
                {"U", "T", pName, "epsilon", "k", "phi"}, // lv0
                {"U", "T", pName, "epsilon", "k"}, // lv1
                {"U", "T", pName, "epsilon", "k"} // lv2
            });
        allCon.set(
            "kRes",
            {
                {"U", "T", pName, "epsilon", "k", "phi"}, // lv0
                {"U", "T", pName, "epsilon", "k"}, // lv1
                {"U", "T", pName, "epsilon", "k"} // lv2
            });
    }
}

void DAkEpsilon::correct(label printToScreen)
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

void DAkEpsilon::calcResiduals(const dictionary& options)
{
    /*
    Descroption:
        If solveTurbState_ == 1, this function solve and update k and epsilon, and 
        is the same as calling turbulence.correct(). If solveTurbState_ == 0,
        this function compute residuals for turbulence variables, e.g., nuTildaRes_

    Input:
        options.isPC: 1 means computing residuals for preconditioner matrix.
        This essentially use the first order scheme for div(phi,nuTilda)

        p_, U_, phi_, etc: State variables in OpenFOAM
    
    Output:
        kRes_/epsilonRes_: If solveTurbState_ == 0, update the residual field variable

        k_/epsilon_: If solveTurbState_ == 1, update them
    */

    // Copy and modify based on the "correct" function

    word divKScheme = "div(phi,k)";
    word divEpsilonScheme = "div(phi,epsilon)";

    label isPC = 0;

    label printToScreen = options.lookupOrDefault("printToScreen", 0);

    if (!solveTurbState_)
    {
        isPC = options.getLabel("isPC");

        if (isPC)
        {
            divKScheme = "div(pc)";
            divEpsilonScheme = "div(pc)";
        }
    }

    volScalarField rho = this->rho();

    // Note: for compressible flow, the "this->phi()" function divides phi by fvc:interpolate(rho),
    // while for the incompresssible "this->phi()" returns phi only
    // see src/TurbulenceModels/compressible/compressibleTurbulenceModel.C line 62 to 73
    volScalarField::Internal divU(
        fvc::div(fvc::absolute(phi_ / fvc::interpolate(rho), U_))().v());

    tmp<volTensorField> tgradU = fvc::grad(U_);
    volScalarField::Internal& G = const_cast<volScalarField::Internal&>(GPtr_());
    G = nut_.v() * (dev(twoSymm(tgradU().v())) && tgradU().v());
    tgradU.clear();

    if (solveTurbState_)
    {
        // Update epsilon and G at the wall
        epsilon_.boundaryFieldRef().updateCoeffs();
    }
    else
    {
        // special treatment for epsilon BC
        this->correctEpsilonBoundaryConditions();
    }

    // Dissipation equation
    tmp<fvScalarMatrix> epsEqn(
        fvm::ddt(phase_, rho, epsilon_)
            + fvm::div(phaseRhoPhi_, epsilon_, divEpsilonScheme)
            - fvm::laplacian(phase_ * rho * DepsilonEff(), epsilon_)
        == C1_ * phase_() * rho() * G * epsilon_() / k_() * betaFIEpsilon_()
            - fvm::SuSp((scalar(2.0 / 3.0) * C1_ - C3_) * phase_() * rho() * divU, epsilon_)
            - fvm::Sp(C2_ * phase_() * rho() * epsilon_() / k_(), epsilon_)
            + epsilonSource());

    epsEqn.ref().relax();

    epsEqn.ref().boundaryManipulate(epsilon_.boundaryFieldRef());

    if (solveTurbState_)
    {
        // get the solver performance info such as initial
        // and final residuals
        SolverPerformance<scalar> solverEpsilon = solve(epsEqn);
        DAUtility::primalResidualControl(solverEpsilon, printToScreen, "epsilon", daGlobalVar_.primalMaxRes);

        DAUtility::boundVar(allOptions_, epsilon_, printToScreen);
    }
    else
    {
        // reset the corrected omega near wall cell to its perturbed value
        this->setEpsilonNearWall();

        // calculate residuals
        epsilonRes_ = epsEqn() & epsilon_;
        // need to normalize residuals
        normalizeResiduals(epsilonRes);
    }

    // Turbulent kinetic energy equation
    tmp<fvScalarMatrix> kEqn(
        fvm::ddt(phase_, rho, k_)
            + fvm::div(phaseRhoPhi_, k_, divKScheme)
            - fvm::laplacian(phase_ * rho * DkEff(), k_)
        == phase_() * rho() * G * betaFIK_()
            - fvm::SuSp((2.0 / 3.0) * phase_() * rho() * divU, k_)
            - fvm::Sp(phase_() * rho() * epsilon_() / k_(), k_)
            + kSource());

    kEqn.ref().relax();

    if (solveTurbState_)
    {

        // get the solver performance info such as initial
        // and final residuals
        SolverPerformance<scalar> solverK = solve(kEqn);
        DAUtility::primalResidualControl(solverK, printToScreen, "k", daGlobalVar_.primalMaxRes);

        DAUtility::boundVar(allOptions_, k_, printToScreen);

        this->correctNut();
    }
    else
    {
        // calculate residuals
        kRes_ = kEqn() & k_;
        // need to normalize residuals
        normalizeResiduals(kRes);
    }

    return;
}

void DAkEpsilon::getFvMatrixFields(
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

    if (varName != "k" && varName != "epsilon")
    {
        FatalErrorIn(
            "varName not valid. It has to be k or epsilon")
            << exit(FatalError);
    }

    volScalarField rho = this->rho();

    // Note: for compressible flow, the "this->phi()" function divides phi by fvc:interpolate(rho),
    // while for the incompresssible "this->phi()" returns phi only
    // see src/TurbulenceModels/compressible/compressibleTurbulenceModel.C line 62 to 73
    volScalarField::Internal divU(
        fvc::div(fvc::absolute(phi_ / fvc::interpolate(rho), U_))().v());

    tmp<volTensorField> tgradU = fvc::grad(U_);
    volScalarField::Internal& G = const_cast<volScalarField::Internal&>(GPtr_());
    G = nut_.v() * (dev(twoSymm(tgradU().v())) && tgradU().v());
    tgradU.clear();

    // special treatment for epsilon BC
    this->correctEpsilonBoundaryConditions();

    if (varName == "epsilon")
    {
        // Dissipation equation
        fvScalarMatrix epsEqn(
            fvm::ddt(phase_, rho, epsilon_)
                + fvm::div(phaseRhoPhi_, epsilon_, "div(pc)")
                - fvm::laplacian(phase_ * rho * DepsilonEff(), epsilon_)
            == C1_ * phase_() * rho() * G * epsilon_() / k_() * betaFIEpsilon_()
                - fvm::SuSp((scalar(2.0 / 3.0) * C1_ - C3_) * phase_() * rho() * divU, epsilon_)
                - fvm::Sp(C2_ * phase_() * rho() * epsilon_() / k_(), epsilon_)
                + epsilonSource());

        epsEqn.relax();

        // reset the corrected omega near wall cell to its perturbed value
        this->setEpsilonNearWall();

        diag = epsEqn.D();
        upper = epsEqn.upper();
        lower = epsEqn.lower();
    }
    else if (varName == "k")
    {
        fvScalarMatrix kEqn(
            fvm::ddt(phase_, rho, k_)
                + fvm::div(phaseRhoPhi_, k_, "div(pc)")
                - fvm::laplacian(phase_ * rho * DkEff(), k_)
            == phase_() * rho() * G * betaFIK_()
                - fvm::SuSp((2.0 / 3.0) * phase_() * rho() * divU, k_)
                - fvm::Sp(phase_() * rho() * epsilon_() / k_(), k_)
                + kSource());

        kEqn.relax();

        diag = kEqn.D();
        upper = kEqn.upper();
        lower = kEqn.lower();
    }
}

void DAkEpsilon::getTurbProdOverDestruct(volScalarField& PoD) const
{
    /*
    Description:
        Return the value of the production over destruction term from the turbulence model 
    */
    tmp<volTensorField> tgradU = fvc::grad(U_);
    volScalarField::Internal& G = const_cast<volScalarField::Internal&>(GPtr_());
    G = nut_.v() * (dev(twoSymm(tgradU().v())) && tgradU().v());

    volScalarField rho = this->rho();

    volScalarField::Internal P = phase_() * rho() * G;
    volScalarField::Internal D = phase_() * rho() * epsilon_();

    forAll(P, cellI)
    {
        PoD[cellI] = P[cellI] / (D[cellI] + P[cellI] + 1e-16);
    }
}

void DAkEpsilon::getTurbConvOverProd(volScalarField& CoP) const
{
    /*
    Description:
        Return the value of the convective over production term from the turbulence model 
    */

    tmp<volTensorField> tgradU = fvc::grad(U_);
    volScalarField::Internal& G = const_cast<volScalarField::Internal&>(GPtr_());
    G = nut_.v() * (dev(twoSymm(tgradU().v())) && tgradU().v());

    volScalarField rho = this->rho();

    volScalarField::Internal P = phase_() * rho() * G;
    volScalarField C = fvc::div(phaseRhoPhi_, k_);

    forAll(P, cellI)
    {
        CoP[cellI] = C[cellI] / (P[cellI] + C[cellI] + 1e-16);
    }
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
