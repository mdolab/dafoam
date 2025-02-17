/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    This file is modified from OpenFOAM's source code
    src/TurbulenceModels/turbulenceModels/RAS/kOmega/kOmega.C

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

#include "DAkOmega.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAkOmega, 0);
addToRunTimeSelectionTable(DATurbulenceModel, DAkOmega, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAkOmega::DAkOmega(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption)
    : DATurbulenceModel(modelType, mesh, daOption),
      // kOmega parameters
      Cmu_(dimensioned<scalar>::lookupOrAddToDict(
          "betaStar",
          this->coeffDict_,
          0.09)),
      beta_(dimensioned<scalar>::lookupOrAddToDict(
          "beta",
          this->coeffDict_,
          0.072)),
      gamma_(dimensioned<scalar>::lookupOrAddToDict(
          "gamma",
          this->coeffDict_,
          0.52)),
      alphaK_(dimensioned<scalar>::lookupOrAddToDict(
          "alphaK",
          this->coeffDict_,
          0.5)),
      alphaOmega_(dimensioned<scalar>::lookupOrAddToDict(
          "alphaOmega",
          this->coeffDict_,
          0.5)),
      // Augmented variables
      omega_(const_cast<volScalarField&>(
          mesh_.thisDb().lookupObject<volScalarField>("omega"))),
      omegaRes_(
          IOobject(
              "omegaRes",
              mesh.time().timeName(),
              mesh,
              IOobject::NO_READ,
              IOobject::NO_WRITE),
          mesh,
          dimensionedScalar("omegaRes", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
          zeroGradientFvPatchField<scalar>::typeName),
      k_(
          const_cast<volScalarField&>(
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
      betaFIOmega_(
          IOobject(
              "betaFIOmega",
              mesh.time().timeName(),
              mesh,
              IOobject::READ_IF_PRESENT,
              IOobject::AUTO_WRITE),
          mesh,
          dimensionedScalar("betaFIOmega", dimensionSet(0, 0, 0, 0, 0, 0, 0), 1.0),
          "zeroGradient")
{

    if (turbModelType_ == "incompressible")
    {
        omegaRes_.dimensions().reset(dimensionSet(0, 0, -2, 0, 0, 0, 0));
        kRes_.dimensions().reset(dimensionSet(0, 2, -3, 0, 0, 0, 0));
    }
    else if (turbModelType_ == "compressible")
    {
        omegaRes_.dimensions().reset(dimensionSet(1, -3, -2, 0, 0, 0, 0));
        kRes_.dimensions().reset(dimensionSet(1, -1, -3, 0, 0, 0, 0));
    }

    // calculate the size of omegaWallFunction faces
    label nWallFaces = 0;
    forAll(omega_.boundaryField(), patchI)
    {
        if (omega_.boundaryField()[patchI].type() == "omegaWallFunction"
            && omega_.boundaryField()[patchI].size() > 0)
        {
            forAll(omega_.boundaryField()[patchI], faceI)
            {
                nWallFaces++;
            }
        }
    }

    // initialize omegaNearWall
    omegaNearWall_.setSize(nWallFaces);

    // initialize the G field
    tmp<volTensorField> tgradU = fvc::grad(U_);
    GPtr_.reset(new volScalarField("kOmega:G", nut_ * (tgradU() && dev(twoSymm(tgradU())))));
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Augmented functions
void DAkOmega::correctModelStates(wordList& modelStates) const
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
        
        We don't udpate the names for the radiation model because users are 
        supposed to set modelStates={"G"}
    */

    // For SST model, we need to replace nut with k, omega

    forAll(modelStates, idxI)
    {
        word stateName = modelStates[idxI];
        if (stateName == "nut")
        {
            modelStates[idxI] = "omega";
            modelStates.append("k");
        }
    }
}

void DAkOmega::correctNut()
{
    /*
    Description:
        Update nut based on other turbulence variables and update the BCs
        Also update alphat if is present
    */

    nut_ = k_ / omega_;

    nut_.correctBoundaryConditions(); // nutkWallFunction: update wall face nut based on k

    // this is basically BasicTurbulenceModel::correctNut();
    this->correctAlphat();

    return;
}

void DAkOmega::correctBoundaryConditions()
{
    /*
    Description:
        Update turbulence variable boundary values
    */

    // correct the BCs for the perturbed fields
    // kqWallFunction is a zero-gradient BC
    k_.correctBoundaryConditions();
}

void DAkOmega::correctOmegaBoundaryConditions()
{
    /*
    Description:
        this is a special treatment for omega BC because we cant directly call omega.
        correctBoundaryConditions() because it will modify the internal omega and G that 
        are right next to walls. This will mess up adjoint Jacobians
        To solve this issue,
        1. we store the near wall omega before calling omega.correctBoundaryConditions()
        2. call omega.correctBoundaryConditions()
        3. Assign the stored near wall omega back
        4. Apply a zeroGradient BC for omega at the wall patches
        *********** NOTE *************
        this treatment will obviously downgrade the accuracy of adjoint derivative since it is 
        not 100% consistent with what is used for the flow solver; however, our observation 
        shows that the impact is not very large.
    */

    // save the perturbed omega at the wall
    this->saveOmegaNearWall();
    // correct omega boundary conditions, this includes updating wall face and near wall omega values,
    // updating the inter-proc BCs
    omega_.correctBoundaryConditions();
    // reset the corrected omega near wall cell to its perturbed value
    this->setOmegaNearWall();
}

void DAkOmega::saveOmegaNearWall()
{
    /*
    Description:
        Save the current near wall omega values to omegaNearWall_
    */
    label counterI = 0;
    forAll(omega_.boundaryField(), patchI)
    {
        if (omega_.boundaryField()[patchI].type() == "omegaWallFunction"
            and omega_.boundaryField()[patchI].size() > 0)
        {
            const UList<label>& faceCells = mesh_.boundaryMesh()[patchI].faceCells();
            forAll(faceCells, faceI)
            {
                //Info<<"Near Wall cellI: "<<faceCells[faceI]<<endl;
                omegaNearWall_[counterI] = omega_[faceCells[faceI]];
                counterI++;
            }
        }
    }
    return;
}

void DAkOmega::setOmegaNearWall()
{
    /*
    Description:
        Set the current near wall omega values from omegaNearWall_
        Here we also apply a zeroGradient BC to the wall faces
    */
    label counterI = 0;
    forAll(omega_.boundaryField(), patchI)
    {
        if (omega_.boundaryField()[patchI].type() == "omegaWallFunction"
            && omega_.boundaryField()[patchI].size() > 0)
        {
            const UList<label>& faceCells = mesh_.boundaryMesh()[patchI].faceCells();
            forAll(faceCells, faceI)
            {
                omega_[faceCells[faceI]] = omegaNearWall_[counterI];
                // zeroGradient BC
                omega_.boundaryFieldRef()[patchI][faceI] = omega_[faceCells[faceI]];
                counterI++;
            }
        }
    }
    return;
}

void DAkOmega::updateIntermediateVariables()
{
    /*
    Description:
        Update nut based on nuTilda. Note: we need to update nut and its BC since we 
        may have perturbed other turbulence vars that affect the nut values
    */

    this->correctNut();
}

void DAkOmega::correctStateResidualModelCon(List<List<word>>& stateCon) const
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

    label stateConSize = stateCon.size();
    forAll(stateCon, idxI)
    {
        label addUCon = 0;
        forAll(stateCon[idxI], idxJ)
        {
            word conStateName = stateCon[idxI][idxJ];
            if (conStateName == "nut")
            {
                stateCon[idxI][idxJ] = "omega"; // replace nut with omega
                stateCon[idxI].append("k"); // also add k for that level
                addUCon = 1;
            }
        }
        // add U for the current level and level+1 if it is not there yet
        label isU;
        if (addUCon == 1)
        {
            // first add U for the current level
            isU = 0;
            forAll(stateCon[idxI], idxJ)
            {
                word conStateName = stateCon[idxI][idxJ];
                if (conStateName == "U")
                {
                    isU = 1;
                }
            }
            if (!isU)
            {
                stateCon[idxI].append("U");
            }

            // now add U for level+1 if idxI is not the largest level
            // if idxI is already the largest level, we have a problem
            if (idxI != stateConSize - 1)
            {
                isU = 0;
                forAll(stateCon[idxI + 1], idxJ)
                {
                    word conStateName = stateCon[idxI + 1][idxJ];
                    if (conStateName == "U")
                    {
                        isU = 1;
                    }
                }
                if (!isU)
                {
                    stateCon[idxI + 1].append("U");
                }
            }
            else
            {
                FatalErrorIn(
                    "In DAStateInfo, nut shows in the largest connectivity level! "
                    "This is not supported!")
                    << exit(FatalError);
            }
        }
    }
}

void DAkOmega::addModelResidualCon(HashTable<List<List<word>>>& allCon) const
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
            "omegaRes",
            {
                {"U", "omega", "k", "phi"}, // lv0
                {"U", "omega", "k"}, // lv1
                {"U", "omega", "k"} // lv2
            });
        allCon.set(
            "kRes",
            {
                {"U", "omega", "k", "phi"}, // lv0
                {"U", "omega", "k"}, // lv1
                {"U", "omega", "k"} // lv2
            });
    }
    else if (turbModelType_ == "compressible")
    {
        allCon.set(
            "omegaRes",
            {
                {"U", "T", pName, "omega", "k", "phi"}, // lv0
                {"U", "T", pName, "omega", "k"}, // lv1
                {"U", "T", pName, "omega", "k"} // lv2
            });
        allCon.set(
            "kRes",
            {
                {"U", "T", pName, "omega", "k", "phi"}, // lv0
                {"U", "T", pName, "omega", "k"}, // lv1
                {"U", "T", pName, "omega", "k"} // lv2
            });
    }
}

void DAkOmega::correct(label printToScreen)
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

void DAkOmega::calcResiduals(const dictionary& options)
{
    /*
    Descroption:
        If solveTurbState_ == 1, this function solve and update k and omega, and 
        is the same as calling turbulence.correct(). If solveTurbState_ == 0,
        this function compute residuals for turbulence variables, e.g., nuTildaRes_

    Input:
        options.isPC: 1 means computing residuals for preconditioner matrix.
        This essentially use the first order scheme for div(phi,nuTilda)

        p_, U_, phi_, etc: State variables in OpenFOAM
    
    Output:
        kRes_/omegaRes_: If solveTurbState_ == 0, update the residual field variable

        k_/omega_: If solveTurbState_ == 1, update them
    */

    // Copy and modify based on the "correct" function

    word divKScheme = "div(phi,k)";
    word divOmegaScheme = "div(phi,omega)";

    label isPC = 0;

    label printToScreen = options.lookupOrDefault("printToScreen", 0);

    if (!solveTurbState_)
    {
        isPC = options.getLabel("isPC");

        if (isPC)
        {
            divKScheme = "div(pc)";
            divOmegaScheme = "div(pc)";
        }
    }

    volScalarField rho = this->rho();

    // Note: for compressible flow, the "this->phi()" function divides phi by fvc:interpolate(rho),
    // while for the incompresssible "this->phi()" returns phi only
    // see src/TurbulenceModels/compressible/compressibleTurbulenceModel.C line 62 to 73
    volScalarField divU(fvc::div(fvc::absolute(phi_ / fvc::interpolate(rho), U_)));

    tmp<volTensorField> tgradU = fvc::grad(U_);
    volScalarField& G = const_cast<volScalarField&>(GPtr_());
    G = nut_ * (tgradU() && dev(twoSymm(tgradU())));
    tgradU.clear();

    if (solveTurbState_)
    {
        // Update omega and G at the wall
        omega_.boundaryFieldRef().updateCoeffs();
    }
    else
    {
        // NOTE instead of calling omega_.boundaryFieldRef().updateCoeffs();
        // here we call our self-defined boundary conditions
        this->correctOmegaBoundaryConditions();
    }

    // Turbulent frequency equation
    tmp<fvScalarMatrix> omegaEqn(
        fvm::ddt(phase_, rho, omega_)
            + fvm::div(phaseRhoPhi_, omega_, divOmegaScheme)
            - fvm::laplacian(phase_ * rho * DomegaEff(), omega_)
        == gamma_ * phase_ * rho * G * omega_ / k_ * betaFIOmega_
            - fvm::SuSp(scalar(2.0 / 3.0) * gamma_ * phase_ * rho * divU, omega_)
            - fvm::Sp(beta_ * phase_ * rho * omega_, omega_));

    omegaEqn.ref().relax();
    omegaEqn.ref().boundaryManipulate(omega_.boundaryFieldRef());

    if (solveTurbState_)
    {
        // get the solver performance info such as initial
        // and final residuals
        SolverPerformance<scalar> solverOmega = solve(omegaEqn);
        DAUtility::primalResidualControl(solverOmega, printToScreen, "omega", daGlobalVar_.primalMaxRes);

        DAUtility::boundVar(allOptions_, omega_, printToScreen);
    }
    else
    {
        // reset the corrected omega near wall cell to its perturbed value
        this->setOmegaNearWall();

        // calculate residuals
        omegaRes_ = omegaEqn() & omega_;
        // need to normalize residuals
        normalizeResiduals(omegaRes);
    }

    // Turbulent kinetic energy equation
    tmp<fvScalarMatrix> kEqn(
        fvm::ddt(phase_, rho, k_)
            + fvm::div(phaseRhoPhi_, k_, divKScheme)
            - fvm::laplacian(phase_ * rho * DkEff(), k_)
        == phase_ * rho * G * betaFIK_
            - fvm::SuSp((2.0 / 3.0) * phase_ * rho * divU, k_)
            - fvm::Sp(Cmu_ * phase_ * rho * omega_, k_));

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

void DAkOmega::getFvMatrixFields(
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

    if (varName != "k" && varName != "omega")
    {
        FatalErrorIn(
            "varName not valid. It has to be k or omega")
            << exit(FatalError);
    }

    volScalarField rho = this->rho();

    // Note: for compressible flow, the "this->phi()" function divides phi by fvc:interpolate(rho),
    // while for the incompresssible "this->phi()" returns phi only
    // see src/TurbulenceModels/compressible/compressibleTurbulenceModel.C line 62 to 73
    volScalarField divU(fvc::div(fvc::absolute(phi_ / fvc::interpolate(rho), U_)));

    tmp<volTensorField> tgradU = fvc::grad(U_);
    volScalarField& G = const_cast<volScalarField&>(GPtr_());
    G = nut_ * (tgradU() && dev(twoSymm(tgradU())));
    tgradU.clear();

    // NOTE instead of calling omega_.boundaryFieldRef().updateCoeffs();
    // here we call our self-defined boundary conditions
    this->correctOmegaBoundaryConditions();

    if (varName == "omega")
    {
        // Turbulent frequency equation
        fvScalarMatrix omegaEqn(
            fvm::ddt(phase_, rho, omega_)
                + fvm::div(phaseRhoPhi_, omega_, "div(pc)")
                - fvm::laplacian(phase_ * rho * DomegaEff(), omega_)
            == gamma_ * phase_ * rho * G * omega_ / k_ * betaFIOmega_
                - fvm::SuSp(scalar(2.0 / 3.0) * gamma_ * phase_ * rho * divU, omega_)
                - fvm::Sp(beta_ * phase_ * rho * omega_, omega_));

        omegaEqn.relax();

        // reset the corrected omega near wall cell to its perturbed value
        this->setOmegaNearWall();

        diag = omegaEqn.D();
        upper = omegaEqn.upper();
        lower = omegaEqn.lower();
    }
    else if (varName == "k")
    {
        // Turbulent kinetic energy equation
        fvScalarMatrix kEqn(
            fvm::ddt(phase_, rho, k_)
                + fvm::div(phaseRhoPhi_, k_, "div(pc)")
                - fvm::laplacian(phase_ * rho * DkEff(), k_)
            == phase_ * rho * G * betaFIK_
                - fvm::SuSp((2.0 / 3.0) * phase_ * rho * divU, k_)
                - fvm::Sp(Cmu_ * phase_ * rho * omega_, k_));

        kEqn.relax();

        diag = kEqn.D();
        upper = kEqn.upper();
        lower = kEqn.lower();
    }
}

void DAkOmega::getTurbProdOverDestruct(volScalarField& PoD) const
{
    /*
    Description:
        Return the value of the production over destruction term from the turbulence model 
    */
    tmp<volTensorField> tgradU = fvc::grad(U_);
    volScalarField& G = const_cast<volScalarField&>(GPtr_());
    G = nut_ * (tgradU() && dev(twoSymm(tgradU())));

    volScalarField rho = this->rho();

    volScalarField P = phase_ * rho * G;
    volScalarField D = Cmu_ * phase_ * rho * omega_ * k_;

    forAll(P, cellI)
    {
        PoD[cellI] = P[cellI] / (D[cellI] + P[cellI] + 1e-16);
    }
}

void DAkOmega::getTurbConvOverProd(volScalarField& CoP) const
{
    /*
    Description:
        Return the value of the convective over production term from the turbulence model 
    */

    tmp<volTensorField> tgradU = fvc::grad(U_);
    volScalarField& G = const_cast<volScalarField&>(GPtr_());
    G = nut_ * (tgradU() && dev(twoSymm(tgradU())));

    volScalarField rho = this->rho();

    volScalarField P = phase_ * rho * G;
    volScalarField C = fvc::div(phaseRhoPhi_, k_);

    forAll(P, cellI)
    {
        CoP[cellI] = C[cellI] / (P[cellI] + C[cellI] + 1e-16);
    }
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
