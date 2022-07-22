/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

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

#include "DAkOmegaSSTFIML.H"
#include "IFstream.H"
// not sure if these are necessary..
#include <vector>
#include <math.h>
#include <omp.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAkOmegaSSTFIML, 0);
addToRunTimeSelectionTable(DATurbulenceModel, DAkOmegaSSTFIML, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAkOmegaSSTFIML::DAkOmegaSSTFIML(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption)
    : DATurbulenceModel(modelType, mesh, daOption),
      // SST parameters
      alphaK1_(dimensioned<scalar>::lookupOrAddToDict(
          "alphaK1",
          this->coeffDict_,
          0.85)),
      alphaK2_(dimensioned<scalar>::lookupOrAddToDict(
          "alphaK2",
          this->coeffDict_,
          1.0)),
      alphaOmega1_(dimensioned<scalar>::lookupOrAddToDict(
          "alphaOmega1",
          this->coeffDict_,
          0.5)),
      alphaOmega2_(dimensioned<scalar>::lookupOrAddToDict(
          "alphaOmega2",
          this->coeffDict_,
          0.856)),
      gamma1_(dimensioned<scalar>::lookupOrAddToDict(
          "gamma1",
          this->coeffDict_,
          5.0 / 9.0)),
      gamma2_(dimensioned<scalar>::lookupOrAddToDict(
          "gamma2",
          this->coeffDict_,
          0.44)),
      beta1_(dimensioned<scalar>::lookupOrAddToDict(
          "beta1",
          this->coeffDict_,
          0.075)),
      beta2_(dimensioned<scalar>::lookupOrAddToDict(
          "beta2",
          this->coeffDict_,
          0.0828)),
      betaStar_(dimensioned<scalar>::lookupOrAddToDict(
          "betaStar",
          this->coeffDict_,
          0.09)),
      a1_(dimensioned<scalar>::lookupOrAddToDict(
          "a1",
          this->coeffDict_,
          0.31)),
      b1_(dimensioned<scalar>::lookupOrAddToDict(
          "b1",
          this->coeffDict_,
          1.0)),
      c1_(dimensioned<scalar>::lookupOrAddToDict(
          "c1",
          this->coeffDict_,
          10.0)),
      F3_(Switch::lookupOrAddToDict(
          "F3",
          this->coeffDict_,
          false)),
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
#ifdef CompressibleFlow
          dimensionedScalar("omegaRes", dimensionSet(1, -3, -2, 0, 0, 0, 0), 0.0),
#endif
#ifdef IncompressibleFlow
          dimensionedScalar("omegaRes", dimensionSet(0, 0, -2, 0, 0, 0, 0), 0.0),
#endif
          zeroGradientFvPatchField<scalar>::typeName),
      omegaResRef_(
          IOobject(
              "omegaResRef",
              mesh.time().timeName(),
              mesh,
              IOobject::NO_READ,
              IOobject::NO_WRITE),
          omegaRes_),
      omegaResPartDeriv_(
          IOobject(
              "omegaResPartDeriv",
              mesh.time().timeName(),
              mesh,
              IOobject::NO_READ,
              IOobject::NO_WRITE),
          omegaRes_),
      omegaRef_(
          IOobject(
              "omegaRef",
              mesh.time().timeName(),
              mesh,
              IOobject::NO_READ,
              IOobject::NO_WRITE),
          omega_),
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
#ifdef CompressibleFlow
          dimensionedScalar("kRes", dimensionSet(1, -1, -3, 0, 0, 0, 0), 0.0),
#endif
#ifdef IncompressibleFlow
          dimensionedScalar("kRes", dimensionSet(0, 2, -3, 0, 0, 0, 0), 0.0),
#endif
          zeroGradientFvPatchField<scalar>::typeName),
      kResRef_(
          IOobject(
              "kResRef",
              mesh.time().timeName(),
              mesh,
              IOobject::NO_READ,
              IOobject::NO_WRITE),
          kRes_),
      kResPartDeriv_(
          IOobject(
              "kResPartDeriv",
              mesh.time().timeName(),
              mesh,
              IOobject::NO_READ,
              IOobject::NO_WRITE),
          kRes_),
      kRef_(
          IOobject(
              "kRef",
              mesh.time().timeName(),
              mesh,
              IOobject::NO_READ,
              IOobject::NO_WRITE),
          k_),
      /// field inversion parameters
      betaFieldInversion_(
          IOobject(
              "betaFieldInversion",
              mesh.time().timeName(),
              mesh_,
              IOobject::READ_IF_PRESENT,
              IOobject::AUTO_WRITE),
          mesh_,
          dimensionedScalar("betaFieldInversion", dimensionSet(0, 0, 0, 0, 0, 0, 0), 1.0),
          zeroGradientFvPatchScalarField::typeName),
      betaFieldInversionML_(
          IOobject(
              "betaFieldInversionML",
              mesh.time().timeName(),
              mesh_,
              IOobject::NO_READ,
              IOobject::AUTO_WRITE),
          mesh_,
          dimensionedScalar("betaFieldInversionML", dimensionSet(0, 0, 0, 0, 0, 0, 0), 1.0),
          zeroGradientFvPatchScalarField::typeName),
      QCriterion_(
          IOobject(
              "QCriterion",
              mesh.time().timeName(),
              mesh_,
              IOobject::NO_READ,
              IOobject::AUTO_WRITE),
          mesh_,
          dimensionedScalar("QCriterion", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
          zeroGradientFvPatchScalarField::typeName),
      p_(const_cast<volScalarField&>(
          mesh_.thisDb().lookupObject<volScalarField>("p"))),
      pGradAlongStream_(
          IOobject(
              "pGradAlongStream",
              mesh.time().timeName(),
              mesh_,
              IOobject::NO_READ,
              IOobject::AUTO_WRITE),
          mesh_,
          dimensionedScalar("pressureAlongStream", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
          zeroGradientFvPatchScalarField::typeName),
      turbulenceIntensity_(
          IOobject(
              "turbulenceIntensity",
              mesh.time().timeName(),
              mesh_,
              IOobject::NO_READ,
              IOobject::AUTO_WRITE),
          mesh_,
          dimensionedScalar("turbulenceIntensity", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
          zeroGradientFvPatchScalarField::typeName),
      transportProperties_(
          IOobject(
              "transportProperties",
              mesh.time().constant(),
              mesh_,
              IOobject::MUST_READ,
              IOobject::NO_WRITE)),
      ReT_(
          IOobject(
              "ReT",
              mesh.time().timeName(),
              mesh_,
              IOobject::NO_READ,
              IOobject::AUTO_WRITE),
          mesh_,
          dimensionedScalar("ReT", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
          zeroGradientFvPatchScalarField::typeName),
      convectionTKE_(
          IOobject(
              "convectionTKE",
              mesh.time().timeName(),
              mesh_,
              IOobject::NO_READ,
              IOobject::AUTO_WRITE),
          mesh_,
          dimensionedScalar("convectionTKE", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
          zeroGradientFvPatchScalarField::typeName),
      tauRatio_(
          IOobject(
              "tauRatio",
              mesh.time().timeName(),
              mesh_,
              IOobject::NO_READ,
              IOobject::AUTO_WRITE),
          mesh_,
          dimensionedScalar("tauRatio", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
          zeroGradientFvPatchScalarField::typeName),
      pressureStress_(
          IOobject(
              "pressureStress",
              mesh.time().timeName(),
              mesh_,
              IOobject::NO_READ,
              IOobject::AUTO_WRITE),
          mesh_,
          dimensionedScalar("pressureStress", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
          zeroGradientFvPatchScalarField::typeName),
      curvature_(
          IOobject(
              "curvature",
              mesh.time().timeName(),
              mesh_,
              IOobject::NO_READ,
              IOobject::AUTO_WRITE),
          mesh_,
          dimensionedScalar("curvature", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
          zeroGradientFvPatchScalarField::typeName),
      UGradMisalignment_(
          IOobject(
              "UGradMisalignment",
              mesh.time().timeName(),
              mesh_,
              IOobject::NO_READ,
              IOobject::AUTO_WRITE),
          mesh_,
          dimensionedScalar("UGradMisalignment", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
          zeroGradientFvPatchScalarField::typeName),
      y_(mesh_.thisDb().lookupObject<volScalarField>("yWall"))
{

    // initialize printInterval_ we need to check whether it is a steady state
    // or unsteady primal solver
    IOdictionary fvSchemes(
        IOobject(
            "fvSchemes",
            mesh.time().system(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false));
    word ddtScheme = word(fvSchemes.subDict("ddtSchemes").lookup("default"));
    if (ddtScheme == "steadyState")
    {
        printInterval_ =
            daOption.getAllOptions().lookupOrDefault<label>("printInterval", 100);
    }
    else
    {
        printInterval_ =
            daOption.getAllOptions().lookupOrDefault<label>("printIntervalUnsteady", 500);
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

    // read in the tensor flow graph
    graph_ = tf_utils::LoadGraph("./kOmegaSSTFIML.pb");
    input_ph_ = {TF_GraphOperationByName(graph_, "input_placeholder"), 0};
    output_ = {TF_GraphOperationByName(graph_, "output_value/BiasAdd"), 0};
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// SA member functions. these functions are copied from
tmp<volScalarField> DAkOmegaSSTFIML::F1(
    const volScalarField& CDkOmega) const
{

    tmp<volScalarField> CDkOmegaPlus = max(
        CDkOmega,
        dimensionedScalar("1.0e-10", dimless / sqr(dimTime), 1.0e-10));

    tmp<volScalarField> arg1 = min(
        min(
            max(
                (scalar(1) / betaStar_) * sqrt(k_) / (omega_ * y_),
                scalar(500) * (this->nu()) / (sqr(y_) * omega_)),
            (scalar(4) * alphaOmega2_) * k_ / (CDkOmegaPlus * sqr(y_))),
        scalar(10));

    return tanh(pow4(arg1));
}

tmp<volScalarField> DAkOmegaSSTFIML::F2() const
{

    tmp<volScalarField> arg2 = min(
        max(
            (scalar(2) / betaStar_) * sqrt(k_) / (omega_ * y_),
            scalar(500) * (this->nu()) / (sqr(y_) * omega_)),
        scalar(100));

    return tanh(sqr(arg2));
}

tmp<volScalarField> DAkOmegaSSTFIML::F3() const
{

    tmp<volScalarField> arg3 = min(
        150 * (this->nu()) / (omega_ * sqr(y_)),
        scalar(10));

    return 1 - tanh(pow4(arg3));
}

tmp<volScalarField> DAkOmegaSSTFIML::F23() const
{
    tmp<volScalarField> f23(F2());

    if (F3_)
    {
        f23.ref() *= F3();
    }

    return f23;
}

tmp<volScalarField::Internal> DAkOmegaSSTFIML::GbyNu(
    const volScalarField::Internal& GbyNu0,
    const volScalarField::Internal& F2,
    const volScalarField::Internal& S2) const
{
    return min(
        GbyNu0,
        (c1_ / a1_) * betaStar_ * omega_()
            * max(a1_ * omega_(), b1_ * F2 * sqrt(S2)));
}

tmp<volScalarField::Internal> DAkOmegaSSTFIML::Pk(
    const volScalarField::Internal& G) const
{
    return min(G, (c1_ * betaStar_) * k_() * omega_());
}

tmp<volScalarField::Internal> DAkOmegaSSTFIML::epsilonByk(
    const volScalarField& F1,
    const volTensorField& gradU) const
{
    return betaStar_ * omega_();
}

tmp<fvScalarMatrix> DAkOmegaSSTFIML::kSource() const
{
    const volScalarField& rho = rho_;
    return tmp<fvScalarMatrix>(
        new fvScalarMatrix(
            k_,
            dimVolume * rho.dimensions() * k_.dimensions() / dimTime));
}

tmp<fvScalarMatrix> DAkOmegaSSTFIML::omegaSource() const
{
    const volScalarField& rho = rho_;
    return tmp<fvScalarMatrix>(
        new fvScalarMatrix(
            omega_,
            dimVolume * rho.dimensions() * omega_.dimensions() / dimTime));
}

tmp<fvScalarMatrix> DAkOmegaSSTFIML::Qsas(
    const volScalarField::Internal& S2,
    const volScalarField::Internal& gamma,
    const volScalarField::Internal& beta) const
{
    const volScalarField& rho = rho_;
    return tmp<fvScalarMatrix>(
        new fvScalarMatrix(
            omega_,
            dimVolume * rho.dimensions() * omega_.dimensions() / dimTime));
}

// Augmented functions
void DAkOmegaSSTFIML::correctModelStates(wordList& modelStates) const
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

void DAkOmegaSSTFIML::correctNut()
{
    /*
    Description:
        Update nut based on other turbulence variables and update the BCs
        Also update alphat if is present
    */

    const volVectorField U = mesh_.thisDb().lookupObject<volVectorField>("U");
    tmp<volTensorField> tgradU = fvc::grad(U);
    volScalarField S2(2 * magSqr(symm(tgradU())));

    nut_ = a1_ * k_ / max(a1_ * omega_, b1_ * F23() * sqrt(S2));

    nut_.correctBoundaryConditions(); // nutkWallFunction: update wall face nut based on k

    // this is basically BasicTurbulenceModel::correctNut();
    this->correctAlphat();

    return;
}

void DAkOmegaSSTFIML::correctBoundaryConditions()
{
    /*
    Description:
        Update turbulence variable boundary values
    */

    // correct the BCs for the perturbed fields
    // kqWallFunction is a zero-gradient BC
    k_.correctBoundaryConditions();
}

void DAkOmegaSSTFIML::correctOmegaBoundaryConditions()
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

void DAkOmegaSSTFIML::saveOmegaNearWall()
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

void DAkOmegaSSTFIML::setOmegaNearWall()
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

void DAkOmegaSSTFIML::updateIntermediateVariables()
{
    /*
    Description:
        Update nut based on nuTilda. Note: we need to update nut and its BC since we 
        may have perturbed other turbulence vars that affect the nut values
    */

    this->correctNut();
}

void DAkOmegaSSTFIML::correctStateResidualModelCon(List<List<word>>& stateCon) const
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

void DAkOmegaSSTFIML::addModelResidualCon(HashTable<List<List<word>>>& allCon) const
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
#ifdef IncompressibleFlow
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
#endif

#ifdef CompressibleFlow
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
#endif
}

void DAkOmegaSSTFIML::correct()
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
    this->calcResiduals(dummyOptions);
    // after it, we reset solveTurbState_ = 0 such that calcResiduals will not
    // update nuTilda when calling from the adjoint class, i.e., solveAdjoint from DASolver.
    solveTurbState_ = 0;
}

void DAkOmegaSSTFIML::calcResiduals(const dictionary& options)
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

    label printToScreen = this->isPrintTime(mesh_.time(), printInterval_);

    word divKScheme = "div(phi,k)";
    word divOmegaScheme = "div(phi,omega)";

    label isPC = 0;

    if (!solveTurbState_)
    {
        isPC = options.getLabel("isPC");

        if (isPC)
        {
            divKScheme = "div(pc)";
            divOmegaScheme = "div(pc)";
        }
    }

    // Read scaling parameters (do we need to scale?)
    RectangularMatrix<doubleScalar> meanStdVals(IFstream("means")());

    label numInputs = 9;
    label numOutputs = 1;
    scalar meanArray[numInputs + numOutputs] = {0};
    scalar stdArray[numInputs + numOutputs] = {0};

    for (label i = 0; i <= numInputs + numOutputs; i++)
    {
        meanArray[i] = meanStdVals(0, i);
        stdArray[i] = meanStdVals(1, i);
    }

    // COMPUTE MACHINE LEARNING FEATURES
    //////////////////////////Q-criterion//////////////////////////////////
    volTensorField UGrad(fvc::grad(U_));
    volTensorField Omega("Omega", skew(UGrad));
    volScalarField magOmegaSqr(magSqr(Omega));
    volSymmTensorField S("S", symm(UGrad));
    volScalarField magS(mag(S));
    volScalarField magSSqr(magSqr(S));
    QCriterion_ = (magOmegaSqr - magSSqr) / (magOmegaSqr + magSSqr);

    //////////////////////////pGradAlongStream////////////////////////////
    volVectorField pGrad("gradP", fvc::grad(p_));
    volScalarField pG_denominator(mag(U_) * mag(pGrad) + mag(U_ & pGrad));
    pGradAlongStream_ = (U_ & pGrad) / Foam::max(pG_denominator, dimensionedScalar("minpG", dimensionSet(0, 2, -3, 0, 0, 0, 0), SMALL));

    //////////////////////////Turbulence intensity/////////////////////////
    turbulenceIntensity_ = k_ / (0.5 * (U_ & U_) + k_);

    //////////////////////////ReT/////////////////////////////////////////
    //dimensionedScalar refViscosity(this->nu());
    dimensionedScalar maxReT("maxReT", dimless, 2.0);
    ReT_ = Foam::min((sqrt(k_) * y_) / (scalar(50.0) * this->nu()), maxReT);

    //////////////////////////Convection TKE///////////////////////////////
    volSymmTensorField tau(2.0 / 3.0 * I * k_ - nut_ * twoSymm(fvc::grad(U_)));
    volVectorField kGrad("gradK", fvc::grad(k_));
    convectionTKE_ = (U_ & kGrad) / (mag(tau && S) + mag(U_ & kGrad));

    //////////////////////////tauRatio////////////////////////////////////
    tauRatio_ = mag(tau) / (k_ + mag(tau));

    //////////////////////////pressure stress/////////////////////////////
    volVectorField diagUGrad(
        IOobject("diagUGrad",
                 mesh_.time().timeName(),
                 mesh_,
                 IOobject::NO_READ,
                 IOobject::NO_WRITE),
        mesh_,
        dimensionedVector("diagUGrad", dimensionSet(0, 0, 0, 0, 0, 0, 0), Foam::vector(0, 0, 0)),
        zeroGradientFvPatchScalarField::typeName);

    forAll(mesh_.C(), cI)
    {
        diagUGrad[cI].component(0) = UGrad[cI].xx();
        diagUGrad[cI].component(1) = UGrad[cI].yy();
        diagUGrad[cI].component(2) = UGrad[cI].zz();
        pressureStress_[cI] = mag(pGrad[cI]) / (mag(pGrad[cI]) + mag(3.0 * cmptAv(U_[cI] & diagUGrad[cI])));
    }

    //////////////////////////Curvature//////////////////////////////////
    forAll(mesh_.C(), cI)
    {
        curvature_[cI] = mag(U_[cI] & UGrad[cI]) / (mag(U_[cI] & U_[cI]) + mag(U_[cI] & UGrad[cI]));
    }

    //////////////////////////UGradMisalignment//////////////////////////////////
    forAll(mesh_.C(), cI)
    {
        UGradMisalignment_[cI] = mag(U_[cI] & UGrad[cI] & U_[cI])
            / (mag(U_[cI]) * mag(UGrad[cI] & U_[cI]) + mag(U_[cI] & UGrad[cI] & U_[cI]));
    }

    // TENSORFLOW

    // Datastructure for output
    volScalarField betaML_ = betaFieldInversionML_;

    // Structure for tensors in ML
    label numCells = mesh_.cells().size();

    // Some tensorflow pointer requirements
    TF_Status* status_ = TF_NewStatus();
    TF_SessionOptions* options_ = TF_NewSessionOptions();
    TF_Session* sess_ = TF_NewSession(graph_, options_, status_);

    float inputVals[numCells][numInputs];
    const std::vector<std::int64_t> inputDims = {numCells, numInputs};

    forAll(mesh_.C(), cI)
    {
        scalar i1 = (QCriterion_[cI] - meanArray[0]) / (stdArray[0]);
        scalar i2 = (UGradMisalignment_[cI] - meanArray[1]) / (stdArray[1]);
        scalar i3 = (pGradAlongStream_[cI] - meanArray[2]) / (stdArray[2]);
        scalar i4 = (turbulenceIntensity_[cI] - meanArray[3]) / (stdArray[3]);
        scalar i5 = (ReT_[cI] - meanArray[4]) / (stdArray[4]);
        scalar i6 = (convectionTKE_[cI] - meanArray[5]) / (stdArray[5]);
        scalar i7 = (curvature_[cI] - meanArray[6]) / (stdArray[6]);
        scalar i8 = (pressureStress_[cI] - meanArray[7]) / (stdArray[7]);
        scalar i9 = (tauRatio_[cI] - meanArray[8]) / (stdArray[8]);

        assignValueCheckAD(inputVals[cI][0], i1);
        assignValueCheckAD(inputVals[cI][1], i2);
        assignValueCheckAD(inputVals[cI][2], i3);
        assignValueCheckAD(inputVals[cI][3], i4);
        assignValueCheckAD(inputVals[cI][4], i5);
        assignValueCheckAD(inputVals[cI][5], i6);
        assignValueCheckAD(inputVals[cI][6], i7);
        assignValueCheckAD(inputVals[cI][7], i8);
        assignValueCheckAD(inputVals[cI][8], i9);
    }

    // Set up TF C API stuff
    TF_Tensor* outputTensor_ = nullptr;
    TF_Tensor* inputTensor_ = tf_utils::CreateTensor(TF_FLOAT,
                                                     inputDims.data(),
                                                     inputDims.size(),
                                                     &inputVals,
                                                     numCells * numInputs * sizeof(float));

    // Arrays of tensors
    TF_Tensor* inputTensors_[1] = {inputTensor_};
    TF_Tensor* outputTensors_[1] = {outputTensor_};
    // Arrays of operations
    TF_Output inputs[1] = {input_ph_};
    TF_Output outputs[1] = {output_};

    TF_SessionRun(
        sess_,
        nullptr, // Run options.
        inputs,
        inputTensors_,
        1, // Input tensor ops, input tensor values, number of inputs.
        outputs,
        outputTensors_,
        1, // Output tensor ops, output tensor values, number of outputs.
        nullptr,
        0, // Target operations, number of targets.
        nullptr, // Run metadata.
        status_ // Output status.
    );

    const auto data = static_cast<float*>(TF_TensorData(outputTensors_[0]));
    for (label i = 0; i < numCells; i++)
    {
        betaML_[i] = data[numOutputs * i] * stdArray[numInputs] + meanArray[numInputs]; // Funnel changes back into OF - row major order
    }

    tf_utils::DeleteTensor(inputTensor_);
    tf_utils::DeleteTensor(outputTensor_);
    TF_DeleteSessionOptions(options_);
    TF_DeleteStatus(status_);
    tf_utils::DeleteSession(sess_);

    //betaML_ = MyFilter_(betaML_);

    forAll(betaFieldInversionML_.internalField(), cI)
    {
        betaFieldInversionML_[cI] = betaML_[cI];
    }

    // *********** TURBULENCE MODEL FUNCTIONS ***********

    // Note: for compressible flow, the "this->phi()" function divides phi by fvc:interpolate(rho),
    // while for the incompresssible "this->phi()" returns phi only
    // see src/TurbulenceModels/compressible/compressibleTurbulenceModel.C line 62 to 73
    volScalarField::Internal divU(fvc::div(fvc::absolute(phi_ / fvc::interpolate(rho_), U_)));

    tmp<volTensorField> tgradU = fvc::grad(U_);
    volScalarField S2(2 * magSqr(symm(tgradU())));
    volScalarField::Internal GbyNu0((tgradU() && dev(twoSymm(tgradU()))));
    volScalarField::Internal G("kOmegaSSTFIML:G", nut_ * GbyNu0);

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

    volScalarField CDkOmega(
        (scalar(2) * alphaOmega2_) * (fvc::grad(k_) & fvc::grad(omega_)) / omega_);

    volScalarField F1(this->F1(CDkOmega));
    volScalarField F23(this->F23());

    {

        volScalarField::Internal gamma(this->gamma(F1));
        volScalarField::Internal beta(this->beta(F1));

        // Turbulent frequency equation
        tmp<fvScalarMatrix> omegaEqn(
            fvm::ddt(phase_, rho_, omega_)
                + fvm::div(phaseRhoPhi_, omega_, divOmegaScheme)
                - fvm::laplacian(phase_ * rho_ * DomegaEff(F1), omega_)
            == betaFieldInversionML_() * phase_() * rho_() * gamma * GbyNu(GbyNu0, F23(), S2())
                - fvm::SuSp((2.0 / 3.0) * phase_() * rho_() * gamma * divU, omega_)
                - fvm::Sp(phase_() * rho_() * beta * omega_(), omega_)
                - fvm::SuSp(
                    phase_() * rho_() * (F1() - scalar(1)) * CDkOmega() / omega_(),
                    omega_)
                + Qsas(S2(), gamma, beta)
                + omegaSource()

        );

        omegaEqn.ref().relax();
        omegaEqn.ref().boundaryManipulate(omega_.boundaryFieldRef());

        if (solveTurbState_)
        {

            // get the solver performance info such as initial
            // and final residuals
            SolverPerformance<scalar> solverOmega = solve(omegaEqn);
            if (printToScreen)
            {
                Info << "omega Initial residual: " << solverOmega.initialResidual() << endl
                     << "        Final residual: " << solverOmega.finalResidual() << endl;
            }

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
    }

    // Turbulent kinetic energy equation
    tmp<fvScalarMatrix> kEqn(
        fvm::ddt(phase_, rho_, k_)
            + fvm::div(phaseRhoPhi_, k_, divKScheme)
            - fvm::laplacian(phase_ * rho_ * DkEff(F1), k_)
        == phase_() * rho_() * Pk(G)
            - fvm::SuSp((2.0 / 3.0) * phase_() * rho_() * divU, k_)
            - fvm::Sp(phase_() * rho_() * epsilonByk(F1, tgradU()), k_)
            + kSource());

    tgradU.clear();

    kEqn.ref().relax();

    if (solveTurbState_)
    {

        // get the solver performance info such as initial
        // and final residuals
        SolverPerformance<scalar> solverK = solve(kEqn);
        if (printToScreen)
        {
            Info << "k Initial residual: " << solverK.initialResidual() << endl
                 << "    Final residual: " << solverK.finalResidual() << endl;
        }

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
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
