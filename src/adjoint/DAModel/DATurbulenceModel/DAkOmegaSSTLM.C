/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    This file is modified from OpenFOAM's source code
    src/TurbulenceModels/turbulenceModels/RAS/kOmegaSSTLM/kOmegaSSTLM.C

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

#include "DAkOmegaSSTLM.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAkOmegaSSTLM, 0);
addToRunTimeSelectionTable(DATurbulenceModel, DAkOmegaSSTLM, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAkOmegaSSTLM::DAkOmegaSSTLM(
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
      ca1_(dimensionedScalar::lookupOrAddToDict(
          "ca1",
          this->coeffDict_,
          2)),
      ca2_(dimensionedScalar::lookupOrAddToDict(
          "ca2",
          this->coeffDict_,
          0.06)),
      ce1_(dimensionedScalar::lookupOrAddToDict(
          "ce1",
          this->coeffDict_,
          1)),
      ce2_(dimensionedScalar::lookupOrAddToDict(
          "ce2",
          this->coeffDict_,
          50)),
      cThetat_(dimensionedScalar::lookupOrAddToDict(
          "cThetat",
          this->coeffDict_,
          0.03)),
      sigmaThetat_(dimensionedScalar::lookupOrAddToDict(
          "sigmaThetat",
          this->coeffDict_,
          2)),
      lambdaErr_(this->coeffDict_.lookupOrDefault("lambdaErr", 1e-6)),
      maxLambdaIter_(this->coeffDict_.lookupOrDefault("maxLambdaIter", 10)),
      deltaU_("deltaU", dimVelocity, SMALL),
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
      ReThetat_(const_cast<volScalarField&>(
          mesh_.thisDb().lookupObject<volScalarField>("ReThetat"))),
      ReThetatRes_(
          IOobject(
              "ReThetatRes",
              mesh_.time().timeName(),
              mesh_,
              IOobject::NO_READ,
              IOobject::NO_WRITE),
          mesh_,
          dimensionedScalar("ReThetatRes", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
          zeroGradientFvPatchScalarField::typeName),
      gammaInt_(const_cast<volScalarField&>(
          mesh_.thisDb().lookupObject<volScalarField>("gammaInt"))),
      gammaIntRes_(
          IOobject(
              "gammaIntRes",
              mesh_.time().timeName(),
              mesh_,
              IOobject::NO_READ,
              IOobject::NO_WRITE),
          mesh_,
          dimensionedScalar("gammaIntRes", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
          zeroGradientFvPatchScalarField::typeName),
      gammaIntEff_(const_cast<volScalarField::Internal&>(
          mesh_.thisDb().lookupObject<volScalarField::Internal>("gammaIntEff"))),
      y_(mesh_.thisDb().lookupObject<volScalarField>("yWall")),
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
        ReThetatRes_.dimensions().reset(dimensionSet(0, 0, -1, 0, 0, 0, 0));
        gammaIntRes_.dimensions().reset(dimensionSet(0, 0, -1, 0, 0, 0, 0));
    }
    else if (turbModelType_ == "compressible")
    {
        omegaRes_.dimensions().reset(dimensionSet(1, -3, -2, 0, 0, 0, 0));
        kRes_.dimensions().reset(dimensionSet(1, -1, -3, 0, 0, 0, 0));
        ReThetatRes_.dimensions().reset(dimensionSet(1, -3, -1, 0, 0, 0, 0));
        gammaIntRes_.dimensions().reset(dimensionSet(1, -3, -1, 0, 0, 0, 0));
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
    volScalarField S2(2 * magSqr(symm(tgradU())));
    volScalarField::Internal GbyNu0((tgradU() && dev(twoSymm(tgradU()))));
    GPtr_.reset(new volScalarField::Internal("kOmegaSSTLM:G", nut_ * GbyNu0));
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// SA member functions. these functions are copied from
tmp<volScalarField> DAkOmegaSSTLM::F1SST(
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

tmp<volScalarField> DAkOmegaSSTLM::F2() const
{

    tmp<volScalarField> arg2 = min(
        max(
            (scalar(2) / betaStar_) * sqrt(k_) / (omega_ * y_),
            scalar(500) * (this->nu()) / (sqr(y_) * omega_)),
        scalar(100));

    return tanh(sqr(arg2));
}

tmp<volScalarField> DAkOmegaSSTLM::F3() const
{

    tmp<volScalarField> arg3 = min(
        150 * (this->nu()) / (omega_ * sqr(y_)),
        scalar(10));

    return 1 - tanh(pow4(arg3));
}

tmp<volScalarField> DAkOmegaSSTLM::F23() const
{
    tmp<volScalarField> f23(F2());

    if (F3_)
    {
        f23.ref() *= F3();
    }

    return f23;
}

tmp<volScalarField::Internal> DAkOmegaSSTLM::GbyNu(
    const volScalarField::Internal& GbyNu0,
    const volScalarField::Internal& F2,
    const volScalarField::Internal& S2) const
{
    return min(
        GbyNu0,
        (c1_ / a1_) * betaStar_ * omega_()
            * max(a1_ * omega_(), b1_ * F2 * sqrt(S2)));
}

tmp<volScalarField::Internal> DAkOmegaSSTLM::PkSST(
    const volScalarField::Internal& G) const
{
    return min(G, (c1_ * betaStar_) * k_() * omega_());
}

tmp<volScalarField::Internal> DAkOmegaSSTLM::epsilonBykSST(
    const volScalarField& F1,
    const volTensorField& gradU) const
{
    return betaStar_ * omega_();
}

tmp<fvScalarMatrix> DAkOmegaSSTLM::kSource() const
{
    return tmp<fvScalarMatrix>(
        new fvScalarMatrix(
            k_,
            dimVolume * this->rhoDimensions() * k_.dimensions() / dimTime));
}

tmp<fvScalarMatrix> DAkOmegaSSTLM::omegaSource() const
{
    return tmp<fvScalarMatrix>(
        new fvScalarMatrix(
            omega_,
            dimVolume * this->rhoDimensions() * omega_.dimensions() / dimTime));
}

tmp<fvScalarMatrix> DAkOmegaSSTLM::Qsas(
    const volScalarField::Internal& S2,
    const volScalarField::Internal& gamma,
    const volScalarField::Internal& beta) const
{
    return tmp<fvScalarMatrix>(
        new fvScalarMatrix(
            omega_,
            dimVolume * this->rhoDimensions() * omega_.dimensions() / dimTime));
}

// SSTLM functions
tmp<volScalarField> DAkOmegaSSTLM::F1(
    const volScalarField& CDkOmega) const
{
    const volScalarField Ry(y_ * sqrt(k_) / this->nu());
    const volScalarField F3(exp(-pow(Ry / 120.0, 8)));

    return max(this->F1SST(CDkOmega), F3);
}

tmp<volScalarField::Internal> DAkOmegaSSTLM::Pk(
    const volScalarField::Internal& G) const
{
    return gammaIntEff_ * this->PkSST(G);
}

tmp<volScalarField::Internal> DAkOmegaSSTLM::epsilonByk(
    const volScalarField& F1,
    const volTensorField& gradU) const
{
    return min(max(gammaIntEff_, scalar(0.1)), scalar(1))
        * this->epsilonBykSST(F1, gradU);
}

tmp<volScalarField::Internal> DAkOmegaSSTLM::Fthetat(
    const volScalarField::Internal& Us,
    const volScalarField::Internal& Omega,
    const volScalarField::Internal& nu) const
{
    const volScalarField::Internal& omega = omega_();
    const volScalarField::Internal& y = y_();

    const volScalarField::Internal delta(375 * Omega * nu * ReThetat_() * y / sqr(Us));
    const volScalarField::Internal ReOmega(sqr(y) * omega / nu);
    const volScalarField::Internal Fwake(exp(-sqr(ReOmega / 1e5)));

    return tmp<volScalarField::Internal>(
        new volScalarField::Internal(
            IOobject::groupName("Fthetat", U_.group()),
            min(
                max(
                    Fwake * exp(-pow4((y / delta))),
                    (1 - sqr((gammaInt_() - 1.0 / ce2_) / (1 - 1.0 / ce2_)))),
                scalar(1))));
}

tmp<volScalarField::Internal> DAkOmegaSSTLM::ReThetac() const
{

    tmp<volScalarField::Internal> tReThetac(
        new volScalarField::Internal(
            IOobject(
                IOobject::groupName("ReThetac", U_.group()),
                mesh_.time().timeName(),
                mesh_),
            mesh_,
            dimless));
    volScalarField::Internal& ReThetac = tReThetac.ref();

    forAll(ReThetac, celli)
    {
        const scalar ReThetat = ReThetat_[celli];

        ReThetac[celli] =
            ReThetat <= 1870
            ? scalar(ReThetat
                     - 396.035e-2
                     + 120.656e-4 * ReThetat
                     - 868.230e-6 * sqr(ReThetat)
                     + 696.506e-9 * pow3(ReThetat)
                     - 174.105e-12 * pow4(ReThetat))
            : scalar(ReThetat - 593.11 - 0.482 * (ReThetat - 1870));
    }

    return tReThetac;
}

tmp<volScalarField::Internal> DAkOmegaSSTLM::Flength(
    const volScalarField::Internal& nu) const
{

    tmp<volScalarField::Internal> tFlength(
        new volScalarField::Internal(
            IOobject(
                IOobject::groupName("Flength", U_.group()),
                mesh_.time().timeName(),
                mesh_),
            mesh_,
            dimless));
    volScalarField::Internal& Flength = tFlength.ref();

    const volScalarField::Internal& omega = omega_();
    const volScalarField::Internal& y = y_();

    forAll(ReThetat_, celli)
    {
        const scalar ReThetat = ReThetat_[celli];

        if (ReThetat < 400)
        {
            Flength[celli] =
                398.189e-1
                - 119.270e-4 * ReThetat
                - 132.567e-6 * sqr(ReThetat);
        }
        else if (ReThetat < 596)
        {
            Flength[celli] =
                263.404
                - 123.939e-2 * ReThetat
                + 194.548e-5 * sqr(ReThetat)
                - 101.695e-8 * pow3(ReThetat);
        }
        else if (ReThetat < 1200)
        {
            Flength[celli] = 0.5 - 3e-4 * (ReThetat - 596);
        }
        else
        {
            Flength[celli] = 0.3188;
        }

        const scalar Fsublayer =
            exp(-sqr(sqr(y[celli]) * omega[celli] / (200 * nu[celli])));

        Flength[celli] = Flength[celli] * (1 - Fsublayer) + 40 * Fsublayer;
    }

    return tFlength;
}

tmp<volScalarField::Internal> DAkOmegaSSTLM::Fonset(
    const volScalarField::Internal& Rev,
    const volScalarField::Internal& ReThetac,
    const volScalarField::Internal& RT) const
{
    const volScalarField::Internal Fonset1(Rev / (2.193 * ReThetac));

    const volScalarField::Internal Fonset2(
        min(max(Fonset1, pow4(Fonset1)), scalar(2)));

    const volScalarField::Internal Fonset3(max(1 - pow3(RT / 2.5), scalar(0)));

    return tmp<volScalarField::Internal>(
        new volScalarField::Internal(
            IOobject::groupName("Fonset", U_.group()),
            max(Fonset2 - Fonset3, scalar(0))));
}

tmp<volScalarField::Internal> DAkOmegaSSTLM::ReThetat0(
    const volScalarField::Internal& Us,
    const volScalarField::Internal& dUsds,
    const volScalarField::Internal& nu) const
{

    tmp<volScalarField::Internal> tReThetat0(
        new volScalarField::Internal(
            IOobject(
                IOobject::groupName("ReThetat0", U_.group()),
                mesh_.time().timeName(),
                mesh_),
            mesh_,
            dimless));
    volScalarField::Internal& ReThetat0 = tReThetat0.ref();

    const volScalarField& k = k_;

    label maxIter = 0;

    forAll(ReThetat0, celli)
    {
        const scalar Tu(
            max(100 * sqrt((2.0 / 3.0) * k[celli]) / Us[celli], scalar(0.027)));

        // Initialize lambda to zero.
        // If lambda were cached between time-steps convergence would be faster
        // starting from the previous time-step value.
        scalar lambda = 0;

        scalar lambdaErr;
        scalar thetat;
        label iter = 0;

        do
        {
            // Previous iteration lambda for convergence test
            const scalar lambda0 = lambda;

            if (Tu <= 1.3)
            {
                const scalar Flambda =
                    dUsds[celli] <= 0
                    ? scalar(1
                             - (-12.986 * lambda
                                - 123.66 * sqr(lambda)
                                - 405.689 * pow3(lambda))
                                 * exp(-pow(Tu / 1.5, 1.5)))
                    : scalar(1
                             + 0.275 * (1 - exp(-35 * lambda))
                                 * exp(-Tu / 0.5));

                thetat =
                    (1173.51 - 589.428 * Tu + 0.2196 / sqr(Tu))
                    * Flambda * nu[celli]
                    / Us[celli];
            }
            else
            {
                const scalar Flambda =
                    dUsds[celli] <= 0
                    ? scalar(1
                             - (-12.986 * lambda
                                - 123.66 * sqr(lambda)
                                - 405.689 * pow3(lambda))
                                 * exp(-pow(Tu / 1.5, 1.5)))
                    : scalar(1
                             + 0.275 * (1 - exp(-35 * lambda))
                                 * exp(-2 * Tu));

                thetat =
                    331.50 * pow((Tu - 0.5658), -0.671)
                    * Flambda * nu[celli] / Us[celli];
            }

            lambda = sqr(thetat) / nu[celli] * dUsds[celli];
            lambda = max(min(lambda, 0.1), -0.1);

            lambdaErr = mag(lambda - lambda0);

            maxIter = max(maxIter, ++iter);

        } while (lambdaErr > lambdaErr_);

        ReThetat0[celli] = max(thetat * Us[celli] / nu[celli], scalar(20));
    }

    if (maxIter > maxLambdaIter_)
    {
        WarningInFunction
            << "Number of lambda iterations exceeds maxLambdaIter("
            << maxLambdaIter_ << ')' << endl;
    }

    return tReThetat0;
}

// Augmented functions
void DAkOmegaSSTLM::correctModelStates(wordList& modelStates) const
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
            modelStates.append("ReThetat");
            modelStates.append("gammaInt");
        }
    }
}

void DAkOmegaSSTLM::correctNut()
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

void DAkOmegaSSTLM::correctBoundaryConditions()
{
    /*
    Description:
        Update turbulence variable boundary values
    */

    // correct the BCs for the perturbed fields
    // kqWallFunction is a zero-gradient BC
    k_.correctBoundaryConditions();

    ReThetat_.correctBoundaryConditions();
    gammaInt_.correctBoundaryConditions();
}

void DAkOmegaSSTLM::correctOmegaBoundaryConditions()
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

void DAkOmegaSSTLM::saveOmegaNearWall()
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

void DAkOmegaSSTLM::setOmegaNearWall()
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

void DAkOmegaSSTLM::updateIntermediateVariables()
{
    /*
    Description:
        Update nut based on nuTilda. Note: we need to update nut and its BC since we 
        may have perturbed other turbulence vars that affect the nut values
    */

    this->correctNut();

    // for SSTLM also update gammaIntEff_
    // NOTE: this is not implemented yet!!!
}

void DAkOmegaSSTLM::correctStateResidualModelCon(List<List<word>>& stateCon) const
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
                stateCon[idxI].append("ReThetat");
                stateCon[idxI].append("gammaInt");
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

void DAkOmegaSSTLM::addModelResidualCon(HashTable<List<List<word>>>& allCon) const
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
                {"U", "omega", "k", "ReThetat", "gammaInt", "phi"}, // lv0
                {"U", "omega", "k", "ReThetat", "gammaInt"}, // lv1
                {"U", "omega", "k", "ReThetat", "gammaInt"} // lv2
            });
        allCon.set(
            "kRes",
            {
                {"U", "omega", "k", "ReThetat", "gammaInt", "phi"}, // lv0
                {"U", "omega", "k", "ReThetat", "gammaInt"}, // lv1
                {"U", "omega", "k", "ReThetat", "gammaInt"} // lv2
            });

        allCon.set(
            "ReThetatRes",
            {
                {"U", "omega", "k", "ReThetat", "gammaInt", "phi"}, // lv0
                {"U", "omega", "k", "ReThetat", "gammaInt"}, // lv1
                {"U", "omega", "k", "ReThetat", "gammaInt"} // lv2
            });

        allCon.set(
            "gammaIntRes",
            {
                {"U", "omega", "k", "ReThetat", "gammaInt", "phi"}, // lv0
                {"U", "omega", "k", "ReThetat", "gammaInt"}, // lv1
                {"U", "omega", "k", "ReThetat", "gammaInt"} // lv2
            });
    }
    else if (turbModelType_ == "compressible")
    {
        allCon.set(
            "omegaRes",
            {
                {"U", "T", pName, "omega", "k", "ReThetat", "gammaInt", "phi"}, // lv0
                {"U", "T", pName, "omega", "k", "ReThetat", "gammaInt"}, // lv1
                {"U", "T", pName, "omega", "k", "ReThetat", "gammaInt"} // lv2
            });
        allCon.set(
            "kRes",
            {
                {"U", "T", pName, "omega", "k", "ReThetat", "gammaInt", "phi"}, // lv0
                {"U", "T", pName, "omega", "k", "ReThetat", "gammaInt"}, // lv1
                {"U", "T", pName, "omega", "k", "ReThetat", "gammaInt"} // lv2
            });

        allCon.set(
            "ReThetatRes",
            {
                {"U", "T", pName, "omega", "k", "ReThetat", "gammaInt", "phi"}, // lv0
                {"U", "T", pName, "omega", "k", "ReThetat", "gammaInt"}, // lv1
                {"U", "T", pName, "omega", "k", "ReThetat", "gammaInt"} // lv2
            });

        allCon.set(
            "gammaIntRes",
            {
                {"U", "T", pName, "omega", "k", "ReThetat", "gammaInt", "phi"}, // lv0
                {"U", "T", pName, "omega", "k", "ReThetat", "gammaInt"}, // lv1
                {"U", "T", pName, "omega", "k", "ReThetat", "gammaInt"} // lv2
            });
    }
}

void DAkOmegaSSTLM::correct(label printToScreen)
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

void DAkOmegaSSTLM::calcResiduals(const dictionary& options)
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

    label printToScreen = options.lookupOrDefault("printToScreen", 0);

    word divKScheme = "div(phi,k)";
    word divOmegaScheme = "div(phi,omega)";
    word divReThetatScheme = "div(phi,ReThetat)";
    word divGammaIntScheme = "div(phi,gammaInt)";

    volScalarField rho = this->rho();

    label isPC = 0;

    if (!solveTurbState_)
    {
        isPC = options.getLabel("isPC");

        if (isPC)
        {
            divKScheme = "div(pc)";
            divOmegaScheme = "div(pc)";
            divReThetatScheme = "div(pc)";
            divGammaIntScheme = "div(pc)";
        }
    }
    // ************ SST part ********
    {
        // Note: for compressible flow, the "this->phi()" function divides phi by fvc:interpolate(rho),
        // while for the incompresssible "this->phi()" returns phi only
        // see src/TurbulenceModels/compressible/compressibleTurbulenceModel.C line 62 to 73
        volScalarField::Internal divU(fvc::div(fvc::absolute(phi_ / fvc::interpolate(rho), U_)));

        tmp<volTensorField> tgradU = fvc::grad(U_);
        volScalarField S2(2 * magSqr(symm(tgradU())));
        volScalarField::Internal GbyNu0((tgradU() && dev(twoSymm(tgradU()))));
        volScalarField::Internal& G = const_cast<volScalarField::Internal&>(GPtr_());
        G = nut_() * GbyNu0;

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
                fvm::ddt(phase_, rho, omega_)
                    + fvm::div(phaseRhoPhi_, omega_, divOmegaScheme)
                    - fvm::laplacian(phase_ * rho * DomegaEff(F1), omega_)
                == phase_() * rho() * gamma * GbyNu(GbyNu0, F23(), S2()) * betaFIOmega_()
                    - fvm::SuSp((2.0 / 3.0) * phase_() * rho() * gamma * divU, omega_)
                    - fvm::Sp(phase_() * rho() * beta * omega_(), omega_)
                    - fvm::SuSp(
                        phase_() * rho() * (F1() - scalar(1)) * CDkOmega() / omega_(),
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
        }

        // Turbulent kinetic energy equation
        tmp<fvScalarMatrix> kEqn(
            fvm::ddt(phase_, rho, k_)
                + fvm::div(phaseRhoPhi_, k_, divKScheme)
                - fvm::laplacian(phase_ * rho * DkEff(F1), k_)
            == phase_() * rho() * Pk(G) * betaFIK_()
                - fvm::SuSp((2.0 / 3.0) * phase_() * rho() * divU, k_)
                - fvm::Sp(phase_() * rho() * epsilonByk(F1, tgradU()), k_)
                + kSource());

        tgradU.clear();

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
    }

    // ************ LM part ********
    {
        // we need to bound before computing residuals
        // this will avoid having NaN residuals
        DAUtility::boundVar(allOptions_, ReThetat_, printToScreen);

        // Local references
        const tmp<volScalarField> tnu = this->nu();
        const volScalarField::Internal& nu = tnu()();
        const volScalarField::Internal& y = y_();

        // Fields derived from the velocity gradient
        tmp<volTensorField> tgradULM = fvc::grad(U_);
        const volScalarField::Internal Omega(sqrt(2 * magSqr(skew(tgradULM()()))));
        const volScalarField::Internal S(sqrt(2 * magSqr(symm(tgradULM()()))));
        const volScalarField::Internal Us(max(mag(U_()), deltaU_));
        const volScalarField::Internal dUsds((U_() & (U_() & tgradULM()())) / sqr(Us));
        tgradULM.clear();

        const volScalarField::Internal Fthetat(this->Fthetat(Us, Omega, nu));

        {
            const volScalarField::Internal t(500 * nu / sqr(Us));
            const volScalarField::Internal Pthetat(
                phase_() * rho() * (cThetat_ / t) * (1 - Fthetat));

            // Transition onset momentum-thickness Reynolds number equation
            tmp<fvScalarMatrix> ReThetatEqn(
                fvm::ddt(phase_, rho, ReThetat_)
                    + fvm::div(phaseRhoPhi_, ReThetat_, divReThetatScheme)
                    - fvm::laplacian(phase_ * rho * DReThetatEff(), ReThetat_)
                == Pthetat * ReThetat0(Us, dUsds, nu) - fvm::Sp(Pthetat, ReThetat_));

            ReThetatEqn.ref().relax();

            if (solveTurbState_)
            {

                // get the solver performance info such as initial
                // and final residuals
                SolverPerformance<scalar> solverReThetat = solve(ReThetatEqn);
                DAUtility::primalResidualControl(solverReThetat, printToScreen, "ReThetat", daGlobalVar_.primalMaxRes);

                DAUtility::boundVar(allOptions_, ReThetat_, printToScreen);
            }
            else
            {
                // calculate residuals
                ReThetatRes_ = ReThetatEqn() & ReThetat_;
                // need to normalize residuals
                normalizeResiduals(ReThetatRes);
            }
        }

        // we need to bound before computing residuals
        // this will avoid having NaN residuals
        DAUtility::boundVar(allOptions_, gammaInt_, printToScreen);

        const volScalarField::Internal ReThetac(this->ReThetac());
        const volScalarField::Internal Rev(sqr(y) * S / nu);
        const volScalarField::Internal RT(k_() / (nu * omega_()));

        {
            const volScalarField::Internal Pgamma(
                phase_() * rho()
                * ca1_ * Flength(nu) * S * sqrt(gammaInt_() * Fonset(Rev, ReThetac, RT)));

            const volScalarField::Internal Fturb(exp(-pow4(0.25 * RT)));

            const volScalarField::Internal Egamma(
                phase_() * rho() * ca2_ * Omega * Fturb * gammaInt_());

            // Intermittency equation
            tmp<fvScalarMatrix> gammaIntEqn(
                fvm::ddt(phase_, rho, gammaInt_)
                    + fvm::div(phaseRhoPhi_, gammaInt_, divGammaIntScheme)
                    - fvm::laplacian(phase_ * rho * DgammaIntEff(), gammaInt_)
                == Pgamma - fvm::Sp(ce1_ * Pgamma, gammaInt_)
                    + Egamma - fvm::Sp(ce2_ * Egamma, gammaInt_));

            gammaIntEqn.ref().relax();

            if (solveTurbState_)
            {

                // get the solver performance info such as initial
                // and final residuals
                SolverPerformance<scalar> solverGammaInt = solve(gammaIntEqn);
                DAUtility::primalResidualControl(solverGammaInt, printToScreen, "gammaInt", daGlobalVar_.primalMaxRes);

                DAUtility::boundVar(allOptions_, gammaInt_, printToScreen);

                const volScalarField::Internal Freattach(exp(-pow4(RT / 20.0)));
                const volScalarField::Internal gammaSep(
                    min(2 * max(Rev / (3.235 * ReThetac) - 1, scalar(0)) * Freattach, scalar(2))
                    * Fthetat);

                gammaIntEff_ = max(gammaInt_(), gammaSep);
            }
            else
            {
                // calculate residuals
                gammaIntRes_ = gammaIntEqn() & gammaInt_;
                // need to normalize residuals
                normalizeResiduals(gammaIntRes);
            }
        }
    }

    return;
}

void DAkOmegaSSTLM::getFvMatrixFields(
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

    // NOTE: there is a bug in this func. The calcPCMatWithFvMatrix returns an error
    // saying try to insert a value that is out of range...
    Info << "Warning!!!!!! this child class is not implemented!" << endl;

    /*
    if (varName != "k" && varName != "omega" && varName != "ReThetat" && varName != "gammaInt")
    {
        FatalErrorIn(
            "varName not valid. It has to be k, omega, ReThetat, or gamma")
            << exit(FatalError);
    }

    volScalarField rho = this->rho();

    // Note: for compressible flow, the "this->phi()" function divides phi by fvc:interpolate(rho),
    // while for the incompresssible "this->phi()" returns phi only
    // see src/TurbulenceModels/compressible/compressibleTurbulenceModel.C line 62 to 73
    volScalarField::Internal divU(fvc::div(fvc::absolute(phi_ / fvc::interpolate(rho), U_)));

    // NOTE instead of calling omega_.boundaryFieldRef().updateCoeffs();
    // here we call our self-defined boundary conditions
    this->correctOmegaBoundaryConditions();

    volScalarField CDkOmega(
        (scalar(2) * alphaOmega2_) * (fvc::grad(k_) & fvc::grad(omega_)) / omega_);

    volScalarField F1(this->F1(CDkOmega));
    volScalarField F23(this->F23());

    if (varName == "omega" || varName == "k")
    {
        // Note: for compressible flow, the "this->phi()" function divides phi by fvc:interpolate(rho),
        // while for the incompresssible "this->phi()" returns phi only
        // see src/TurbulenceModels/compressible/compressibleTurbulenceModel.C line 62 to 73
        volScalarField::Internal divU(fvc::div(fvc::absolute(phi_ / fvc::interpolate(rho), U_)));

        tmp<volTensorField> tgradU = fvc::grad(U_);
        volScalarField S2(2 * magSqr(symm(tgradU())));
        volScalarField::Internal GbyNu0((tgradU() && dev(twoSymm(tgradU()))));
        volScalarField::Internal& G = const_cast<volScalarField::Internal&>(GPtr_());
        G = nut_() * GbyNu0;

        // NOTE instead of calling omega_.boundaryFieldRef().updateCoeffs();
        // here we call our self-defined boundary conditions
        this->correctOmegaBoundaryConditions();

        volScalarField CDkOmega(
            (scalar(2) * alphaOmega2_) * (fvc::grad(k_) & fvc::grad(omega_)) / omega_);

        volScalarField F1(this->F1(CDkOmega));
        volScalarField F23(this->F23());

        if (varName == "omega")
        {

            volScalarField::Internal gamma(this->gamma(F1));
            volScalarField::Internal beta(this->beta(F1));

            // Turbulent frequency equation
            fvScalarMatrix omegaEqn(
                fvm::ddt(phase_, rho, omega_)
                    + fvm::div(phaseRhoPhi_, omega_, "div(pc)")
                    - fvm::laplacian(phase_ * rho * DomegaEff(F1), omega_)
                == phase_() * rho() * gamma * GbyNu(GbyNu0, F23(), S2()) * betaFIOmega_()
                    - fvm::SuSp((2.0 / 3.0) * phase_() * rho() * gamma * divU, omega_)
                    - fvm::Sp(phase_() * rho() * beta * omega_(), omega_)
                    - fvm::SuSp(
                        phase_() * rho() * (F1() - scalar(1)) * CDkOmega() / omega_(),
                        omega_)
                    + Qsas(S2(), gamma, beta)
                    + omegaSource()

            );

            omegaEqn.relax();

            // reset the corrected omega near wall cell to its perturbed value
            this->setOmegaNearWall();

            diag = omegaEqn.D();
            upper = omegaEqn.upper();
            lower = omegaEqn.lower();
        }

        if (varName == "k")
        {
            // Turbulent kinetic energy equation
            fvScalarMatrix kEqn(
                fvm::ddt(phase_, rho, k_)
                    + fvm::div(phaseRhoPhi_, k_, "div(pc)")
                    - fvm::laplacian(phase_ * rho * DkEff(F1), k_)
                == phase_() * rho() * Pk(G)
                    - fvm::SuSp((2.0 / 3.0) * phase_() * rho() * divU, k_)
                    - fvm::Sp(phase_() * rho() * epsilonByk(F1, tgradU()), k_)
                    + kSource());

            tgradU.clear();

            kEqn.relax();

            diag = kEqn.D();
            upper = kEqn.upper();
            lower = kEqn.lower();
        }
    }
    else if (varName == "gammaInt" || varName == "ReThetat")
    {
        // we need to bound before computing residuals
        // this will avoid having NaN residuals
        DAUtility::boundVar(allOptions_, ReThetat_, 0);

        // Local references
        const tmp<volScalarField> tnu = this->nu();
        const volScalarField::Internal& nu = tnu()();
        const volScalarField::Internal& y = y_();

        // Fields derived from the velocity gradient
        tmp<volTensorField> tgradULM = fvc::grad(U_);
        const volScalarField::Internal Omega(sqrt(2 * magSqr(skew(tgradULM()()))));
        const volScalarField::Internal S(sqrt(2 * magSqr(symm(tgradULM()()))));
        const volScalarField::Internal Us(max(mag(U_()), deltaU_));
        const volScalarField::Internal dUsds((U_() & (U_() & tgradULM()())) / sqr(Us));
        tgradULM.clear();

        const volScalarField::Internal Fthetat(this->Fthetat(Us, Omega, nu));

        if (varName == "ReThetat")
        {
            const volScalarField::Internal t(500 * nu / sqr(Us));
            const volScalarField::Internal Pthetat(
                phase_() * rho() * (cThetat_ / t) * (1 - Fthetat));

            // Transition onset momentum-thickness Reynolds number equation
            fvScalarMatrix ReThetatEqn(
                fvm::ddt(phase_, rho, ReThetat_)
                    + fvm::div(phaseRhoPhi_, ReThetat_, "div(pc)")
                    - fvm::laplacian(phase_ * rho * DReThetatEff(), ReThetat_)
                == Pthetat * ReThetat0(Us, dUsds, nu) - fvm::Sp(Pthetat, ReThetat_));

            ReThetatEqn.relax();

            diag = ReThetatEqn.D();
            upper = ReThetatEqn.upper();
            lower = ReThetatEqn.lower();
        }

        if (varName == "gammaInt")
        {
            // we need to bound before computing residuals
            // this will avoid having NaN residuals
            DAUtility::boundVar(allOptions_, gammaInt_, 0);

            const volScalarField::Internal ReThetac(this->ReThetac());
            const volScalarField::Internal Rev(sqr(y) * S / nu);
            const volScalarField::Internal RT(k_() / (nu * omega_()));

            const volScalarField::Internal Pgamma(
                phase_() * rho()
                * ca1_ * Flength(nu) * S * sqrt(gammaInt_() * Fonset(Rev, ReThetac, RT)));

            const volScalarField::Internal Fturb(exp(-pow4(0.25 * RT)));

            const volScalarField::Internal Egamma(
                phase_() * rho() * ca2_ * Omega * Fturb * gammaInt_());

            // Intermittency equation
            fvScalarMatrix gammaIntEqn(
                fvm::ddt(phase_, rho, gammaInt_)
                    + fvm::div(phaseRhoPhi_, gammaInt_, "div(pc)")
                    - fvm::laplacian(phase_ * rho * DgammaIntEff(), gammaInt_)
                == Pgamma - fvm::Sp(ce1_ * Pgamma, gammaInt_)
                    + Egamma - fvm::Sp(ce2_ * Egamma, gammaInt_));

            gammaIntEqn.relax();

            diag = gammaIntEqn.D();
            upper = gammaIntEqn.upper();
            lower = gammaIntEqn.lower();
        }
    }
*/
}

void DAkOmegaSSTLM::getTurbProdOverDestruct(volScalarField& PoD) const
{
    /*
    Description:
        Return the value of the production over destruction term from the turbulence model 
    */
    tmp<volTensorField> tgradU = fvc::grad(U_);
    volScalarField S2(2 * magSqr(symm(tgradU())));
    volScalarField::Internal GbyNu0((tgradU() && dev(twoSymm(tgradU()))));
    volScalarField::Internal& G = const_cast<volScalarField::Internal&>(GPtr_());
    G = nut_() * GbyNu0;

    volScalarField rho = this->rho();

    volScalarField CDkOmega(
        (scalar(2) * alphaOmega2_) * (fvc::grad(k_) & fvc::grad(omega_)) / omega_);

    volScalarField F1(this->F1(CDkOmega));

    volScalarField::Internal P = phase_() * rho() * Pk(G);
    volScalarField::Internal D = phase_() * rho() * epsilonByk(F1, tgradU()) * k_();

    forAll(P, cellI)
    {
        PoD[cellI] = P[cellI] / (D[cellI] + P[cellI] + 1e-16);
    }
}

void DAkOmegaSSTLM::getTurbConvOverProd(volScalarField& CoP) const
{
    /*
    Description:
        Return the value of the convective over production term from the turbulence model 
    */

    tmp<volTensorField> tgradU = fvc::grad(U_);
    volScalarField S2(2 * magSqr(symm(tgradU())));
    volScalarField::Internal GbyNu0((tgradU() && dev(twoSymm(tgradU()))));
    volScalarField::Internal& G = const_cast<volScalarField::Internal&>(GPtr_());
    G = nut_() * GbyNu0;

    volScalarField rho = this->rho();

    volScalarField CDkOmega(
        (scalar(2) * alphaOmega2_) * (fvc::grad(k_) & fvc::grad(omega_)) / omega_);

    volScalarField F1(this->F1(CDkOmega));

    volScalarField::Internal P = phase_() * rho() * Pk(G);
    volScalarField C = fvc::div(phaseRhoPhi_, k_);

    forAll(P, cellI)
    {
        CoP[cellI] = C[cellI] / (P[cellI] + C[cellI] + 1e-16);
    }
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
