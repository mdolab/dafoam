/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DATurbulenceModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defineTypeNameAndDebug(DATurbulenceModel, 0);
defineRunTimeSelectionTable(DATurbulenceModel, dictionary);

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DATurbulenceModel::DATurbulenceModel(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption)
    : regIOobject(
        IOobject(
            "DATurbulenceModel",
            mesh.time().timeName(),
            mesh, // register to mesh
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            true // always register object
            )),
      mesh_(mesh),
      daOption_(daOption),
      allOptions_(daOption.getAllOptions()),
      nut_(const_cast<volScalarField&>(
          mesh.thisDb().lookupObject<volScalarField>("nut"))),
      U_(const_cast<volVectorField&>(
          mesh.thisDb().lookupObject<volVectorField>("U"))),
      phi_(const_cast<surfaceScalarField&>(
          mesh.thisDb().lookupObject<surfaceScalarField>("phi"))),
      phase_(
          IOobject(
              "phase",
              mesh.time().timeName(),
              mesh,
              IOobject::NO_READ,
              IOobject::NO_WRITE,
              false),
          mesh,
          dimensionedScalar("phase", dimensionSet(0, 0, 0, 0, 0, 0, 0), 1.0),
          zeroGradientFvPatchField<scalar>::typeName),
      phaseRhoPhi_(const_cast<surfaceScalarField&>(
          mesh.thisDb().lookupObject<surfaceScalarField>("phi"))),
#ifdef IncompressibleFlow
      daRegDbTransport_(mesh.thisDb().lookupObject<DARegDbSinglePhaseTransportModel>(
          "DARegDbSinglePhaseTransportModel")),
      laminarTransport_(daRegDbTransport_.getObject()),
      daRegDbTurbIncomp_(mesh.thisDb().lookupObject<DARegDbTurbulenceModelIncompressible>(
          "DARegDbTurbulenceModelIncompressible")),
      turbulence_(daRegDbTurbIncomp_.getObject()),
      // for incompressible, we use uniform one field for rho
      rho_(
          IOobject(
              "rho",
              mesh.time().timeName(),
              mesh,
              IOobject::NO_READ,
              IOobject::NO_WRITE,
              false),
          mesh,
          dimensionedScalar("rho", dimensionSet(0, 0, 0, 0, 0, 0, 0), 1.0),
          zeroGradientFvPatchField<scalar>::typeName),
#endif
#ifdef CompressibleFlow
      daRegDbThermo_(mesh.thisDb().lookupObject<DARegDbFluidThermo>("DARegDbFluidThermo")),
      thermo_(daRegDbThermo_.getObject()),
      daRegDbTurbComp_(mesh.thisDb().lookupObject<DARegDbTurbulenceModelCompressible>(
          "DARegDbTurbulenceModelCompressible")),
      turbulence_(daRegDbTurbComp_.getObject()),
      // for compressible flow, we lookup rho in fvMesh
      rho_(const_cast<volScalarField&>(mesh.thisDb().lookupObject<volScalarField>("rho"))),
#endif
      turbDict_(
          IOobject(
              "turbulenceProperties",
              mesh.time().constant(),
              mesh,
              IOobject::MUST_READ,
              IOobject::NO_WRITE,
              false)),
      coeffDict_(turbDict_.subDict("RAS")),
      kMin_(dimensioned<scalar>::lookupOrAddToDict(
          "kMin",
          coeffDict_,
          sqr(dimVelocity),
          SMALL)),
      epsilonMin_(dimensioned<scalar>::lookupOrAddToDict(
          "epsilonMin",
          coeffDict_,
          kMin_.dimensions() / dimTime,
          SMALL)),
      omegaMin_(dimensioned<scalar>::lookupOrAddToDict(
          "omegaMin",
          coeffDict_,
          dimless / dimTime,
          SMALL)),
      nuTildaMin_(dimensioned<scalar>::lookupOrAddToDict(
          "nuTildaMin",
          coeffDict_,
          nut_.dimensions(),
          SMALL))
{

    // Now we need to initialize other variables
#ifdef IncompressibleFlow

    // initialize the Prandtl number from transportProperties
    IOdictionary transportProperties(
        IOobject(
            "transportProperties",
            mesh.time().constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false));
    Pr_ = readScalar(transportProperties.lookup("Pr"));

    if (mesh_.thisDb().foundObject<volScalarField>("alphat"))
    {
        Prt_ = readScalar(transportProperties.lookup("Prt"));
    }

#endif

#ifdef CompressibleFlow

    // initialize the Prandtl number from thermophysicalProperties
    IOdictionary thermophysicalProperties(
        IOobject(
            "thermophysicalProperties",
            mesh.time().constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false));
    Pr_ = readScalar(
        thermophysicalProperties.subDict("mixture").subDict("transport").lookup("Pr"));

    if (mesh_.thisDb().foundObject<volScalarField>("alphat"))
    {
        const IOdictionary& turbDict = mesh_.thisDb().lookupObject<IOdictionary>("turbulenceProperties");
        dictionary rasSubDict = turbDict.subDict("RAS");
        Prt_ = rasSubDict.getScalar("Prt");
    }

#endif
}

// * * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * //

autoPtr<DATurbulenceModel> DATurbulenceModel::New(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption)
{
    if (daOption.getAllOptions().lookupOrDefault<label>("debug", 0))
    {
        Info << "Selecting " << modelType << " for DATurbulenceModel" << endl;
    }

    dictionaryConstructorTable::iterator cstrIter =
        dictionaryConstructorTablePtr_->find(modelType);

    if (cstrIter == dictionaryConstructorTablePtr_->end())
    {
        FatalErrorIn(
            "DATurbulenceModel::New"
            "("
            "    const word,"
            "    const fvMesh&,"
            "    const DAOption&"
            ")")
            << "Unknown DATurbulenceModel type "
            << modelType << nl << nl
            << "Valid DATurbulenceModel types:" << endl
            << dictionaryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    return autoPtr<DATurbulenceModel>(
        cstrIter()(modelType, mesh, daOption));
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

// this is a virtual function for regIOobject
bool DATurbulenceModel::writeData(Ostream& os) const
{
    // do nothing
    return true;
}

void DATurbulenceModel::correctAlphat()
{
    // see src/TurbulenceModels/compressible/EddyDiffusivity/EddyDiffusivity.C
    if (mesh_.thisDb().foundObject<volScalarField>("alphat"))
    {
        dimensionedScalar Prt(
            "Prt1",
            dimless,
            Prt_);

        volScalarField& alphat = const_cast<volScalarField&>(
            mesh_.thisDb().lookupObject<volScalarField>("alphat"));
        alphat = rho_ * nut_ / Prt;
        alphat.correctBoundaryConditions();
    }
}

tmp<volScalarField> DATurbulenceModel::nuEff() const
{
    /*
    Description:
        Return nut+nu
    */

    return tmp<volScalarField>(
        new volScalarField(
            "nuEff",
            this->nu() + nut_));
}

tmp<volScalarField> DATurbulenceModel::alphaEff()
{
    /*
    Description:
        Return alphat+alpha
        For compressible flow, we get it from thermo object
        For incompressible flow we use alpha+alphat
    */

#ifdef IncompressibleFlow
    const volScalarField& alphat = mesh_.thisDb().lookupObject<volScalarField>("alphat");
    return tmp<volScalarField>(
        new volScalarField(
            "alphaEff",
            this->getAlpha() + alphat));
#endif

#ifdef CompressibleFlow
    const volScalarField& alphat = mesh_.thisDb().lookupObject<volScalarField>("alphat");
    return tmp<volScalarField>(
        new volScalarField(
            "alphaEff",
            thermo_.alphaEff(alphat)));
#endif
}

tmp<volScalarField> DATurbulenceModel::nu() const
{
    /*
    Description:
        Return nu
        For compressible flow, we get it from mu/rho
        For incompressible flow we get it from ther laminarTransport object
    */

#ifdef IncompressibleFlow
    return laminarTransport_.nu();
#endif

#ifdef CompressibleFlow
    return thermo_.mu() / rho_;
#endif
}

tmp<volScalarField> DATurbulenceModel::getAlpha() const
{
    /*
    Description:
        Return alpha = nu/Pr
    */
    return this->nu() / Pr_;
}

tmp<Foam::volScalarField> DATurbulenceModel::getMu() const
{
    /*
    Description:
        Return mu
        For compressible flow, we get it from thermo
        Not appliable for other flow conditions
    */

#ifdef CompressibleFlow
    return thermo_.mu();
#else
    FatalErrorIn("flowCondition not valid!") << abort(FatalError);
    return nut_;
#endif
}

tmp<volSymmTensorField> DATurbulenceModel::devRhoReff() const
{
    /*
    Description:
        Return devRhoReff for computing viscous force
    */

    return tmp<volSymmTensorField>(
        new volSymmTensorField(
            IOobject(
                IOobject::groupName("devRhoReff", U_.group()),
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE),
            (-phase_ * rho_ * nuEff()) * dev(twoSymm(fvc::grad(U_)))));
}

tmp<fvVectorMatrix> DATurbulenceModel::divDevRhoReff(
    volVectorField& U)
{
    /*
    Description:
        Return divDevRhoReff for the laplacian terms
    */

#ifdef IncompressibleFlow
    word divScheme = "div((nuEff*dev2(T(grad(U)))))";
#endif
#ifdef CompressibleFlow
    word divScheme = "div(((rho*nuEff)*dev2(T(grad(U)))))";
#endif

    volScalarField& phase = phase_;
    volScalarField& rho = rho_;

    return (
        -fvm::laplacian(phase * rho * nuEff(), U)
        - fvc::div((phase * rho * nuEff()) * dev2(T(fvc::grad(U))), divScheme));
}

tmp<fvVectorMatrix> DATurbulenceModel::divDevReff(
    volVectorField& U)
{
    /*
    Description:
        Call divDevRhoReff
    */

    return divDevRhoReff(U);
}

void DATurbulenceModel::correctWallDist()
{
    /*
    Description:
        Update the near wall distance
    */

    // ********* TODO ********
    // ********* this is not implemented yet ********
    //d_.correct();
    // need to correct turbulence boundary conditions
    // this is because when the near wall distance changes, the nut, omega, epsilon at wall
    // may change if you use wall functions
    this->correctBoundaryConditions();
}

#ifdef CompressibleFlow
const fluidThermo& DATurbulenceModel::getThermo() const
{
    /*
    Description:
        Return the thermo object, only for compressible flow
    */
    return thermo_;
}
#endif

void DATurbulenceModel::printYPlus() const
{
    /*
    Description:
        Calculate the min, max, and mean yPlus for all walls and print the 
        values to screen
        Modified from src/functionObjects/field/yPlus/yPlus.C
    */

    volScalarField yPlus(
        IOobject(
            typeName,
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE),
        mesh_,
        dimensionedScalar(dimless, Zero));
    volScalarField::Boundary& yPlusBf = yPlus.boundaryFieldRef();

    const nearWallDist nwd(mesh_);
    const volScalarField::Boundary& d = nwd.y();

    const fvPatchList& patches = mesh_.boundary();

    const volScalarField::Boundary& nutBf = nut_.boundaryField();
    const volVectorField::Boundary& UBf = U_.boundaryField();

    volScalarField nuEff = this->nuEff();
    volScalarField nu = this->nu();

    const volScalarField::Boundary& nuEffBf = nuEff.boundaryField();
    const volScalarField::Boundary& nuBf = nu.boundaryField();

    forAll(patches, patchI)
    {
        const fvPatch& patch = patches[patchI];

        if (isA<nutWallFunctionFvPatchScalarField>(nutBf[patchI]))
        {
            const nutWallFunctionFvPatchScalarField& nutPf =
                dynamic_cast<const nutWallFunctionFvPatchScalarField&>(
                    nutBf[patchI]);

            yPlusBf[patchI] = nutPf.yPlus();
        }
        else if (isA<wallFvPatch>(patch))
        {
            yPlusBf[patchI] =
                d[patchI]
                * sqrt(
                    nuEffBf[patchI]
                    * mag(UBf[patchI].snGrad()))
                / nuBf[patchI];
        }
    }

    // now compute the global min, max, and mean
    scalarList yPlusAll;
    forAll(patches, patchI)
    {
        const fvPatch& patch = patches[patchI];
        if (isA<wallFvPatch>(patch))
        {
            forAll(yPlusBf[patchI], faceI)
            {
                yPlusAll.append(yPlusBf[patchI][faceI]);
            }
        }
    }
    scalar minYplus = gMin(yPlusAll);
    scalar maxYplus = gMax(yPlusAll);
    scalar avgYplus = gAverage(yPlusAll);

    Info << "yPlus min: " << minYplus
         << " max: " << maxYplus
         << " mean: " << avgYplus << endl;
}

label DATurbulenceModel::isPrintTime(
    const Time& runTime,
    const label printInterval) const
{
    /*
    Description:
        Check if it is print time
    */

    if (runTime.timeIndex() % printInterval == 0 || runTime.timeIndex() == 1)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

void DATurbulenceModel::getTurbProdTerm(scalarList& prodTerm) const
{
    /*
    Description:
        Return the value of the production term from the turbulence model 
    */

    FatalErrorIn("DATurbulenceModel::getSAProdTerm")
        << "Child class not implemented!"
        << abort(FatalError);
}

void DATurbulenceModel::invTranProdNuTildaEqn(
    const volScalarField& mySource,
    volScalarField& pseudoNuTilda)
{
    /*
    Description:
        Inverse transpose product, M_nuTilda^(-T)
    */

    FatalErrorIn("DATurbulenceModel::invTranProdNuTildaEqn")
        << "Child class not implemented!"
        << abort(FatalError);
}

void DATurbulenceModel::calcLduResidualTurb(volScalarField& nuTildaRes)
{
    /*
    Description:
        calculate the turbulence residual using LDU matrix
    */

    FatalErrorIn("DATurbulenceModel::calcLduResidualTurb")
        << "Child class not implemented!"
        << abort(FatalError);
}

void DATurbulenceModel::constructPseudoNuTildaEqn()
{
    /*
    Description:
        construct the pseudo nuTildaEqn
    */

    FatalErrorIn("DATurbulenceModel::constructPseudoNuTildaEqn")
        << "Child class not implemented!"
        << abort(FatalError);
}

void DATurbulenceModel::rhsSolvePseudoNuTildaEqn(const volScalarField& nuTildaSource)
{
    /*
    Description:
        solve the pseudo nuTildaEqn with overwritten rhs
    */

    FatalErrorIn("DATurbulenceModel::rhsSolvePseudoNuTildaEqn")
        << "Child class not implemented!"
        << abort(FatalError);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
