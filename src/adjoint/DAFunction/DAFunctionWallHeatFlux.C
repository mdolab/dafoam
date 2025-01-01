/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAFunctionWallHeatFlux.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAFunctionWallHeatFlux, 0);
addToRunTimeSelectionTable(DAFunction, DAFunctionWallHeatFlux, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFunctionWallHeatFlux::DAFunctionWallHeatFlux(
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex,
    const word functionName,
    const word functionPart,
    const dictionary& functionDict)
    : DAFunction(
        mesh,
        daOption,
        daModel,
        daIndex,
        functionName,
        functionPart,
        functionDict),
      wallHeatFlux_(
          IOobject(
              "wallHeatFlux",
              mesh.time().timeName(),
              mesh,
              IOobject::NO_READ,
              IOobject::AUTO_WRITE),
          mesh,
          dimensionedScalar("wallHeatFlux", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
          "calculated")
{
    // Assign type, this is common for all objectives
    functionDict_.readEntry<word>("type", functionType_);

    functionDict_.readEntry<scalar>("scale", scale_);

    if (mesh_.thisDb().foundObject<DATurbulenceModel>("DATurbulenceModel"))
    {
        DATurbulenceModel& daTurbModel =
            const_cast<DATurbulenceModel&>(daModel_.getDATurbulenceModel());
        if (daTurbModel.getTurbModelType() == "incompressible")
        {
            // initialize the Prandtl number from transportProperties
            IOdictionary transportProperties(
                IOobject(
                    "transportProperties",
                    mesh.time().constant(),
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE,
                    false));
            // for incompressible flow, we need to read Cp from transportProperties
            if (Cp_ < 0)
            {
                Cp_ = readScalar(transportProperties.lookup("Cp"));
            }

            wallHeatFlux_.dimensions().reset(dimensionSet(1, -2, 1, 1, 0, 0, 0));
        }
        else if (daTurbModel.getTurbModelType() == "compressible")
        {
            wallHeatFlux_.dimensions().reset(dimensionSet(1, 0, -3, 0, 0, 0, 0));
        }
    }
    else
    {
        // it is solid model
        IOdictionary solidProperties(
            IOobject(
                "solidProperties",
                mesh.time().constant(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE,
                false));
        // for solid, we need to read k from transportProperties
        if (k_ < 0)
        {
            k_ = readScalar(solidProperties.lookup("k"));
        }

        wallHeatFlux_.dimensions().reset(dimensionSet(1, -2, 1, 1, 0, 0, 0));
    }
}

/// calculate the value of objective function
void DAFunctionWallHeatFlux::calcFunction(scalar& functionValue)
{
    /*
    Description:
        Calculate the heat flux F=k*dT/dz from the first cell. Modified based on
        OpenFOAM/OpenFOAM-v1812/src/functionObjects/field/wallWallHeatFlux/wallWallHeatFlux.C

    Output:
        functionValue: the sum of objective, reduced across all processors and scaled by "scale"
    */

    // always calculate the area of all the heat flux patches
    areaSum_ = 0.0;
    forAll(faceSources_, idxI)
    {
        const label& functionFaceI = faceSources_[idxI];
        label bFaceI = functionFaceI - daIndex_.nLocalInternalFaces;
        const label patchI = daIndex_.bFacePatchI[bFaceI];
        const label faceI = daIndex_.bFaceFaceI[bFaceI];
        areaSum_ += mesh_.magSf().boundaryField()[patchI][faceI];
    }
    reduce(areaSum_, sumOp<scalar>());

    // initialize objFunValue
    functionValue = 0.0;

    volScalarField::Boundary& wallHeatFluxBf = wallHeatFlux_.boundaryFieldRef();

    if (mesh_.thisDb().foundObject<DATurbulenceModel>("DATurbulenceModel"))
    {
        DATurbulenceModel& daTurbModel =
            const_cast<DATurbulenceModel&>(daModel_.getDATurbulenceModel());
        if (daTurbModel.getTurbModelType() == "incompressible")
        {
            // incompressible flow does not have he, so we do H = Cp * alphaEff * dT/dz
            const objectRegistry& db = mesh_.thisDb();
            const volScalarField& T = db.lookupObject<volScalarField>("T");
            volScalarField alphaEff = daTurbModel.alphaEff();
            const volScalarField::Boundary& TBf = T.boundaryField();
            const volScalarField::Boundary& alphaEffBf = alphaEff.boundaryField();
            forAll(wallHeatFluxBf, patchI)
            {
                if (!wallHeatFluxBf[patchI].coupled())
                {
                    wallHeatFluxBf[patchI] = Cp_ * alphaEffBf[patchI] * TBf[patchI].snGrad();
                }
            }
        }
        else if (daTurbModel.getTurbModelType() == "compressible")
        {
            // compressible flow, H = alphaEff * dHE/dz
            fluidThermo& thermo = const_cast<fluidThermo&>(
                mesh_.thisDb().lookupObject<fluidThermo>("thermophysicalProperties"));
            volScalarField& he = thermo.he();
            const volScalarField::Boundary& heBf = he.boundaryField();
            volScalarField alphaEff = daTurbModel.alphaEff();
            const volScalarField::Boundary& alphaEffBf = alphaEff.boundaryField();
            forAll(wallHeatFluxBf, patchI)
            {
                if (!wallHeatFluxBf[patchI].coupled())
                {
                    wallHeatFluxBf[patchI] = alphaEffBf[patchI] * heBf[patchI].snGrad();
                }
            }
        }
    }
    else
    {
        // solid. H = k * dT/dz, where k = DT / rho / Cp
        const objectRegistry& db = mesh_.thisDb();
        const volScalarField& T = db.lookupObject<volScalarField>("T");
        const volScalarField::Boundary& TBf = T.boundaryField();
        forAll(wallHeatFluxBf, patchI)
        {
            if (!wallHeatFluxBf[patchI].coupled())
            {
                wallHeatFluxBf[patchI] = k_ * TBf[patchI].snGrad();
            }
        }
    }
    // calculate area weighted heat flux
    forAll(faceSources_, idxI)
    {
        const label& functionFaceI = faceSources_[idxI];
        label bFaceI = functionFaceI - daIndex_.nLocalInternalFaces;
        const label patchI = daIndex_.bFacePatchI[bFaceI];
        const label faceI = daIndex_.bFaceFaceI[bFaceI];

        scalar area = mesh_.magSf().boundaryField()[patchI][faceI];
        scalar val = scale_ * wallHeatFluxBf[patchI][faceI] * area / areaSum_;
        functionValue += val;
    }

    // need to reduce the sum of force across all processors
    reduce(functionValue, sumOp<scalar>());

    // check if we need to calculate refDiff.
    this->calcRefVar(functionValue);

    return;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
