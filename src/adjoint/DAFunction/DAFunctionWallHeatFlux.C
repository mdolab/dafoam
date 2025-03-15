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
    const word functionName)
    : DAFunction(
        mesh,
        daOption,
        daModel,
        daIndex,
        functionName),
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

    // check and assign values for scheme and formulation
    distanceMode_ = daOption_.getAllOptions().getWord("wallDistanceMethod");
    if (distanceMode_ != "daCustom" && distanceMode_ != "default")
    {
        FatalErrorIn(" ") << "wallDistanceMethod: "
                          << distanceMode_ << " not supported!"
                          << " Options are: default and daCustom."
                          << abort(FatalError);
    }
    calcMode_ = functionDict_.lookupOrDefault<bool>("byUnitArea", true);

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

//---------- Calculate Objective Function Value ----------
scalar DAFunctionWallHeatFlux::calcFunction()
{
    /*
    Description:
        Calculate the heat flux F=k*dT/dz from the first cell. Modified based on
        OpenFOAM/OpenFOAM-v1812/src/functionObjects/field/wallWallHeatFlux/wallWallHeatFlux.C

    Output:
        functionValue: the sum of objective, reduced across all processors and scaled by "scale"
    */

    // only calculate the area of all the heat flux patches if scheme is byUnitArea
    if (calcMode_)
    {
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
    }

    // initialize objFunValue
    scalar functionValue = 0.0;

    volScalarField::Boundary& wallHeatFluxBf = wallHeatFlux_.boundaryFieldRef();

    // calculate HFX for fluid domain
    if (mesh_.thisDb().foundObject<DATurbulenceModel>("DATurbulenceModel"))
    {
        DATurbulenceModel& daTurbModel =
            const_cast<DATurbulenceModel&>(daModel_.getDATurbulenceModel());

        // calculate HFX for incompressible flow (no he for incompressible -> HFX = Cp * alphaEff * dT/dz)
        if (daTurbModel.getTurbModelType() == "incompressible")
        {
            const objectRegistry& db = mesh_.thisDb();
            const volScalarField& T = db.lookupObject<volScalarField>("T");
            volScalarField alphaEff = daTurbModel.alphaEff();
            const volScalarField::Boundary& TBf = T.boundaryField();
            const volScalarField::Boundary& alphaEffBf = alphaEff.boundaryField();

            forAll(wallHeatFluxBf, patchI)
            {
                if (!wallHeatFluxBf[patchI].coupled())
                {
                    // use OpenFOAM's snGrad()
                    if (distanceMode_ == "default")
                    {
                        wallHeatFluxBf[patchI] = Cp_ * alphaEffBf[patchI] * TBf[patchI].snGrad();
                    }
                    // use DAFOAM's custom formulation
                    else if (distanceMode_ == "daCustom")
                    {
                        forAll(wallHeatFluxBf[patchI], faceI)
                        {
                            scalar T2 = TBf[patchI][faceI];
                            label nearWallCellIndex = mesh_.boundaryMesh()[patchI].faceCells()[faceI];
                            scalar T1 = T[nearWallCellIndex];
                            vector c1 = mesh_.Cf().boundaryField()[patchI][faceI];
                            vector c2 = mesh_.C()[nearWallCellIndex];
                            scalar d = mag(c1 - c2);
                            scalar dTdz = (T2 - T1) / d;
                            wallHeatFluxBf[patchI][faceI] = Cp_ * alphaEffBf[patchI][faceI] * dTdz;
                        }
                    }
                }
            }
        }
        // calculate HFX for compressible flow (HFX = alphaEff * dHe/dz)
        else if (daTurbModel.getTurbModelType() == "compressible")
        {
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
                    // use OpenFOAM's snGrad()
                    if (distanceMode_ == "default")
                    {
                        wallHeatFluxBf[patchI] = alphaEffBf[patchI] * heBf[patchI].snGrad();
                    }
                    // use DAFOAM's custom formulation
                    else if (distanceMode_ == "daCustom")
                    {
                        forAll(wallHeatFluxBf[patchI], faceI)
                        {
                            scalar He2 = heBf[patchI][faceI];
                            label nearWallCellIndex = mesh_.boundaryMesh()[patchI].faceCells()[faceI];
                            scalar He1 = he[nearWallCellIndex];
                            vector c1 = mesh_.Cf().boundaryField()[patchI][faceI];
                            vector c2 = mesh_.C()[nearWallCellIndex];
                            scalar d = mag(c1 - c2);
                            scalar dHedz = (He2 - He1) / d;
                            wallHeatFluxBf[patchI][faceI] = alphaEffBf[patchI][faceI] * dHedz;
                        }
                    }
                }
            }
        }
    }

    // calculate HFX for solid domain (HFX = k * dT/dz, where k = DT / rho / Cp)
    else
    {
        const objectRegistry& db = mesh_.thisDb();
        const volScalarField& T = db.lookupObject<volScalarField>("T");
        const volScalarField::Boundary& TBf = T.boundaryField();

        forAll(wallHeatFluxBf, patchI)
        {
            if (!wallHeatFluxBf[patchI].coupled())
            {

                // use OpenFOAM's snGrad()
                if (distanceMode_ == "default")
                {
                    wallHeatFluxBf[patchI] = k_ * TBf[patchI].snGrad();
                }
                // use DAFOAM's custom formulation
                else if (distanceMode_ == "daCustom")
                {
                    forAll(wallHeatFluxBf[patchI], faceI)
                    {
                        scalar T2 = TBf[patchI][faceI];
                        label nearWallCellIndex = mesh_.boundaryMesh()[patchI].faceCells()[faceI];
                        scalar T1 = T[nearWallCellIndex];
                        vector c1 = mesh_.Cf().boundaryField()[patchI][faceI];
                        vector c2 = mesh_.C()[nearWallCellIndex];
                        scalar d = mag(c1 - c2);
                        scalar dTdz = (T2 - T1) / d;
                        wallHeatFluxBf[patchI][faceI] = k_ * dTdz;
                    }
                }
            }
        }
    }

    // calculate area weighted heat flux
    scalar val = 0;
    forAll(faceSources_, idxI)
    {
        const label& functionFaceI = faceSources_[idxI];
        label bFaceI = functionFaceI - daIndex_.nLocalInternalFaces;
        const label patchI = daIndex_.bFacePatchI[bFaceI];
        const label faceI = daIndex_.bFaceFaceI[bFaceI];
        scalar area = mesh_.magSf().boundaryField()[patchI][faceI];

        // represent wallHeatFlux by unit area
        if (calcMode_)
        {
            val = scale_ * wallHeatFluxBf[patchI][faceI] * area / areaSum_;
        }
        // represent wallHeatFlux as total heat transfer through surface
        else if (!calcMode_)
        {
            val = scale_ * wallHeatFluxBf[patchI][faceI] * area;
        }
        // error message incase of invalid entry
        else
        {
            FatalErrorIn(" ") << "byUnitArea: "
                              << calcMode_ << " not supported!"
                              << " Options are: True (default) and False."
                              << abort(FatalError);
        }

        // update obj. func. val
        functionValue += val;
    }

    // need to reduce the sum of force across all processors
    reduce(functionValue, sumOp<scalar>());

    // check if we need to calculate refDiff.
    this->calcRefVar(functionValue);

    return functionValue;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
