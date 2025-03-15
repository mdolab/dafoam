/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAInputThermalCoupling.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAInputThermalCoupling, 0);
addToRunTimeSelectionTable(DAInput, DAInputThermalCoupling, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAInputThermalCoupling::DAInputThermalCoupling(
    const word inputName,
    const word inputType,
    fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : DAInput(
        inputName,
        inputType,
        mesh,
        daOption,
        daModel,
        daIndex)
{

    daOption_.getAllOptions().subDict("inputInfo").subDict(inputName_).readEntry("patches", patches_);
    // NOTE: always sort the patch because the order of the patch element matters in CHT coupling
    sort(patches_);

    // check discipline
    discipline_ = daOption_.getAllOptions().getWord("discipline");

    // check coupling mode and validate
    distanceMode_ = daOption_.getAllOptions().getWord("wallDistanceMethod");
    if (distanceMode_ != "daCustom" && distanceMode_ != "default")
    {
        FatalErrorIn(" ") << "wallDistanceMethod: "
                          << distanceMode_ << " not supported!"
                          << " Options are: default and daCustom."
                          << abort(FatalError);
    }

    size_ = 0;
    forAll(patches_, idxI)
    {
        word patchName = patches_[idxI];
        label patchI = mesh_.boundaryMesh().findPatchID(patchName);
        forAll(mesh_.boundaryMesh()[patchI], faceI)
        {
            size_++;
        }
    }
    // we have two sets of variable to transfer
    size_ *= 2;
}

void DAInputThermalCoupling::run(const scalarList& input)
{
    /*
    Description:
        Assign the input to OF fields
    */

    volScalarField& T =
        const_cast<volScalarField&>(mesh_.thisDb().lookupObject<volScalarField>("T"));

    // ********* first loop, set the refValue
    label counterI = 0;
    forAll(patches_, idxI)
    {
        // get the patch id label
        word patchName = patches_[idxI];
        label patchI = mesh_.boundaryMesh().findPatchID(patchName);

        mixedFvPatchField<scalar>& mixedPatch =
            refCast<mixedFvPatchField<scalar>>(T.boundaryFieldRef()[patchI]);

        forAll(mixedPatch.refValue(), faceI)
        {
            mixedPatch.refValue()[faceI] = input[counterI];
            mixedPatch.refGrad()[faceI] = 0;
            counterI++;
        }
    }

    // ********* second loop, set the valueFraction:
    // neighKDeltaCoeffs / ( neighKDeltaCoeffs + myKDeltaCoeffs)
    scalar deltaCoeffs = 0;

    if (discipline_ == "aero")
    {

        // for incompressible flow  Q = Cp * alphaEff * dT/dz, so kappa = Cp * alphaEff
        DATurbulenceModel& daTurb = const_cast<DATurbulenceModel&>(daModel_.getDATurbulenceModel());
        word turbModelType = daTurb.getTurbModelType();

        if (turbModelType == "incompressible")
        {
            volScalarField alphaEff = daTurb.alphaEff();

            IOdictionary transportProperties(
                IOobject(
                    "transportProperties",
                    mesh_.time().constant(),
                    mesh_,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE,
                    false));
            scalar Cp = readScalar(transportProperties.lookup("Cp"));

            forAll(patches_, idxI)
            {
                // get the patch id label
                word patchName = patches_[idxI];
                label patchI = mesh_.boundaryMesh().findPatchID(patchName);

                mixedFvPatchField<scalar>& mixedPatch =
                    refCast<mixedFvPatchField<scalar>>(T.boundaryFieldRef()[patchI]);

                forAll(mesh_.boundaryMesh()[patchI], faceI)
                {
                    if (distanceMode_ == "default")
                    {
                        // deltaCoeffs = 1 / d
                        deltaCoeffs = T.boundaryField()[patchI].patch().deltaCoeffs()[faceI];
                    }
                    else if (distanceMode_ == "daCustom")
                    {
                        label nearWallCellIndex = mesh_.boundaryMesh()[patchI].faceCells()[faceI];
                        vector c1 = mesh_.Cf().boundaryField()[patchI][faceI];
                        vector c2 = mesh_.C()[nearWallCellIndex];
                        scalar d = mag(c1 - c2);
                        deltaCoeffs = 1 / d;
                    }
                    scalar alphaEffBf = alphaEff.boundaryField()[patchI][faceI];
                    scalar myKDeltaCoeffs = Cp * alphaEffBf * deltaCoeffs;
                    // NOTE: we continue to use the counterI from the first loop
                    scalar neighKDeltaCoeffs = input[counterI];
                    mixedPatch.valueFraction()[faceI] = neighKDeltaCoeffs / (myKDeltaCoeffs + neighKDeltaCoeffs);
                    counterI++;
                }
            }
        }

        if (turbModelType == "compressible")
        {
            // for compressible flow Q = alphaEff * dHE/dz, so if enthalpy is used, kappa = Cp * alphaEff
            // if the internalEnergy is used, kappa = (Cp - R) * alphaEff
            volScalarField alphaEff = daTurb.alphaEff();
            // compressible flow, H = alphaEff * dHE/dz
            const fluidThermo& thermo = mesh_.thisDb().lookupObject<fluidThermo>("thermophysicalProperties");
            const volScalarField& he = thermo.he();

            const IOdictionary& thermoDict = mesh_.thisDb().lookupObject<IOdictionary>("thermophysicalProperties");
            dictionary mixSubDict = thermoDict.subDict("mixture");
            dictionary specieSubDict = mixSubDict.subDict("specie");
            scalar molWeight = specieSubDict.getScalar("molWeight");
            dictionary thermodynamicsSubDict = mixSubDict.subDict("thermodynamics");
            scalar Cp = thermodynamicsSubDict.getScalar("Cp");

            // 8314.4700665  gas constant in OpenFOAM
            // src/OpenFOAM/global/constants/thermodynamic/thermodynamicConstants.H
            scalar RR = Foam::constant::thermodynamic::RR;

            // R = RR/molWeight
            // Foam::specie::R() function in src/thermophysicalModels/specie/specie/specieI.H
            scalar R = RR / molWeight;

            scalar tmpVal = 0;
            // e = (Cp - R) * T, so Q = alphaEff * (Cp-R) * dT/dz
            if (he.name() == "e")
            {
                tmpVal = Cp - R;
            }
            // h = Cp * T, so Q = alphaEff * Cp * dT/dz
            else
            {
                tmpVal = Cp;
            }

            forAll(patches_, idxI)
            {
                // get the patch id label
                word patchName = patches_[idxI];
                label patchI = mesh_.boundaryMesh().findPatchID(patchName);

                mixedFvPatchField<scalar>& mixedPatch =
                    refCast<mixedFvPatchField<scalar>>(T.boundaryFieldRef()[patchI]);

                forAll(mesh_.boundaryMesh()[patchI], faceI)
                {
                    if (distanceMode_ == "default")
                    {
                        // deltaCoeffs = 1 / d
                        deltaCoeffs = T.boundaryField()[patchI].patch().deltaCoeffs()[faceI];
                    }
                    else if (distanceMode_ == "daCustom")
                    {
                        label nearWallCellIndex = mesh_.boundaryMesh()[patchI].faceCells()[faceI];
                        vector c1 = mesh_.Cf().boundaryField()[patchI][faceI];
                        vector c2 = mesh_.C()[nearWallCellIndex];
                        scalar d = mag(c1 - c2);
                        deltaCoeffs = 1 / d;
                    }
                    scalar alphaEffBf = alphaEff.boundaryField()[patchI][faceI];
                    scalar myKDeltaCoeffs = tmpVal * alphaEffBf * deltaCoeffs;
                    // NOTE: we continue to use the counterI from the first loop
                    scalar neighKDeltaCoeffs = input[counterI];
                    mixedPatch.valueFraction()[faceI] = neighKDeltaCoeffs / (myKDeltaCoeffs + neighKDeltaCoeffs);
                    counterI++;
                }
            }
        }
    }
    else if (discipline_ == "thermal")
    {
        // for solid solvers Q = k * dT/dz, so kappa = k
        IOdictionary solidProperties(
            IOobject(
                "solidProperties",
                mesh_.time().constant(),
                mesh_,
                IOobject::MUST_READ,
                IOobject::NO_WRITE,
                false));
        scalar k = readScalar(solidProperties.lookup("k"));

        forAll(patches_, idxI)
        {
            // get the patch id label
            word patchName = patches_[idxI];
            label patchI = mesh_.boundaryMesh().findPatchID(patchName);

            forAll(mesh_.boundaryMesh()[patchI], faceI)
            {
                if (distanceMode_ == "default")
                {
                    // deltaCoeffs = 1 / d
                    deltaCoeffs = T.boundaryField()[patchI].patch().deltaCoeffs()[faceI];
                }
                else if (distanceMode_ == "daCustom")
                {
                    label nearWallCellIndex = mesh_.boundaryMesh()[patchI].faceCells()[faceI];
                    vector c1 = mesh_.Cf().boundaryField()[patchI][faceI];
                    vector c2 = mesh_.C()[nearWallCellIndex];
                    scalar d = mag(c1 - c2);
                    deltaCoeffs = 1 / d;
                }
                mixedFvPatchField<scalar>& mixedPatch = refCast<mixedFvPatchField<scalar>>(T.boundaryFieldRef()[patchI]);
                scalar myKDeltaCoeffs = k * deltaCoeffs;
                // NOTE: we continue to use the counterI from the first loop
                scalar neighKDeltaCoeffs = input[counterI];
                mixedPatch.valueFraction()[faceI] = neighKDeltaCoeffs / (myKDeltaCoeffs + neighKDeltaCoeffs);
                counterI++;
            }
        }
    }
    else
    {
        FatalErrorIn("DAInputThermalCoupling::run") << " discipline not valid! "
                                                    << abort(FatalError);
    }

    T.correctBoundaryConditions();
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
