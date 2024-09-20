/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

Description
    Extract the reference data for field inversion

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "OFstream.H"

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char* argv[])
{

    Info << "Get ref data for field inversion...." << endl;

    argList::addOption(
        "refFieldName",
        "U",
        "field variables name to extract");

    argList::addOption(
        "refFieldType",
        "vector",
        "field variables type");

    argList::addOption(
        "time",
        "-1",
        "prescribe a specific time to extract data");

#include "setRootCase.H"
#include "createTime.H"
#include "createMesh.H"

    word refFieldName;
    if (args.optionFound("refFieldName"))
    {
        refFieldName = word(args.optionLookup("refFieldName")());
    }
    else
    {
        Info << "refFieldName not set! Exit." << endl;
        return 1;
    }

    word refFieldType;
    if (args.optionFound("refFieldType"))
    {
        refFieldType = word(args.optionLookup("refFieldType")());
    }
    else
    {
        Info << "refFieldType not set! Exit." << endl;
        return 1;
    }

    scalar time = -1.0;
    if (args.optionFound("time"))
    {
        time = readScalar(args.optionLookup("time")());
        if (time == 9999)
        {
            Info << "Extract latestTime" << endl;
        }
        else
        {
            Info << "Extract time = " << time << endl;
        }
    }
    else
    {
        Info << "time not set! Extract all time instances." << endl;
    }

    scalar endTime = runTime.endTime().value();
    scalar deltaT = runTime.deltaT().value();
    label nInstances = -1;
    if (time == -1.0)
    {
        nInstances = round(endTime / deltaT);
    }
    else
    {
        nInstances = 1;
    }

    Info << "Extracting field " << refFieldName << endl;

    for (label n = 1; n < nInstances + 1; n++)
    {
        scalar t = -1.0;
        if (time == -1.0)
        {
            // read all times
            t = n * deltaT;
        }
        else if (time == 9999)
        {
            // read from the latestTime (it is not necessarily the endTime)
            instantList timeDirs = runTime.findTimes(runTime.path(), runTime.constant());
            t = timeDirs.last().value();
        }
        else
        {
            // read from the specified time
            t = time;
        }
        runTime.setTime(t, n);

        if (refFieldName == "wallHeatFlux")
        {

            volScalarField hfx(
                IOobject(
                    "wallHeatFluxData",
                    mesh.time().timeName(),
                    mesh,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE),
                mesh,
                dimensionedScalar("wallHeatFluxData", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
                "fixedValue");

            volScalarField rho(
                IOobject(
                    "rho",
                    mesh.time().timeName(),
                    mesh,
                    IOobject::READ_IF_PRESENT,
                    IOobject::NO_WRITE),
                mesh,
                dimensionedScalar("rho", dimensionSet(1, -3, 0, 0, 0, 0, 0), 1.0),
                "zeroGradient");

            volScalarField alphat(
                IOobject(
                    "alphat",
                    mesh.time().timeName(),
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE),
                mesh);

            volScalarField T(
                IOobject(
                    "T",
                    mesh.time().timeName(),
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE),
                mesh);

            if (rho[0] == 1.0)
            {
                // incompressible cases

                volScalarField alpha(
                    IOobject(
                        "alpha",
                        mesh.time().timeName(),
                        mesh,
                        IOobject::NO_READ,
                        IOobject::NO_WRITE),
                    mesh,
                    dimensionedScalar("alpha", dimensionSet(0, 2, -1, 0, 0, 0, 0), 0.0),
                    "zeroGradient");

                IOdictionary transportProperties(
                    IOobject(
                        "transportProperties",
                        mesh.time().constant(),
                        mesh,
                        IOobject::MUST_READ,
                        IOobject::NO_WRITE,
                        false));

                scalar nu = transportProperties.getScalar("nu");
                scalar Pr = transportProperties.getScalar("Pr");
                scalar Cp = transportProperties.getScalar("Cp");
                forAll(alpha, cellI)
                {
                    alpha[cellI] = nu / Pr;
                }
                alpha.correctBoundaryConditions();

                volScalarField alphaEff("alphaEff", alphat + alpha);

                forAll(mesh.boundaryMesh(), patchI)
                {
                    if (mesh.boundaryMesh()[patchI].type() == "wall" && mesh.boundaryMesh()[patchI].size() > 0)
                    {
                        scalarField flux = Cp * alphaEff.boundaryField()[patchI] * T.boundaryField()[patchI].snGrad();
                        forAll(mesh.boundaryMesh()[patchI], faceI)
                        {
                            hfx.boundaryFieldRef()[patchI][faceI] = flux[faceI];
                        }
                    }
                }

                hfx.write();
            }
            else
            {
                // compressible

                volScalarField alpha(
                    IOobject(
                        "alpha",
                        mesh.time().timeName(),
                        mesh,
                        IOobject::NO_READ,
                        IOobject::NO_WRITE),
                    mesh,
                    dimensionedScalar("alpha", dimensionSet(1, -1, -1, 0, 0, 0, 0), 0.0),
                    "zeroGradient");

                IOdictionary thermophysicalProperties(
                    IOobject(
                        "thermophysicalProperties",
                        mesh.time().constant(),
                        mesh,
                        IOobject::MUST_READ,
                        IOobject::NO_WRITE,
                        false));

                dictionary subDict = thermophysicalProperties.subDict("mixture").subDict("transport");
                scalar mu = subDict.getScalar("mu");
                scalar Pr = subDict.getScalar("Pr");
                scalar Cp = thermophysicalProperties.subDict("mixture").subDict("thermodynamics").getScalar("Cp");
                scalar molWeight = thermophysicalProperties.subDict("mixture").subDict("specie").getScalar("molWeight");
                forAll(alpha, cellI)
                {
                    alpha[cellI] = mu * rho[cellI] / Pr;
                }
                alpha.correctBoundaryConditions();

                volScalarField alphaEff("alphaEff", alphat + alpha);

                volScalarField he(
                    IOobject(
                        "he",
                        mesh.time().timeName(),
                        mesh,
                        IOobject::NO_READ,
                        IOobject::NO_WRITE),
                    mesh,
                    dimensionedScalar("he", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
                    "fixedValue");

                word energy = thermophysicalProperties.subDict("thermoType").getWord("energy");
                if (energy == "sensibleInternalEnergy")
                {
                    // e
                    scalar RR = Foam::constant::thermodynamic::RR;
                    scalar R = RR / molWeight;
                    forAll(he, cellI)
                    {
                        he[cellI] = (Cp - R) * T[cellI];
                    }
                    forAll(he.boundaryField(), patchI)
                    {
                        forAll(he.boundaryField()[patchI], faceI)
                        {
                            he.boundaryFieldRef()[patchI][faceI] = (Cp - R) * T.boundaryField()[patchI][faceI];
                        }
                    }
                    he.correctBoundaryConditions();
                }
                else if (energy == "sensibleEnthalpy")
                {
                    // h
                    forAll(he, cellI)
                    {
                        he[cellI] = Cp * T[cellI];
                    }
                    forAll(he.boundaryField(), patchI)
                    {
                        forAll(he.boundaryField()[patchI], faceI)
                        {
                            he.boundaryFieldRef()[patchI][faceI] = Cp * T.boundaryField()[patchI][faceI];
                        }
                    }
                    he.correctBoundaryConditions();
                }
                else
                {
                    FatalErrorIn("") << "energy type: " << energy << " not supported in thermophysicalProperties" << abort(FatalError);
                }

                forAll(mesh.boundaryMesh(), patchI)
                {
                    if (mesh.boundaryMesh()[patchI].type() == "wall" && mesh.boundaryMesh()[patchI].size() > 0)
                    {
                        scalarField flux = alphaEff.boundaryField()[patchI] * he.boundaryField()[patchI].snGrad();
                        forAll(mesh.boundaryMesh()[patchI], faceI)
                        {
                            hfx.boundaryFieldRef()[patchI][faceI] = flux[faceI];
                        }
                    }
                }

                hfx.write();
            }
        }
        else if (refFieldName == "wallShearStress")
        {

            // the following codes are adjusted from OpenFOAM/src/functionObjects/field/wallShearStress/wallShearStress.C

            volVectorField shear(
                IOobject(
                    "wallShearStressData",
                    mesh.time().timeName(),
                    mesh,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE),
                mesh,
                dimensionedVector("wallShearStressData", dimensionSet(0, 0, 0, 0, 0, 0, 0), vector::zero),
                "fixedValue");

            volScalarField rho(
                IOobject(
                    "rho",
                    mesh.time().timeName(),
                    mesh,
                    IOobject::READ_IF_PRESENT,
                    IOobject::NO_WRITE),
                mesh,
                dimensionedScalar("rho", dimensionSet(1, -3, 0, 0, 0, 0, 0), 1.0),
                "zeroGradient");

            volScalarField nut(
                IOobject(
                    "nut",
                    mesh.time().timeName(),
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE),
                mesh);

            volVectorField U(
                IOobject(
                    "U",
                    mesh.time().timeName(),
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE),
                mesh);

            volScalarField nu(
                IOobject(
                    "nu",
                    mesh.time().timeName(),
                    mesh,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE),
                mesh,
                dimensionedScalar("nu", dimensionSet(0, 2, -1, 0, 0, 0, 0), 0.0),
                "zeroGradient");

            if (rho[0] == 1.0)
            {
                // incompressible cases
                IOdictionary transportProperties(
                    IOobject(
                        "transportProperties",
                        mesh.time().constant(),
                        mesh,
                        IOobject::MUST_READ,
                        IOobject::NO_WRITE,
                        false));

                scalar nuRead = transportProperties.getScalar("nu");
                forAll(nu, cellI)
                {
                    nu[cellI] = nuRead;
                }
                nu.correctBoundaryConditions();
            }
            else
            {
                // compressible
                IOdictionary thermophysicalProperties(
                    IOobject(
                        "thermophysicalProperties",
                        mesh.time().constant(),
                        mesh,
                        IOobject::MUST_READ,
                        IOobject::NO_WRITE,
                        false));

                dictionary subDict = thermophysicalProperties.subDict("mixture").subDict("transport");
                scalar mu = subDict.getScalar("mu");
                forAll(nu, cellI)
                {
                    nu[cellI] = mu * rho[cellI];
                }
                nu.correctBoundaryConditions();
            }

            volScalarField nuEff("nuEff", nut + nu);

            volSymmTensorField devRhoReff = (-rho * nuEff) * dev(twoSymm(fvc::grad(U)));

            forAll(shear.boundaryField(), patchI)
            {
                vectorField& shearp = shear.boundaryFieldRef()[patchI];
                const vectorField& Sfp = mesh.Sf().boundaryField()[patchI];
                const scalarField& magSfp = mesh.magSf().boundaryField()[patchI];
                const symmTensorField& Reffp = devRhoReff.boundaryField()[patchI];

                shearp = (-Sfp / magSfp) & Reffp;
            }

            shear.write();
        }
        else if (refFieldType == "scalar")
        {
            volScalarField fieldRead(
                IOobject(
                    refFieldName,
                    mesh.time().timeName(),
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE),
                mesh);

            fieldRead.rename(refFieldName + "Data");
            fieldRead.write();
        }
        else if (refFieldType == "vector")
        {
            volVectorField fieldRead(
                IOobject(
                    refFieldName,
                    mesh.time().timeName(),
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE),
                mesh);

            fieldRead.rename(refFieldName + "Data");
            fieldRead.write();
        }
        else
        {
            Info << "refFieldName and type not supported!" << endl;
        }
    }

    Info << "Finished!" << endl;

    return 0;
}

// ************************************************************************* //
