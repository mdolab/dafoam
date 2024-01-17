/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

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

    scalar endTime = runTime.endTime().value();
    scalar deltaT = runTime.deltaT().value();
    label nInstances = round(endTime / deltaT);

    Info << "Extracting field " << refFieldName << endl;

    for (label n = 1; n < nInstances + 1; n++)
    {
        scalar t = n * deltaT;
        runTime.setTime(t, n);

        if (refFieldName == "wallShearStress")
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
