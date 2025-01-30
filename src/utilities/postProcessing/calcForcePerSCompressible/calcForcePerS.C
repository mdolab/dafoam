/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Calculating force per surface area. This will be used to plot the 
        spanwise force distribution

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "argList.H"
#include "autoPtr.H"
#include "Time.H"
#include "timeSelector.H"
#include "TimePaths.H"
#include "ListOps.H"
#include "fvMesh.H"
#include "OFstream.H"
#include "simpleControl.H"
#include "fvOptions.H"
#include "fluidThermo.H"

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char* argv[])
{
    Info << "Computing forces...." << endl;

    argList::addOption(
        "patchNames",
        "'(wing)'",
        "List of patch names to compute");

    argList::addOption(
        "time",
        "1000",
        "Time instance to compute, if not provided runs all times");

#include "setRootCase.H"
#include "createTime.H"

    if (!args.optionFound("time"))
    {
        Info << "Time not set, running all times." << endl;
    }

    // Create the processor databases
    autoPtr<TimePaths> timePaths;
    timePaths = autoPtr<TimePaths>::New(args.rootPath(), args.caseName());

    const instantList timeDirs(timeSelector::select(timePaths->times(), args));

    List<wordRe> patchNames;
    if (args.optionFound("patchNames"))
    {
        patchNames = wordReList(args.optionLookup("patchNames")());
    }
    else
    {
        Info << "patchNames not set! Exit." << endl;
        return 1;
    }

    forAll(timeDirs, iTime)
    {
        runTime.setTime(timeDirs[iTime].value(), 0);

#include "createMesh.H"
#include "createFields.H"

        // Initialize surface field for face-centered forces
        volVectorField forcePerS(
            IOobject(
                "forcePerS",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE),
            mesh,
            dimensionedVector("forcePerS", dimensionSet(1, -1, -2, 0, 0, 0, 0), vector::zero),
            fixedValueFvPatchScalarField::typeName);

        // this code is pulled from:
        // src/functionObjects/forcces/forces.C
        // modified slightly
        vector forces(vector::zero);

        const surfaceVectorField::Boundary& Sfb = mesh.Sf().boundaryField();
        const surfaceScalarField::Boundary& magSfb = mesh.magSf().boundaryField();

        volSymmTensorField devRhoReff(
            IOobject(
                IOobject::groupName("devRhoReff", U.group()),
                mesh.time().timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE),
            (-rho * nuEff) * dev(twoSymm(fvc::grad(U))));

        const volSymmTensorField::Boundary& devRhoReffb = devRhoReff.boundaryField();

        vector totalForce(vector::zero);
        forAll(patchNames, cI)
        {
            // get the patch id label
            label patchI = mesh.boundaryMesh().findPatchID(patchNames[cI]);
            if (patchI < 0)
            {
                Info << "ERROR: Patch name " << patchNames[cI] << " not found in constant/polyMesh/boundary! Exit!" << endl;
                return 1;
            }
            // create a shorter handle for the boundary patch
            const fvPatch& patch = mesh.boundary()[patchI];
            // normal force
            vectorField fN(Sfb[patchI] * p.boundaryField()[patchI]);
            // tangential force
            vectorField fT(Sfb[patchI] & devRhoReffb[patchI]);
            // sum them up
            forAll(patch, faceI)
            {
                // compute forces
                forces.x() = fN[faceI].x() + fT[faceI].x();
                forces.y() = fN[faceI].y() + fT[faceI].y();
                forces.z() = fN[faceI].z() + fT[faceI].z();

                // project force direction
                forcePerS.boundaryFieldRef()[patchI][faceI] = forces / magSfb[patchI][faceI];

                totalForce.x() += forces.x();
                totalForce.y() += forces.y();
                totalForce.z() += forces.z();
            }
        }
        forcePerS.write();

        Info << "Total force: " << totalForce << endl;

        Info << "Computing forces.... Completed!" << endl;
    }
    return 0;
}

// ************************************************************************* //
