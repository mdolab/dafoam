/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Calculating force per surface area. This will be used to plot the 
        spanwise force distribution

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "OFstream.H"
#include "simpleControl.H"
#include "fvOptions.H"
#include "fluidThermo.H"
#include "turbulentFluidThermoModel.H"

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char* argv[])
{
    Info << "Computing forcePerS...." << endl;

    argList::addOption(
        "patchNames",
        "'(inlet)'",
        "List of patch names to compute");

    argList::addOption(
        "forceDir",
        "'(0 0 1)'",
        "Force direction");

    argList::addOption(
        "time",
        "1000",
        "Tme instance to compute");

#include "setRootCase.H"
#include "createTime.H"

    scalar time;
    if (args.optionFound("time"))
    {
        time = readScalar(args.optionLookup("time")());
    }
    else
    {
        Info << "time not set! Exit." << endl;
        return 1;
    }
    runTime.setTime(time, 0);

#include "createMesh.H"
#include "createFields.H"

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

    List<scalar> forceDir1;
    if (args.optionFound("forceDir"))
    {
        forceDir1 = scalarList(args.optionLookup("forceDir")());
    }
    else
    {
        Info << "forceDir not set! Exit." << endl;
        return 1;
    }
    vector forceDir(vector::zero);
    forceDir.x() = forceDir1[0];
    forceDir.y() = forceDir1[1];
    forceDir.z() = forceDir1[2];

    volScalarField forcePerS(
        IOobject(
            "forcePerS",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE),
        mesh,
        dimensionedScalar("forcePerS", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
        fixedValueFvPatchScalarField::typeName);

    // this code is pulled from:
    // src/functionObjects/forcces/forces.C
    // modified slightly
    vector forces(vector::zero);

    const surfaceVectorField::Boundary& Sfb = mesh.Sf().boundaryField();
    const surfaceScalarField::Boundary& magSfb = mesh.magSf().boundaryField();

    tmp<volSymmTensorField> tdevRhoReff = turbulence->devRhoReff();
    const volSymmTensorField::Boundary& devRhoReffb = tdevRhoReff().boundaryField();

    forAll(patchNames, cI)
    {
        // get the patch id label
        label patchI = mesh.boundaryMesh().findPatchID(patchNames[cI]);
        // create a shorter handle for the boundary patch
        const fvPatch& patch = mesh.boundary()[patchI];
        // normal force
        vectorField fN(
            Sfb[patchI] * p.boundaryField()[patchI]);
        // tangential force
        vectorField fT(Sfb[patchI] & devRhoReffb[patchI]);
        // sum them up
        forAll(patch, faceI)
        {
            forces.x() = fN[faceI].x() + fT[faceI].x();
            forces.y() = fN[faceI].y() + fT[faceI].y();
            forces.z() = fN[faceI].z() + fT[faceI].z();
            scalar force = forces & forceDir;
            forcePerS.boundaryFieldRef()[patchI][faceI] = force / magSfb[patchI][faceI];
        }
    }
    forcePerS.write();

    Info << "Force: " << forces << endl;

    Info << "Computing forcePerS.... Completed!" << endl;

    return 0;
}

// ************************************************************************* //
