/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Calculating lift per surface area. This will be used to plot the 
        spanwise lift distribution

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "OFstream.H"
#include "simpleControl.H"
#include "fvOptions.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char* argv[])
{
    Info << "Computing liftPerS...." << endl;

    argList::addOption(
        "patchNames",
        "'(inlet)'",
        "List of patch names to compute");

    argList::addOption(
        "liftDir",
        "'(0 0 1)'",
        "Lift direction");

#include "setRootCase.H"
#include "createTime.H"
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
        return 0;
    }

    List<scalar> liftDir1;
    if (args.optionFound("liftDir"))
    {
        liftDir1 = scalarList(args.optionLookup("liftDir")());
    }
    else
    {
        Info << "liftDir not set! Exit." << endl;
        return 0;
    }
    vector liftDir(vector::zero);
    liftDir.x() = liftDir1[0];
    liftDir.y() = liftDir1[1];
    liftDir.z() = liftDir1[2];

    volScalarField liftPerS(
        IOobject(
            "liftPerS",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE),
        mesh,
        dimensionedScalar("liftPerS", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
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
            scalar lift = forces & liftDir;
            liftPerS.boundaryFieldRef()[patchI][faceI] = lift / magSfb[patchI][faceI];
        }
    }
    liftPerS.write();

    Info << "Force: " << forces << endl;

    Info << "Computing liftPerS.... Completed!" << endl;

    return 0;
}

// ************************************************************************* //
