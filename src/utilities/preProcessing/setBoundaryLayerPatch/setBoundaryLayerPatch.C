/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v5

    Description
        Extract the reference data for field inversion

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "OFstream.H"
#include "wallDist.H"
#include "processorFvPatchField.H"
#include "zeroGradientFvPatchField.H"

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char* argv[])
{

    Info << "Set boundary layer velocity profile to patch" << endl;

    argList::addOption(
        "patches",
        "(patch1 patch2)",
        "which patch to set the boundary layer U");

    argList::addOption(
        "mode",
        "parabolic",
        "parabolic boundary layer profile U=2*U_0/L^2 *(L*y-0.5*y^2) power law");

    argList::addOption(
        "flowAxis",
        "x",
        "flow direction");

    argList::addOption(
        "blHeight",
        "0.1",
        "boundary layer height");

    argList::addOption(
        "U0",
        "10",
        "free stream velocity");

#include "setRootCase.H"
#include "createTime.H"
#include "createMesh.H"

    word mode = "parabolic";
    if (args.optionFound("mode"))
    {
        mode = word(args.optionLookup("mode")());
        Info << "Using mode: " << mode << endl;
    }
    else
    {
        Info << "mode not set! Using the default mode: " << mode << endl;
    }

    scalar blHeight = 0.0;

    if (args.optionFound("blHeight"))
    {
        blHeight = readScalar(args.optionLookup("blHeight")());
        Info << "Using blHeight: " << blHeight << endl;
    }
    else
    {
        Info << "mode not set! Exit!" << endl;
        return 1;
    }

    scalar U0 = 0.0;
    if (args.optionFound("U0"))
    {
        U0 = readScalar(args.optionLookup("U0")());
        Info << "Using U0: " << U0 << endl;
    }
    else
    {
        Info << "U0 not set! Exit!" << endl;
        return 1;
    }

    word flowAxis = "x";
    if (args.optionFound("flowAxis"))
    {
        flowAxis = word(args.optionLookup("flowAxis")());
        Info << "Using flowAxis: " << flowAxis << endl;
    }
    else
    {
        Info << "flowAxis not set! Using the default flowAxis: " << flowAxis << endl;
    }
    label compI = 0;
    if (flowAxis == "x")
    {
        compI = 0;
    }
    else if (flowAxis == "y")
    {
        compI = 1;
    }
    else if (flowAxis == "z")
    {
        compI = 2;
    }
    else
    {
        FatalErrorInFunction
            << "flowAxis not valid. Options are: x, y, or z"
            << exit(FatalError);
    }

    wordList patches = {};
    if (args.readIfPresent("patches", patches))
    {
        Info << "The patches are " << patches << endl;
    }
    else
    {
        Info << "patches not set! Exit!" << endl;
        return 1;
    }

    Info << "Reading field U\n"
         << endl;
    volVectorField U(
        IOobject(
            "U",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE),
        mesh);

    Info << "Calculating wall distance field" << endl;
    volScalarField y(
        IOobject(
            "y",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE),
        mesh,
        dimensionedScalar(dimLength, Zero),
        zeroGradientFvPatchScalarField::typeName);
    y.primitiveFieldRef() = wallDist::New(mesh).y().primitiveField();
    y.correctBoundaryConditions();

    forAll(patches, idxI)
    {
        word patchName = patches[idxI];
        // get the patch id label
        label patchI = mesh.boundaryMesh().findPatchID(patchName);
        forAll(U.boundaryField()[patchI], faceI)
        {
            scalar yPatch = y.boundaryField()[patchI][faceI];
            if (mode == "parabolic")
            {
                if (yPatch <= blHeight)
                {
                    U.boundaryFieldRef()[patchI][faceI][compI] = (2.0 * U0 / (blHeight * blHeight)) * (blHeight * yPatch - 0.5 * yPatch * yPatch);
                }
                else
                {
                    U.boundaryFieldRef()[patchI][faceI][compI] = U0;
                }
            }
            else
            {
                FatalErrorInFunction
                    << "mode not valid. Options are: parabolic"
                    << exit(FatalError);
            }
        }
    }
    U.correctBoundaryConditions();
    U.write();

    Info << "Finished!" << endl;
    return 0;
}

// ************************************************************************* //
