/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

Description
    Deform the mesh for field inversion

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

    Info << "Deform the mesh for field inversion...." << endl;

    argList::addOption(
        "origin",
        "(0 0 0)",
        "prescribe the origin to deform the mesh");

    argList::addOption(
        "axis",
        "(0 0 0)",
        "prescribe the axis to deform the mesh");

    argList::addOption(
        "omega",
        "0",
        "prescribe the angular velocity to deform the mesh");

    argList::addOption(
        "time",
        "-1",
        "prescribe a specific time to deform the mesh");

#include "setRootCase.H"
#include "createTime.H"
#include "createMesh.H"

    vector origin = {0.0, 0.0, 0.0};
    if (args.readIfPresent("origin", origin))
    {
        Info << "The origin to rotate the mesh: " << origin << endl;
    }
    else
    {
        Info << "origin not set!" << endl;
    }

    vector axis = {0, 0, 0};
    if (args.readIfPresent("axis", axis))
    {
        Info << "The axis to rotate the mesh: " << axis << endl;
    }
    else
    {
        Info << "axis center not set!" << endl;
    }

    scalar omega = 0.0;
    if (args.readIfPresent("omega", omega))
    {
        Info << "The angular velocity to rotate the mesh: " << omega << endl;
    }
    else
    {
        Info << "omega not set! Don't rotate mesh." << endl;
    }

    scalar time = -1.0;
    if (args.optionFound("time"))
    {
        time = readScalar(args.optionLookup("time")());
        if (time == 9999)
        {
            Info << "Deform latestTime" << endl;
        }
        else
        {
            Info << "Deform time = " << time << endl;
        }
    }
    else
    {
        Info << "Time not set! Deform all time instances." << endl;
    }

    Info << "Deforming the mesh..." << endl;

    while (runTime.run())
    {
        ++runTime;

        Info << "Time = " << runTime.timeName() << nl << endl;

        pointField ourNewPoints(mesh.points());
        scalar theta = omega * runTime.deltaT().value();
        scalar cosTheta = std::cos(theta);
        scalar sinTheta = std::sin(theta);

        forAll(ourNewPoints, pointI)
        {
            scalar xTemp = ourNewPoints[pointI][0] - origin[0];
            scalar yTemp = ourNewPoints[pointI][1] - origin[1];

            ourNewPoints[pointI][0] = cosTheta * xTemp - sinTheta * yTemp + origin[0];
            ourNewPoints[pointI][1] = sinTheta * xTemp + cosTheta * yTemp + origin[1];
        }

        mesh.movePoints(ourNewPoints);

        pointIOField writePoints(
            IOobject(
                "points",
                runTime.timeName(),
                "polyMesh",
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE),
            mesh.points());

        runTime.write();
    }

    Info << "Finished!" << endl;

    return 0;
}

// ************************************************************************* //
