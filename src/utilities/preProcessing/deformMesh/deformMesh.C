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
        "omega",
        "0",
        "prescribe an rotational angular velocity to deform the mesh");

    argList::addOption(
        "xc",
        "0",
        "prescribe the x center to deform the mesh");

    argList::addOption(
        "yc",
        "0",
        "prescribe the y center to deform the mesh");

    argList::addOption(
        "time",
        "-1",
        "prescribe a specific time to deform the mesh");

#include "setRootCase.H"
#include "createTime.H"
#include "createMesh.H"

    scalar omega = 0.0;
    if (args.optionFound("omega"))
    {
        omega = readScalar(args.optionLookup("omega")());
    }
    else
    {
        Info << "Omega not set! Don't rotate mesh." << endl;
    }

    scalar xc = 0.0;
    if (args.optionFound("xc"))
    {
        xc = readScalar(args.optionLookup("xc")());
    }
    else
    {
        Info << "x center not set!" << endl;
    }

    scalar yc = 0.0;
    if (args.optionFound("yc"))
    {
        yc = readScalar(args.optionLookup("yc")());
    }
    else
    {
        Info << "y center not set!" << endl;
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
        Info<< "Time = " << runTime.timeName() << nl << endl;

        pointField ourNewPoints(mesh.points());
        scalar theta = omega;
        scalar cosTheta = std::cos(theta);
        scalar sinTheta = std::sin(theta);

        forAll(ourNewPoints, pointI)
        {
            scalar xTemp = ourNewPoints[pointI][0] - xc;
            scalar yTemp = ourNewPoints[pointI][1] - yc;

            ourNewPoints[pointI][0] = cosTheta * xTemp - sinTheta * yTemp + xc;
            ourNewPoints[pointI][1] = sinTheta * xTemp + cosTheta * yTemp + yc;
        }
        
        mesh.movePoints(ourNewPoints);

        pointIOField writePoints
        (
            IOobject
            (
                "points",
                runTime.timeName(),
                "polyMesh",
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh.points()
        );
        
        runTime.write();
    }

    Info << "Finished!" << endl;

    return 0;
}

// ************************************************************************* //
