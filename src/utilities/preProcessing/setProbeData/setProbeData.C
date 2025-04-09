/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

Description
    Set reference data into a field by prescribing a probe point coordinate

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

    argList::addOption(
        "fieldName",
        "UData",
        "field variables name to set");

    argList::addOption(
        "fieldType",
        "vector",
        "field variables type");

    argList::addOption(
        "time",
        "-1",
        "prescribe a specific time to set data");

    argList::addOption(
        "probeCoord",
        "(0 0 0)",
        "prescribe the probe point coordinate");

    argList::addOption(
        "value",
        "(1 0 0)",
        "prescribe the value to add to the probe point coordinate. If it is a scalar, set the value for first element and set zeros for the other two elements");

    argList::addOption(
        "mode",
        "findCell",
        "The mode can be either findCell or findNearestCell");

#include "setRootCase.H"
#include "createTime.H"
#include "createMesh.H"

    word fieldName;
    if (args.optionFound("fieldName"))
    {
        fieldName = word(args.optionLookup("fieldName")());
    }
    else
    {
        Info << "fieldName not set! Exit." << endl;
        return 1;
    }

    word fieldType;
    if (args.optionFound("fieldType"))
    {
        fieldType = word(args.optionLookup("fieldType")());
    }
    else
    {
        Info << "fieldType not set! Exit." << endl;
        return 1;
    }

    word mode = "findCell";
    if (args.optionFound("mode"))
    {
        mode = word(args.optionLookup("mode")());
    }
    else
    {
        Info << "mode not set! Use the default findCell mode." << endl;
    }

    vector probeCoord = {0.0, 0.0, 0.0};
    if (args.readIfPresent("probeCoord", probeCoord))
    {
        Info << "The probeCoord is: " << probeCoord << endl;
    }
    else
    {
        Info << "probeCoord not set!" << endl;
    }

    vector value = {0.0, 0.0, 0.0};
    if (args.readIfPresent("value", value))
    {
        Info << "The value is: " << value << endl;
    }
    else
    {
        Info << "value not set!" << endl;
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

    Info << "Set field " << fieldName << " value at " << probeCoord << " with " << value << endl;

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
        Info << "Working on time = " << t << endl;

        if (fieldType == "scalar")
        {
            volScalarField fieldRead(
                IOobject(
                    fieldName,
                    mesh.time().timeName(),
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE),
                mesh);

            if (mode == "findCell")
            {
                point coordPoint = {probeCoord[0], probeCoord[1], probeCoord[2]};
                label probeCellI = mesh.findCell(coordPoint);
                if (probeCellI < 0)
                {
                    Info << "Cell not found for coord: " << probeCoord << endl;
                    Info << "Exit!" << endl;
                    return 0;
                }
                fieldRead[probeCellI] = value[0];
            }
            else if (mode == "findNearestCell")
            {
                point coordPoint = {probeCoord[0], probeCoord[1], probeCoord[2]};
                label probeCellI = mesh.findNearestCell(coordPoint);
                fieldRead[probeCellI] = value[0];
            }
            else
            {
                FatalErrorIn("mode not valid! Options are findCell or findNearestCell")
                    << exit(FatalError);
            }
            fieldRead.write();
        }
        else if (fieldType == "vector")
        {
            volVectorField fieldRead(
                IOobject(
                    fieldName,
                    mesh.time().timeName(),
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE),
                mesh);

            if (mode == "findCell")
            {
                point coordPoint = {probeCoord[0], probeCoord[1], probeCoord[2]};
                label probeCellI = mesh.findCell(coordPoint);
                fieldRead[probeCellI] = value;
            }
            else if (mode == "findNearestCell")
            {
                point coordPoint = {probeCoord[0], probeCoord[1], probeCoord[2]};
                label probeCellI = mesh.findNearestCell(coordPoint);
                fieldRead[probeCellI] = value;
            }
            else
            {
                FatalErrorIn("mode not valid! Options are findCell or findNearestCell")
                    << exit(FatalError);
            }
            fieldRead.write();
        }
        else
        {
            Info << "fieldName and type not supported!" << endl;
        }
    }

    Info << "Finished!" << endl;

    return 0;
}

// ************************************************************************* //
