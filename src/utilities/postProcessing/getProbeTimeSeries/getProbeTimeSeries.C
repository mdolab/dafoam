/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v5

    Description:
        Extract time-series data at a given probe point for unsteady simulations

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
        "coords",
        "'(0 0 1)'",
        "probe point coordinates");

    argList::addOption(
        "varName",
        "U",
        "name of the variable to get time-series from");

    argList::addOption(
        "varType",
        "vector",
        "type of the variable, can be either scalar or vector");

    argList::addOption(
        "outputName",
        "VarTimeSeries",
        "name of the output file (optional)");

    argList::addOption(
        "deltaT",
        "-1",
        "Use user-prescribed deltaT to extract time series, otherwise, use the deltaT in controlDict");

#include "setRootCase.H"
#include "createTime.H"
#include "createMesh.H"

    word outputName = "VarTimeSeries";
    if (args.found("outputName"))
    {
        outputName = word(args.lookup("outputName")());
    }

    List<scalar> coords;
    if (args.found("coords"))
    {
        coords = scalarList(args.lookup("coords")());
    }
    else
    {
        Info << "Error: coords not set! Exit." << endl;
        return 1;
    }
    point coordPoint = {coords[0], coords[1], coords[2]};
    label probeCellI = mesh.findCell(coordPoint);

    word varName;
    if (args.found("varName"))
    {
        varName = word(args.lookup("varName")());
    }
    else
    {
        Info << "Error: varName not set! Exit." << endl;
        return 1;
    }

    word varType;
    if (args.found("varType"))
    {
        varType = word(args.lookup("varType")());
    }
    else
    {
        Info << "Error: varType not set! Exit." << endl;
        return 1;
    }

    if (probeCellI < 0)
    {
        Info << "Error: coords " << coords << " are not within a cell! Exit." << endl;
        return 1;
    }

    if (varType != "scalar" && varType != "vector")
    {
        Info << "Error: varType = " << varType << " is not supported. The options are either scalar or vector " << endl;
    }

    OFstream f(outputName + ".txt");

    scalar endTime = runTime.endTime().value();
    scalar deltaT = runTime.deltaT().value();

    if (args.found("deltaT"))
    {
        deltaT = readScalar(args.lookup("deltaT")());
    }
    Info << "Extracting " << varName << " time series" << endl;

    label nSteps = round(endTime / deltaT);

    for (label i = 0; i < nSteps; i++)
    {
        word timeName = Foam::name(i * deltaT);

        if (varType == "vector")
        {
            volVectorField var(
                IOobject(
                    varName,
                    timeName,
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE),
                mesh);

            f << var[probeCellI][0] << " " << var[probeCellI][1] << " " << var[probeCellI][2] << endl;
        }
        else if (varType == "scalar")
        {
            volScalarField var(
                IOobject(
                    varName,
                    timeName,
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE),
                mesh);

            f << var[probeCellI] << endl;
        }
    }

    Info << "Done! " << endl;

    return 0;
}

// ************************************************************************* //
