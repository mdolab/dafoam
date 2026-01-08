/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v5

    Description:
        Extract time-series for field RMSE for unsteady simulations

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
        "varName",
        "U",
        "name of the variable to get time-series from");

    argList::addOption(
        "varType",
        "vector",
        "type of the variable, can be either scalar or vector");

    argList::addOption(
        "fieldType",
        "volume",
        "type of the field variable, can be either volume or surface");

    argList::addOption(
        "patchName",
        "wing",
        "if the fieldType=surface, we need to prescribe the name of the patch");

    argList::addOption(
        "outputName",
        "fieldRefStdTimeSeries",
        "name of the output file (optional)");

    argList::addOption(
        "deltaT",
        "-1",
        "Use user-prescribed deltaT to extract time series, otherwise, use the deltaT in controlDict");

#include "setRootCase.H"
#include "createTime.H"
#include "createMesh.H"

    word outputName = "fieldRefStdTimeSeries";
    if (args.found("outputName"))
    {
        outputName = word(args.lookup("outputName")());
    }

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

    word varRefName = varName + "Data";

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

    word fieldType;
    if (args.found("fieldType"))
    {
        fieldType = word(args.lookup("fieldType")());
    }
    else
    {
        Info << "Error: fieldType not set! Exit." << endl;
        return 1;
    }

    word patchName;
    if (fieldType == "surface")
    {
        if (args.found("patchName"))
        {
            patchName = word(args.lookup("patchName")());
        }
        else
        {
            Info << "Error: patchName not set! Exit." << endl;
            return 1;
        }
    }

    OFstream f(outputName + ".txt");

    scalar endTime = runTime.endTime().value();
    scalar deltaT = runTime.deltaT().value();

    if (args.found("deltaT"))
    {
        deltaT = readScalar(args.lookup("deltaT")());
    }
    Info << "Extracting " << fieldType << " time series for " << varName << endl;

    label nSteps = round(endTime / deltaT);

    for (label i = 0; i < nSteps; i++)
    {
        word timeName = Foam::name((i + 1) * deltaT);

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

            volVectorField varRef(
                IOobject(
                    varRefName,
                    timeName,
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE),
                mesh);

            vector std = vector::zero;

            if (fieldType == "volume")
            {

                forAll(var, cellI)
                {
                    for (label compI = 0; compI < 3; compI++)
                    {
                        std[compI] += (var[cellI][compI] - varRef[cellI][compI]) * (var[cellI][compI] - varRef[cellI][compI]);
                    }
                }

                for (label compI = 0; compI < 3; compI++)
                {
                    std[compI] = Foam::sqrt(std[compI] / mesh.nCells());
                }
            }
            else if (fieldType == "surface")
            {
                label patchI = mesh.boundaryMesh().findPatchID(patchName);
                if (patchI < 0)
                {
                    Info << "Error. The prescribed patchName " << patchName << " not found in constant/polyMesh/boundary" << endl;
                }
                label nFaces = var.boundaryField()[patchI].size();
                forAll(var.boundaryField()[patchI], faceI)
                {
                    vector varBC = var.boundaryField()[patchI][faceI];
                    vector varRefBC = varRef.boundaryField()[patchI][faceI];

                    for (label compI = 0; compI < 3; compI++)
                    {
                        std[compI] += (varBC[compI] - varRefBC[compI]) * (varBC[compI] - varRefBC[compI]);
                    }
                }

                for (label compI = 0; compI < 3; compI++)
                {
                    std[compI] = Foam::sqrt(std[compI] / nFaces);
                }
            }

            f << std[0] << " " << std[1] << " " << std[2] << endl;
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

            volScalarField varRef(
                IOobject(
                    varRefName,
                    timeName,
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE),
                mesh);

            scalar std = 0.0;

            if (fieldType == "volume")
            {
                forAll(var, cellI)
                {
                    std += (var[cellI] - varRef[cellI]) * (var[cellI] - varRef[cellI]);
                }

                std = Foam::sqrt(std / mesh.nCells());
            }
            else if (fieldType == "surface")
            {
                label patchI = mesh.boundaryMesh().findPatchID(patchName);
                if (patchI < 0)
                {
                    Info << "Error. The prescribed patchName " << patchName << " not found in constant/polyMesh/boundary" << endl;
                }
                label nFaces = var.boundaryField()[patchI].size();
                forAll(var.boundaryField()[patchI], faceI)
                {
                    scalar varBC = var.boundaryField()[patchI][faceI];
                    scalar varRefBC = varRef.boundaryField()[patchI][faceI];
                    std += (varBC - varRefBC) * (varBC - varRefBC);
                }

                std = Foam::sqrt(std / nFaces);
            }

            f << std << endl;
        }
        else
        {
            Info << "Error: varType not valid. Can be either scalar or vector! Exit." << endl;
        }
    }

    Info << "Done! " << endl;

    return 0;
}

// ************************************************************************* //
