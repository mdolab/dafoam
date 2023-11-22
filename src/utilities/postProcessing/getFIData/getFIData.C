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

#include "setRootCase.H"
#include "createTime.H"
#include "createMesh.H"

    // Read FIDataDict
    IOdictionary FIDataDict(
        IOobject(
            "FIDataDict",
            mesh.time().system(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE));

    scalar endTime = runTime.endTime().value();
    scalar deltaT = runTime.deltaT().value();
    label nInstances = round(endTime / deltaT);

    forAll(FIDataDict.toc(), idxI)
    {
        word varName = FIDataDict.toc()[idxI];
        dictionary dataSubDict = FIDataDict.subDict(varName);
        word dataType = dataSubDict.getWord("dataType");
        word fieldType = dataSubDict.getWord("fieldType");
        scalar largeValue = dataSubDict.lookupOrDefault<scalar>("largeValue", 1e16);
        labelList comps;
        if (dataSubDict.found("components"))
        {
            dataSubDict.readEntry<List<label>>("components", comps);
        }
        else
        {
            comps = {0, 1, 2};
        }

        if (dataType == "probePoints")
        {
            List<List<scalar>> probePointCoords = {{0, 0, 0}};
            dataSubDict.readEntry<List<List<scalar>>>("pointCoords", probePointCoords);
            // Info << "Reading probePointCoords from FIDataDict. probePointCoords = " << probePointCoords << endl;

            // hard coded probe point info.
            labelList probePointCellIList(probePointCoords.size(), -999);
            forAll(probePointCoords, idxI)
            {
                point point = {probePointCoords[idxI][0], probePointCoords[idxI][1], probePointCoords[idxI][2]};
                probePointCellIList[idxI] = mesh.findCell(point);
            }

            for (label n = 1; n < nInstances + 1; n++)
            {
                scalar t = n * deltaT;
                runTime.setTime(t, n);

                if (fieldType == "scalar")
                {
                    volScalarField varRead(
                        IOobject(
                            varName,
                            mesh.time().timeName(),
                            mesh,
                            IOobject::MUST_READ,
                            IOobject::NO_WRITE),
                        mesh);

                    forAll(varRead, cellI)
                    {
                        if (!probePointCellIList.found(cellI))
                        {
                            varRead[cellI] = largeValue;
                        }
                    }

                    varRead.rename(varName + "Data");
                    varRead.write();
                }
                else if (fieldType == "vector")
                {
                    volVectorField varRead(
                        IOobject(
                            varName,
                            mesh.time().timeName(),
                            mesh,
                            IOobject::MUST_READ,
                            IOobject::NO_WRITE),
                        mesh);

                    forAll(varRead, cellI)
                    {
                        if (!probePointCellIList.found(cellI))
                        {
                            for (label i = 0; i < 3; i++)
                            {
                                varRead[cellI][i] = largeValue;
                            }
                        }
                        else
                        {
                            for (label i = 0; i < 3; i++)
                            {
                                if (!comps.found(i))
                                {
                                    varRead[cellI][i] = largeValue;
                                }
                            }
                        }
                    }

                    varRead.rename(varName + "Data");
                    varRead.write();
                }
                else
                {
                    FatalErrorIn("")
                        << "fieldType" << fieldType << " not supported! Options are: scalar or vector"
                        << abort(FatalError);
                }
            }
        }
        else
        {
            FatalErrorIn("")
                << "dataType" << dataType << " not supported! Options are: probePoints"
                << abort(FatalError);
        }
    }
    Info << "Finished!" << endl;

    return 0;
}

// ************************************************************************* //
