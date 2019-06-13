/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Compute wall distance for a given patch

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "OFstream.H"
#include "wallDist.H"

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{

    argList::addOption
    (
        "patchName",
        "inlet",
        "Patch name to compute wall distance"
    );

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    // read options
    word patchName;
    if (args.optionFound("patchName"))
    {
        patchName = word(args.optionLookup("patchName")());
    }
    else
    {
        Info<<"patchName not set! Exit."<<endl;
        return 0;
    }

    Info<<endl<<"Computing wall distance for "<<patchName<<endl<<endl;

    volScalarField y=wallDist::New(mesh).y();
    scalar maxY=-1000.0,minY=10000.0;
    forAll(mesh.boundaryMesh(),patchI)
    {
        if (mesh.boundaryMesh()[patchI].name()==patchName)
        {
            const UList<label>& pFaceCells = mesh.boundaryMesh()[patchI].faceCells();
            forAll(mesh.boundaryMesh()[patchI],faceI)
            {
                label idxN = pFaceCells[faceI];
                if(y[idxN]>maxY) maxY=y[idxN];
                if(y[idxN]<minY) minY=y[idxN];
                Info<<y[idxN]<<endl;
            }
        }
    }

    Info<<endl<<"maxWallDist: "<<maxY<<endl;
    Info<<endl<<"minWallDist: "<<minY<<endl;
 
    return 0;
}


// ************************************************************************* //
