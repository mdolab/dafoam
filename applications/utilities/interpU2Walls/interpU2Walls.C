/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

    Description:
    Interpolate velocity to walls such that we can use surfaceLIC to view 
    limiting streamline in paraview

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "OFstream.H"

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{

    argList::addOption
    (
        "patchNames",
        "'(body)'",
        "List of patch names to compute"
    );

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

     // read options
    List<wordRe> patchNames;
    if (args.optionFound("patchNames"))
    {
        patchNames = wordReList(args.optionLookup("patchNames")());
    }
    else
    {
        Info<<"patchNames not set! Exit."<<endl;
        return 0;
    }

    Info<<endl<<"Interpolating U to "<<patchNames<<endl<<endl;

    volVectorField U
    (
        IOobject
        (
            "U",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        ),
        mesh
    );

    volVectorField UWall("UWall",U);
    forAll(UWall,idxI) UWall[idxI]=vector::zero;
    forAll(UWall.boundaryField(),patchI)
    {
        forAll(UWall.boundaryField()[patchI],faceI)
        {
            UWall.boundaryFieldRef()[patchI][faceI]=vector::zero;
        }
    }

    forAll(patchNames,idxJ)
    {
        word patchName= patchNames[idxJ];
        label patchI = mesh.boundaryMesh().findPatchID(patchName);
        const UList<label>& pFaceCells = mesh.boundaryMesh()[patchI].faceCells();

        forAll(mesh.boundaryMesh()[patchI],faceI)
        {
            label idxN = pFaceCells[faceI];
            UWall.boundaryFieldRef()[patchI][faceI]=U[idxN];
        }    
        
    }

    UWall.write();

    Info<<endl<<"Interpolating U to "<<patchNames<<" Completed!"<<endl<<endl;

    return 0;
}


// ************************************************************************* //
