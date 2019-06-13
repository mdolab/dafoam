/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Assign the near wall cell values based on the patch values. 

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
        "varName",
        "NUS",
        "Variable name to set"
    );

    argList::addOption
    (
        "varType",
        "scalar",
        "Is the variable a scalar or a vector?"
    );

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    word varName;
    if (args.optionFound("varName"))
    {
        varName = word(args.optionLookup("varName")());
    }
    else
    {
        Info<<"varName not set! Exit."<<endl;
        return 1;
    }

    word varType;
    if (args.optionFound("varType"))
    {
        varType = word(args.optionLookup("varType")());
    }
    else
    {
        Info<<"Using default varType = scalar"<<endl;
        varType = "scalar";
    }

    Info<<argc<<endl;

    for (int i = 0; i < argc; ++i) {
        std::cout << argv[i] << std::endl;
    }

    if (varType == "scalar")
    {
        volScalarField var
        (
            IOobject
            (
                varName,
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            ),
            mesh
        );
    
        forAll(mesh.Cf().boundaryField(),patchI)
        {
            if( mesh.boundaryMesh()[patchI].type()=="wall")
            {
                const UList<label>& pFaceCells = mesh.boundaryMesh()[patchI].faceCells();
                forAll(pFaceCells,faceI)
                {
                    label cellI = pFaceCells[faceI];
                    var[cellI]=var.boundaryField()[patchI][faceI];
                }
            }
        }
        var.rename(varName+"NearWallCells");
        var.write();
    }
    else if (varType == "vector")
    {
        volVectorField var
        (
            IOobject
            (
                varName,
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            ),
            mesh
        );
    
        forAll(mesh.Cf().boundaryField(),patchI)
        {
            if( mesh.boundaryMesh()[patchI].type()=="wall")
            {
                const UList<label>& pFaceCells = mesh.boundaryMesh()[patchI].faceCells();
                forAll(pFaceCells,faceI)
                {
                    label cellI = pFaceCells[faceI];
                    var[cellI]=var.boundaryField()[patchI][faceI];
                }
            }
        }
        var.rename(varName+"NearWallCells");
        var.write();
    }
    else
    {
        Info<<"varType not supported! Exit!"<<endl;
        return 1;
    }
 
    return 0;
}


// ************************************************************************* //
