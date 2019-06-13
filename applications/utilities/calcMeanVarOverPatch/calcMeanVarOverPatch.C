/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Compute averaged variables over a given patch
    Users need to specify the patch and variable names

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "argList.H"
#include "Time.H"
#include "fvMesh.H"

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    Info<<"Computing Mean Variable at...."<<endl;

    argList::addOption
    (
        "patchNames",
        "'(inlet)'",
        "List of patch names to compute"
    );

    argList::addOption
    (
        "varNames",
        "'(T Ux)'",
        "List of variable names to average. Can be either a volScalar or volVector var."
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
    List<wordRe> varNames;
    if (args.optionFound("varNames"))
    {
        varNames = wordReList(args.optionLookup("varNames")());
    }
    else
    {
        Info<<"varNames not set! Exit."<<endl;
        return 0;
    }

    forAll(varNames,idxI)
    {
        word varName = varNames[idxI];
        if (varName == "Ux" or varName=="Uy" or varName=="Uz")
        {
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
            label compI=-100;
            if(varName=="Ux") compI=0;
            else if (varName=="Uy") compI=1;
            else compI=2;

            forAll(patchNames,idxJ)
            {
                word patchName= patchNames[idxJ];
                label patchI = mesh.boundaryMesh().findPatchID(patchName);
                // calculate the area
                scalar Asum=0.0;
                forAll(mesh.boundaryMesh()[patchI],faceI)
                {
                    Asum += mesh.magSf().boundaryField()[patchI][faceI];
                }
                reduce(Asum, sumOp<scalar>() );
            
                scalar varMean=0.0;
                forAll(mesh.boundaryMesh()[patchI],faceI)
                {
                    varMean+=U.boundaryField()[patchI][faceI][compI]*mesh.magSf().boundaryField()[patchI][faceI]/Asum;
                }
                reduce(varMean, sumOp<scalar>() );
            
                Info<<"Mean "<<varName<<" at "<<patchName<<" : "<<varMean<<endl;

            }

        }
        else
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

            forAll(patchNames,idxJ)
            {
                word patchName= patchNames[idxJ];
                label patchI = mesh.boundaryMesh().findPatchID(patchName);
                // calculate the area
                scalar Asum=0.0;
                forAll(mesh.boundaryMesh()[patchI],faceI)
                {
                    Asum += mesh.magSf().boundaryField()[patchI][faceI];
                }
                reduce(Asum, sumOp<scalar>() );
            
                scalar varMean=0.0;
                forAll(mesh.boundaryMesh()[patchI],faceI)
                {
                    varMean+=var.boundaryField()[patchI][faceI]*mesh.magSf().boundaryField()[patchI][faceI]/Asum;
                }
                reduce(varMean, sumOp<scalar>() );
            
                Info<<"Mean "<<varName<<" at "<<patchName<<" : "<<varMean<<endl;

            }
        }
    }
 
    return 0;
}


// ************************************************************************* //
