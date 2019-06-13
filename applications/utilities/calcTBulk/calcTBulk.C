/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Compute bulk temperature for the U bend duct case
    Here we assume a linear increase of temperature from the inlet to outlet
    We then assign the bulk temperature for all the walls
    Users need to input the mean temperature at inlet and outlet

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

    Info<<"Computing TBulk...."<<endl;

    argList::addOption
    (
        "Tin",
        "293.15",
        "Mean temperature at inlet"
    );
    argList::addOption
    (
        "Tout",
        "303.15",
        "Mean temperature at outlet"
    );

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    volScalarField TBulk
    (
        IOobject
        (
            "TBulk",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar("TBulk",dimensionSet(0,0,0,1,0,0,0),0.0),
        fixedValueFvPatchScalarField::typeName
    );


    scalar TIn = readScalar(args.optionLookup("Tin")()),TOut=readScalar(args.optionLookup("Tout")());
    scalar dTInOut = TOut-TIn;
    scalar LTotal = 1.679;

    scalar cRefX = 0.75;
    scalar cRefY = 0.0;

    forAll(mesh.Cf().boundaryField(),patchI)
    {
        if( mesh.boundaryMesh()[patchI].type()=="wall")
        {
            forAll(mesh.Cf().boundaryField()[patchI],faceI)
            {
                scalar xx = mesh.Cf().boundaryField()[patchI][faceI].x();
                scalar yy = mesh.Cf().boundaryField()[patchI][faceI].y();
                if(xx <=0.75 and yy>0)
                {
                    TBulk.boundaryFieldRef()[patchI][faceI] = TIn+xx/LTotal*dTInOut;
                }
                else if (xx <=0.75 and yy<0)
                {
                    TBulk.boundaryFieldRef()[patchI][faceI] = TOut-xx/LTotal*dTInOut;
                }
                else
                {
                    scalar alpha = Foam::atan( (xx-cRefX)/(yy-cRefY) );
                    if (alpha<0) alpha = 3.1415926+alpha;
                    scalar ss = alpha*0.057 + 0.75;
                    TBulk.boundaryFieldRef()[patchI][faceI] = TIn+ss/LTotal*dTInOut ;
                }
            }
        }
    }

    TBulk.write();

    Info<<"Computing TBulk.... Finished!"<<endl;
 
    return 0;
}


// ************************************************************************* //
