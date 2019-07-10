/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Set the rotating velocity for a specified boundary, used in MRF simulations

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
        "rotCenter",
        "'(0 0 0)'",
        "Rotation center"
    );

    argList::addOption
    (
        "rotRad",
        "'(0 0 100)'",
        "Rotation speed in rad/s"
    );

    argList::addOption
    (
        "UBase",
        "'(0 0 0)'",
        "Baseline velocity that will be added on top of the rotating velocity"
    );

    argList::addOption
    (
        "patchNames",
        "'(body)'",
        "List of patch names to compute the rotating velocity"
    );

    argList::addOption
    (
        "UName",
        "U",
        "Variable name for velocity"
    );

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    vector rotCenter=vector::zero;
    if (args.optionFound("rotCenter"))
    {
        scalarList tmpList=args.optionLookup("rotCenter")();
        forAll(tmpList,idxI)
        {
            rotCenter[idxI] = tmpList[idxI];
        }
        Info<<"rotCenter: "<<rotCenter<<endl;
    }
    else
    {
        Info<<"rotCenter not set! Using default (0 0 0)"<<endl;
    }

    vector rotRad=vector::zero;
    if (args.optionFound("rotRad"))
    {
        scalarList tmpList=args.optionLookup("rotRad")();
        forAll(tmpList,idxI)
        {
            rotRad[idxI] = tmpList[idxI];
        }
        Info<<"rotRad: "<<rotRad<<endl;
    }
    else
    {
        Info<<"rotRad not set! Exit!"<<endl;
        return 1;
    }

    vector UBase=vector::zero;
    if (args.optionFound("UBase"))
    {
        scalarList tmpList=args.optionLookup("UBase")();
        forAll(tmpList,idxI)
        {
            UBase[idxI] = tmpList[idxI];
        }
        Info<<"UBase: "<<UBase<<endl;
    }
    else
    {
        Info<<"UBase not set! Using default (0 0 0)"<<endl;
    }

    List<wordRe> patchNames;
    if (args.optionFound("patchNames"))
    {
        patchNames = wordReList(args.optionLookup("patchNames")());
        Info<<"patchNames: "<<patchNames<<endl;
    }
    else
    {
        Info<<"patchNames not set! Exit."<<endl;
        return 0;
    }

    word UName;
    if (args.optionFound("UName"))
    {
        UName = word(args.optionLookup("UName")());
        Info<<"UName: "<<UName<<endl;
    }
    else
    {
        Info<<"UName not set! Using default: U"<<endl;
        UName = "U";
    }

    vector rotDir = vector::zero;
    rotDir = rotRad/ mag(rotRad);

    volVectorField U
    (
        IOobject
        (
            UName,
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        ),
        mesh
    );

    forAll(U.boundaryField(),patchI)
    {
        forAll(patchNames,idxI)
        {
            word patchName = patchNames[idxI];
            if( mesh.boundaryMesh()[patchI].name()==patchName)
            {
                forAll(U.boundaryField()[patchI],faceI)
                {
                    const vector& cellC =  mesh.Cf().boundaryField()[patchI][faceI];
                    vector cellC2AVec = cellC-rotCenter; // cell center to rot center vector
                    
                    tensor cellC2AVecE(tensor::zero); // tmp tensor for calculating the axial/radial components of cellC2AVec
                    cellC2AVecE.xx() = cellC2AVec.x();
                    cellC2AVecE.yy() = cellC2AVec.y();
                    cellC2AVecE.zz() = cellC2AVec.z();
                    
                    vector cellC2AVecA = cellC2AVecE & rotDir; // axial
                    vector cellC2AVecR = cellC2AVec-cellC2AVecA; // radial
    
                    vector Urot = rotRad ^ cellC2AVecR;
    
                    Urot += UBase;
    
                    U.boundaryFieldRef()[patchI][faceI] = Urot;
                    
                }
            }
        }
    }
    U.write();
    Info<<"Done!"<<endl;
 
    return 0;
}


// ************************************************************************* //
