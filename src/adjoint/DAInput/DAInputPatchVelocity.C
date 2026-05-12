/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v5

\*---------------------------------------------------------------------------*/

#include "DAInputPatchVelocity.H"
#include "characteristicBase.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAInputPatchVelocity, 0);
addToRunTimeSelectionTable(DAInput, DAInputPatchVelocity, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAInputPatchVelocity::DAInputPatchVelocity(
    const word inputName,
    const word inputType,
    fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : DAInput(
        inputName,
        inputType,
        mesh,
        daOption,
        daModel,
        daIndex)
{
}

void DAInputPatchVelocity::run(const scalarList& input)
{
    /*
    Description:
        Assign the input array to OF's state variables
    */

    // NOTE: we need to first update DAGlobalVar::patchVelocity here, so that daFunction-force
    // can use it to compute the force direction.
    DAGlobalVar& globalVar =
        const_cast<DAGlobalVar&>(mesh_.thisDb().lookupObject<DAGlobalVar>("DAGlobalVar"));
    globalVar.patchVelocity[0] = input[0];
    globalVar.patchVelocity[1] = input[1];

    wordList patchNames;
    dictionary patchVSubDict = daOption_.getAllOptions().subDict("inputInfo").subDict(inputName_);
    patchVSubDict.readEntry<wordList>("patches", patchNames);

#ifndef CODI_ADR
    Info << "DAInputPatchVelocity. " << endl;
    Info << "Setting UMag = " << input[0] << " AoA = " << input[1] << " degs at " << patchNames << endl;
#endif

    // the streamwise axis of aoa, aoa = tan( U_normal/U_flow )
    word flowAxis = patchVSubDict.getWord("flowAxis");
    word normalAxis = patchVSubDict.getWord("normalAxis");
    scalar UMag = input[0];

    HashTable<label> axisIndices;
    axisIndices.set("x", 0);
    axisIndices.set("y", 1);
    axisIndices.set("z", 2);
    label flowAxisIndex = axisIndices[flowAxis];
    label normalAxisIndex = axisIndices[normalAxis];

    volVectorField& U = mesh_.thisDb().lookupObjectRef<volVectorField>("U");
    volScalarField& p = mesh_.thisDb().lookupObjectRef<volScalarField>("p");
    volScalarField& T = mesh_.thisDb().lookupObjectRef<volScalarField>("T");

    scalar aoaRad = input[1] * constant::mathematical::pi / 180.0;
    scalar UxNew = UMag * cos(aoaRad);
    scalar UyNew = UMag * sin(aoaRad);

    // now we assign UxNew and UyNew to the U patches
    label hasCharFarFieldBC = 0;
    forAll(patchNames, idxI)
    {
        word patchName = patchNames[idxI];
        label patchI = mesh_.boundaryMesh().findPatchID(patchName);
        if (mesh_.boundaryMesh()[patchI].size() > 0)
        {
            if (U.boundaryField()[patchI].type() == "fixedValue")
            {
                forAll(U.boundaryField()[patchI], faceI)
                {
                    U.boundaryFieldRef()[patchI][faceI][flowAxisIndex] = UxNew;
                    U.boundaryFieldRef()[patchI][faceI][normalAxisIndex] = UyNew;
                }
            }
            else if (U.boundaryField()[patchI].type() == "inletOutlet")
            {
                mixedFvPatchField<vector>& inletOutletPatch =
                    refCast<mixedFvPatchField<vector>>(U.boundaryFieldRef()[patchI]);

                forAll(U.boundaryField()[patchI], faceI)
                {
                    inletOutletPatch.refValue()[faceI][flowAxisIndex] = UxNew;
                    inletOutletPatch.refValue()[faceI][normalAxisIndex] = UyNew;
                }
            }
            else if (U.boundaryFieldRef()[patchI].type() == "characteristicFarfieldVelocity")
            {
                // set URef value
                characteristicBase& baseU =
                    refCast<characteristicBase>(U.boundaryFieldRef()[patchI]);

                baseU.URef()[flowAxisIndex] = UxNew;
                baseU.URef()[normalAxisIndex] = UyNew;

                // IMPORTANT: for char far field BC, we also need to update BC for p and T
                // because U, p, and T BCs are inter-coupled in the char far field BC implementation.
                characteristicBase& baseP =
                    refCast<characteristicBase>(p.boundaryFieldRef()[patchI]);
                baseP.URef()[flowAxisIndex] = UxNew;
                baseP.URef()[normalAxisIndex] = UyNew;

                characteristicBase& baseT =
                    refCast<characteristicBase>(T.boundaryFieldRef()[patchI]);
                baseT.URef()[flowAxisIndex] = UxNew;
                baseT.URef()[normalAxisIndex] = UyNew;

                hasCharFarFieldBC = 1;
            }
            else
            {
                FatalErrorIn("DAInputPatchVelocity::run")
                    << "patch type not valid! only support fixedValue, inletOutlet, or characteristicFarfieldVelocity"
                    << exit(FatalError);
            }
        }
    }
    U.correctBoundaryConditions();

    reduce(hasCharFarFieldBC, sumOp<label>());
    if (hasCharFarFieldBC > 0)
    {
        p.correctBoundaryConditions();
        T.correctBoundaryConditions();
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
