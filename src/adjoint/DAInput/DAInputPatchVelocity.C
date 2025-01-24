/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAInputPatchVelocity.H"

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

    volVectorField& U = const_cast<volVectorField&>(mesh_.thisDb().lookupObject<volVectorField>("U"));

    scalar aoaRad = input[1] * constant::mathematical::pi / 180.0;
    scalar UxNew = UMag * cos(aoaRad);
    scalar UyNew = UMag * sin(aoaRad);

    // now we assign UxNew and UyNew to the U patches
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
            else
            {
                FatalErrorIn("DAInputPatchVelocity::run")
                    << "patch type not valid! only support fixedValue or inletOutlet"
                    << exit(FatalError);
            }
        }
    }
    U.correctBoundaryConditions();
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
