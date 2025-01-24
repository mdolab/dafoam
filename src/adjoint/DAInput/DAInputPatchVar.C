/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAInputPatchVar.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAInputPatchVar, 0);
addToRunTimeSelectionTable(DAInput, DAInputPatchVar, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAInputPatchVar::DAInputPatchVar(
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

    varName_ = daOption_.getAllOptions().subDict("inputInfo").subDict(inputName_).getWord("varName");
    varType_ = daOption_.getAllOptions().subDict("inputInfo").subDict(inputName_).getWord("varType");

    if (varType_ != "scalar" && varType_ != "vector")
    {
        FatalErrorIn("DAInputPatchVar::size") << "varType not valid" << exit(FatalError);
    }
}

void DAInputPatchVar::run(const scalarList& input)
{
    /*
    Description:
        Assign the input array to OF's state variables
    */

    wordList patchNames;
    dictionary patchVSubDict = daOption_.getAllOptions().subDict("inputInfo").subDict(inputName_);
    patchVSubDict.readEntry<wordList>("patches", patchNames);

#ifndef CODI_ADR
    Info << "DAInputPatchVar. " << endl;
    Info << "Setting " << varName_ << " = " << input << " at " << patchNames << endl;
#endif

    if (varType_ == "scalar")
    {

        volScalarField& var = const_cast<volScalarField&>(mesh_.thisDb().lookupObject<volScalarField>(varName_));

        // now we assign input to the var patches
        forAll(patchNames, idxI)
        {
            word patchName = patchNames[idxI];
            label patchI = mesh_.boundaryMesh().findPatchID(patchName);
            if (mesh_.boundaryMesh()[patchI].size() > 0)
            {
                if (var.boundaryField()[patchI].type() == "fixedValue")
                {
                    forAll(var.boundaryField()[patchI], faceI)
                    {
                        var.boundaryFieldRef()[patchI][faceI] = input[0];
                    }
                }
                else if (var.boundaryField()[patchI].type() == "inletOutlet")
                {
                    mixedFvPatchField<scalar>& inletOutletPatch =
                        refCast<mixedFvPatchField<scalar>>(var.boundaryFieldRef()[patchI]);

                    forAll(var.boundaryField()[patchI], faceI)
                    {
                        inletOutletPatch.refValue()[faceI] = input[0];
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
        var.correctBoundaryConditions();
    }
    if (varType_ == "vector")
    {

        volVectorField& var = const_cast<volVectorField&>(mesh_.thisDb().lookupObject<volVectorField>(varName_));

        // now we assign input to the var patches
        forAll(patchNames, idxI)
        {
            word patchName = patchNames[idxI];
            label patchI = mesh_.boundaryMesh().findPatchID(patchName);
            if (mesh_.boundaryMesh()[patchI].size() > 0)
            {
                if (var.boundaryField()[patchI].type() == "fixedValue")
                {
                    forAll(var.boundaryField()[patchI], faceI)
                    {
                        for (label i = 0; i < 3; i++)
                        {
                            var.boundaryFieldRef()[patchI][faceI][i] = input[i];
                        }
                    }
                }
                else if (var.boundaryField()[patchI].type() == "inletOutlet")
                {
                    mixedFvPatchField<vector>& inletOutletPatch =
                        refCast<mixedFvPatchField<vector>>(var.boundaryFieldRef()[patchI]);

                    forAll(var.boundaryField()[patchI], faceI)
                    {
                        for (label i = 0; i < 3; i++)
                        {
                            inletOutletPatch.refValue()[faceI][i] = input[i];
                        }
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
        var.correctBoundaryConditions();
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
