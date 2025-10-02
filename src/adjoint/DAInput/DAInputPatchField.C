/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAInputPatchField.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAInputPatchField, 0);
addToRunTimeSelectionTable(DAInput, DAInputPatchField, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAInputPatchField::DAInputPatchField(
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

    fieldName_ = daOption_.getAllOptions().subDict("inputInfo").subDict(inputName).getWord("fieldName");
    fieldType_ = daOption_.getAllOptions().subDict("inputInfo").subDict(inputName).getWord("fieldType");

    daOption_.getAllOptions().subDict("inputInfo").subDict(inputName).readEntry<wordList>("patches", patches_);

    localPatchFaces_ = 0;
    forAll(patches_, idxI)
    {
        word patchName = patches_[idxI];
        label patchI = mesh_.boundaryMesh().findPatchID(patchName);
        localPatchFaces_ += mesh_.boundaryMesh()[patchI].size();
    }

    globalPatchFaceNumbering_ = DAUtility::genGlobalIndex(localPatchFaces_);

    globalPatchFaces_ = globalPatchFaceNumbering_.size();

    if (daOption_.getAllOptions().subDict("inputInfo").subDict(inputName).found("indices"))
    {
        daOption_.getAllOptions().subDict("inputInfo").subDict(inputName).readEntry("indices", indices_);
    }
    else
    {
        indices_.setSize(3);
        for (label i = 0; i < 3; i++)
        {
            indices_[i] = i;
        }
    }
}

void DAInputPatchField::run(const scalarList& input)
{
    /*
    Description:
        Assign the input array to OF's state variables
    */

#ifndef CODI_ADR
    Info << "DAInputPatchField. " << endl;
    Info << "Setting " << fieldName_ << " field at " << patches_ << endl;
#endif

    label inputIdx = -1;
    if (fieldType_ == "scalar")
    {
        volScalarField& var = const_cast<volScalarField&>(mesh_.thisDb().lookupObject<volScalarField>(fieldName_));

        // now we assign input to the var patches
        label localIdx = 0;
        forAll(patches_, idxI)
        {
            word patchName = patches_[idxI];
            label patchI = mesh_.boundaryMesh().findPatchID(patchName);

            if (var.boundaryField()[patchI].type() == "fixedValue")
            {
                forAll(var.boundaryField()[patchI], faceI)
                {
                    if (this->distributed())
                    {
                        inputIdx = localIdx;
                    }
                    else
                    {
                        inputIdx = globalPatchFaceNumbering_.toGlobal(localIdx);
                    }
                    var.boundaryFieldRef()[patchI][faceI] = input[inputIdx];
                    localIdx++;
                }
            }
            else if (var.boundaryField()[patchI].type() == "inletOutlet")
            {
                mixedFvPatchField<scalar>& inletOutletPatch =
                    refCast<mixedFvPatchField<scalar>>(var.boundaryFieldRef()[patchI]);

                forAll(var.boundaryField()[patchI], faceI)
                {
                    if (this->distributed())
                    {
                        inputIdx = localIdx;
                    }
                    else
                    {
                        inputIdx = globalPatchFaceNumbering_.toGlobal(localIdx);
                    }
                    inletOutletPatch.refValue()[faceI] = input[inputIdx];
                    localIdx++;
                }
            }
            else
            {
                FatalErrorIn("DAInputPatchField::run")
                    << "patch type not valid! only support fixedValue or inletOutlet"
                    << exit(FatalError);
            }
        }
        var.correctBoundaryConditions();
    }
    if (fieldType_ == "vector")
    {

        volVectorField& var = const_cast<volVectorField&>(mesh_.thisDb().lookupObject<volVectorField>(fieldName_));

        // now we assign input to the var patches
        label localIdx = 0;
        forAll(patches_, idxI)
        {
            word patchName = patches_[idxI];
            label patchI = mesh_.boundaryMesh().findPatchID(patchName);

            if (var.boundaryField()[patchI].type() == "fixedValue")
            {
                forAll(var.boundaryField()[patchI], faceI)
                {
                    forAll(indices_, j)
                    {
                        label idxJ = indices_[j];
                        if (this->distributed())
                        {
                            inputIdx = localIdx;
                        }
                        else
                        {
                            inputIdx = globalPatchFaceNumbering_.toGlobal(localIdx);
                        }
                        var.boundaryFieldRef()[patchI][faceI][idxJ] = input[inputIdx];
                        localIdx++;
                    }
                }
            }
            else if (var.boundaryField()[patchI].type() == "inletOutlet")
            {
                mixedFvPatchField<vector>& inletOutletPatch =
                    refCast<mixedFvPatchField<vector>>(var.boundaryFieldRef()[patchI]);

                forAll(var.boundaryField()[patchI], faceI)
                {
                    forAll(indices_, j)
                    {
                        label idxJ = indices_[j];
                        if (this->distributed())
                        {
                            inputIdx = localIdx;
                        }
                        else
                        {
                            inputIdx = globalPatchFaceNumbering_.toGlobal(localIdx);
                        }
                        inletOutletPatch.refValue()[faceI][idxJ] = input[inputIdx];
                        localIdx++;
                    }
                }
            }
            else
            {
                FatalErrorIn("DAInputPatchField::run")
                    << "patch type not valid! only support fixedValue or inletOutlet"
                    << exit(FatalError);
            }
        }
        var.correctBoundaryConditions();
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
