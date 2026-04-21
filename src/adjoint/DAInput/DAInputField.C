/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v5

\*---------------------------------------------------------------------------*/

#include "DAInputField.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAInputField, 0);
addToRunTimeSelectionTable(DAInput, DAInputField, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAInputField::DAInputField(
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
    // Cache the input sub-dictionary since we query several optional entries below.
    const dictionary& inputDict = daOption_.getAllOptions().subDict("inputInfo").subDict(inputName);

    fieldName_ = inputDict.getWord("fieldName");
    fieldType_ = inputDict.getWord("fieldType");

    if (inputDict.found("indices"))
    {
        inputDict.readEntry("indices", indices_);
    }
    else
    {
        indices_.setSize(3);
        for (label i = 0; i < 3; i++)
        {
            indices_[i] = i;
        }
    }

    if (inputDict.found("cellSetName"))
    {
        cellSetName_ = inputDict.getWord("cellSetName");
        // Read the local portion of a pre-generated cellSet. In parallel, users are
        // expected to decompose the set together with the mesh.
        cellSet selectedCellSet(mesh_, cellSetName_, IOobject::MUST_READ);
        selectedCells_.setSize(selectedCellSet.size());

        label idx = 0;
        for (const label cellI : selectedCellSet)
        {
            selectedCells_[idx] = cellI;
            idx++;
        }
        // Sort so the subset has a deterministic local ordering across runs.
        sort(selectedCells_);
    }
    else
    {
        // Fall back to the original behavior: use every local cell.
        selectedCells_.setSize(mesh_.nCells());
        forAll(selectedCells_, idxI)
        {
            selectedCells_[idxI] = idxI;
        }
    }

    localSelectedCells_ = selectedCells_.size();
    // Build a compact global numbering for the selected subset instead of all cells.
    globalSelectedCellNumbering_ = DAUtility::genGlobalIndex(localSelectedCells_);
    globalSelectedCells_ = globalSelectedCellNumbering_.size();

    if (globalSelectedCells_ == 0)
    {
        FatalErrorIn("DAInputField::DAInputField")
            << "Input " << inputName_ << " has zero selected cells";
        if (cellSetName_.size())
        {
            FatalError << " in cellSet " << cellSetName_;
        }
        FatalError << exit(FatalError);
    }
}

void DAInputField::run(const scalarList& input)
{
    /*
    Description:
        Assign the input array to OF's field variables. Note that we need different treatment for distributed and 
        non-distributed field inputs
    */

    if (fieldType_ == "scalar")
    {
        volScalarField& field = const_cast<volScalarField&>(mesh_.thisDb().lookupObject<volScalarField>(fieldName_));
        if (this->distributed())
        {
            forAll(selectedCells_, localIdx)
            {
                label cellI = selectedCells_[localIdx];
                field[cellI] = input[localIdx];
            }
        }
        else
        {
            forAll(selectedCells_, localIdx)
            {
                label cellI = selectedCells_[localIdx];
                // Map the local subset index to the compact global subset numbering.
                label globalCellI = globalSelectedCellNumbering_.toGlobal(localIdx);
                field[cellI] = input[globalCellI];
            }
        }
        field.correctBoundaryConditions();
    }
    else if (fieldType_ == "vector")
    {
        volVectorField& field = const_cast<volVectorField&>(mesh_.thisDb().lookupObject<volVectorField>(fieldName_));
        if (this->distributed())
        {
            label cSize = indices_.size();
            forAll(selectedCells_, localIdx)
            {
                label cellI = selectedCells_[localIdx];
                forAll(indices_, idxI)
                {
                    label comp = indices_[idxI];
                    label inputIdx = localIdx * cSize + idxI;
                    field[cellI][comp] = input[inputIdx];
                }
            }
        }
        else
        {
            label cSize = indices_.size();
            forAll(selectedCells_, localIdx)
            {
                label cellI = selectedCells_[localIdx];
                // For non-distributed inputs, each selected cell owns a contiguous block
                // of size cSize in the packed global subset array.
                label globalCellI = globalSelectedCellNumbering_.toGlobal(localIdx);
                forAll(indices_, idxI)
                {
                    label comp = indices_[idxI];
                    label inputIdx = globalCellI * cSize + idxI;
                    field[cellI][comp] = input[inputIdx];
                }
            }
        }
        field.correctBoundaryConditions();
    }
    else
    {
        FatalErrorIn("DAInputField::run") << exit(FatalError);
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
