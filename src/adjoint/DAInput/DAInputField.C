/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

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
    fieldName_ = daOption_.getAllOptions().subDict("inputInfo").subDict(inputName).getWord("fieldName");
    fieldType_ = daOption_.getAllOptions().subDict("inputInfo").subDict(inputName).getWord("fieldType");

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
            forAll(field, cellI)
            {
                field[cellI] = input[cellI];
            }
        }
        else
        {
            for (label globalCellI = 0; globalCellI < daIndex_.nGlobalCells; globalCellI++)
            {
                if (daIndex_.globalCellNumbering.isLocal(globalCellI))
                {
                    label localCellI = daIndex_.globalCellNumbering.toLocal(globalCellI);
                    field[localCellI] = input[globalCellI];
                }
            }
        }
        field.correctBoundaryConditions();
    }
    else if (fieldType_ == "vector")
    {
        volVectorField& field = const_cast<volVectorField&>(mesh_.thisDb().lookupObject<volVectorField>(fieldName_));
        if (this->distributed())
        {
            label counterI = 0;
            forAll(field, cellI)
            {
                forAll(indices_, idxI)
                {
                    label comp = indices_[idxI];
                    field[cellI][comp] = input[counterI];
                    counterI++;
                }
            }
        }
        else
        {
            for (label globalCellI = 0; globalCellI < daIndex_.nGlobalCells; globalCellI++)
            {
                if (daIndex_.globalCellNumbering.isLocal(globalCellI))
                {
                    label localCellI = daIndex_.globalCellNumbering.toLocal(globalCellI);
                    forAll(indices_, idxI)
                    {
                        label comp = indices_[idxI];
                        label inputIdx = globalCellI * 3 + comp;
                        field[localCellI][comp] = input[inputIdx];
                    }
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
