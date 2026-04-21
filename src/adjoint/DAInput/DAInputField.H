/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v5

    Description:
        Child class for field variable input

\*---------------------------------------------------------------------------*/

#ifndef DAInputField_H
#define DAInputField_H

#include "DAInput.H"
#include "addToRunTimeSelectionTable.H"
#include "mixedFvPatchFields.H"
#include "DAGlobalVar.H"
#include "cellSet.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAInputField Declaration
\*---------------------------------------------------------------------------*/

class DAInputField
    : public DAInput
{

protected:
    /// name of the field
    word fieldName_;

    /// field type, can be either scalar or vector
    word fieldType_;

    /// if it is vector fields, we can also provide components/indices
    labelList indices_;

    /// optional cellSet name used to restrict the input support
    word cellSetName_;

    /// locally owned cells that this input updates
    labelList selectedCells_;

    /// local cell count in the selected set
    label localSelectedCells_ = 0;

    /// global cell count in the selected set
    label globalSelectedCells_ = 0;

    /// indexing system to convert between local and global selected-cell indices
    globalIndex globalSelectedCellNumbering_;

public:
    TypeName("field");
    // Constructors

    //- Construct from components
    DAInputField(
        const word inputName,
        const word inputType,
        fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAInputField()
    {
    }

    virtual void run(const scalarList& input);

    virtual label size()
    {
        if (fieldType_ == "scalar")
        {
            if (this->distributed())
            {
                return localSelectedCells_;
            }
            else
            {
                return globalSelectedCells_;
            }
        }
        else if (fieldType_ == "vector")
        {
            label cSize = indices_.size();
            if (this->distributed())
            {
                return localSelectedCells_ * cSize;
            }
            else
            {
                return globalSelectedCells_ * cSize;
            }
        }
        else
        {
            FatalErrorIn("DAInputField::size") << "fieldType not valid" << exit(FatalError);
            return -1;
        }
    }

    virtual label distributed()
    {
        label distributed = daOption_.getAllOptions().subDict("inputInfo").subDict(inputName_).getLabel("distributed");
        return distributed;
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
