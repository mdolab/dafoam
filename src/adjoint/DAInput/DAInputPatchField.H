/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for variables at a selected patch

\*---------------------------------------------------------------------------*/

#ifndef DAInputPatchField_H
#define DAInputPatchField_H

#include "DAInput.H"
#include "addToRunTimeSelectionTable.H"
#include "mixedFvPatchFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAInputPatchField Declaration
\*---------------------------------------------------------------------------*/

class DAInputPatchField
    : public DAInput
{

protected:
    /// the name of the variable
    word fieldName_;

    /// the type of the variable can be either scalar or vector
    word fieldType_;

    /// the patch to assign values to
    wordList patches_;

    /// if it is vector fields, we can also provide components/indices
    labelList indices_;

    /// local face size for all selected patches
    label localPatchFaces_ = 0;

    /// global face size for all selected patches
    label globalPatchFaces_ = 0;

    /// indexing system to convert between global and local indices
    globalIndex globalPatchFaceNumbering_;

public:
    TypeName("patchField");
    // Constructors

    //- Construct from components
    DAInputPatchField(
        const word inputName,
        const word inputType,
        fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAInputPatchField()
    {
    }

    virtual void run(const scalarList& input);

    virtual label size()
    {
        if (fieldType_ == "scalar")
        {
            if (this->distributed())
            {
                return localPatchFaces_;
            }
            else
            {
                return globalPatchFaces_;
            }
        }
        else if (fieldType_ == "vector")
        {
            label cSize = indices_.size();
            if (this->distributed())
            {
                return localPatchFaces_ * cSize;
            }
            else
            {
                return globalPatchFaces_ * cSize;
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
