/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for field variable input that change for each time step 
        (unsteady)

\*---------------------------------------------------------------------------*/

#ifndef DAInputFieldUnsteady_H
#define DAInputFieldUnsteady_H

#include "DAInput.H"
#include "addToRunTimeSelectionTable.H"
#include "mixedFvPatchFields.H"
#include "DAGlobalVar.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAInputFieldUnsteady Declaration
\*---------------------------------------------------------------------------*/

class DAInputFieldUnsteady
    : public DAInput
{

protected:
    /// name of the field
    word fieldName_;

    /// field type, can be either scalar or vector
    word fieldType_;

    /// how many time step interval to use, 1 means updating the field every time step
    label stepInterval_ = 1;

    /// how many parameters to interpolate along the time, they could be the field it self
    label nParameters_ = -1;

    /// interpolation method
    word interpolationMethod_ = "linear";

public:
    TypeName("fieldUnsteady");
    // Constructors

    //- Construct from components
    DAInputFieldUnsteady(
        const word inputName,
        const word inputType,
        fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAInputFieldUnsteady()
    {
    }

    virtual void run(const scalarList& input);

    virtual label size()
    {
        if (fieldType_ == "scalar")
        {
            if (this->distributed())
            {
                return daIndex_.nLocalCells * nParameters_;
            }
            else
            {
                return daIndex_.nGlobalCells * nParameters_;
            }
        }
        else
        {
            FatalErrorIn("DAInputFieldUnsteady::size") << "fieldType not valid" << exit(FatalError);
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
