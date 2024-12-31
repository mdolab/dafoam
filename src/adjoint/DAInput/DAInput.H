/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Assign the input array to the OF variables

\*---------------------------------------------------------------------------*/

#ifndef DAInput_H
#define DAInput_H

#include "runTimeSelectionTables.H"
#include "fvOptions.H"
#include "DAOption.H"
#include "DAModel.H"
#include "DAIndex.H"
#include "DAStateInfo.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                    Class DAInput Declaration
\*---------------------------------------------------------------------------*/

class DAInput
{

private:
    /// Disallow default bitwise copy construct
    DAInput(const DAInput&);

    /// Disallow default bitwise assignment
    void operator=(const DAInput&);

protected:
    /// name of the input
    const word inputName_;

    /// type of the input
    const word inputType_;

    /// fvMesh
    fvMesh& mesh_;

    /// DAOption object
    const DAOption& daOption_;

    /// DAIndex object
    const DAModel& daModel_;

    /// DAIndex object
    const DAIndex& daIndex_;

    /// the stateInfo_ list from DAStateInfo object
    HashTable<wordList> stateInfo_;

public:
    /// Runtime type information
    TypeName("DAInput");

    // Declare run-time constructor selection table
    declareRunTimeSelectionTable(
        autoPtr,
        DAInput,
        dictionary,
        (
            const word inputName,
            const word inputType,
            fvMesh& mesh,
            const DAOption& daOption,
            const DAModel& daModel,
            const DAIndex& daIndex),
        (
            inputName,
            inputType,
            mesh,
            daOption,
            daModel,
            daIndex));

    // Constructors

    //- Construct from components
    DAInput(
        const word inputName,
        const word inputType,
        fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    // Selectors

    //- Return a reference to the selected model
    static autoPtr<DAInput> New(
        const word inputName,
        const word inputType,
        fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAInput()
    {
    }

    virtual void run(const scalarList& input) = 0;

    virtual label size() = 0;

    virtual label distributed() = 0;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
