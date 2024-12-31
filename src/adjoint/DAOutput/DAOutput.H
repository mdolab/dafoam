/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Assign the output array to the OF variables

\*---------------------------------------------------------------------------*/

#ifndef DAOutput_H
#define DAOutput_H

#include "runTimeSelectionTables.H"
#include "fvOptions.H"
#include "DAOption.H"
#include "DAModel.H"
#include "DAIndex.H"
#include "DAStateInfo.H"
#include "DAFunction.H"
#include "DAResidual.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                    Class DAOutput Declaration
\*---------------------------------------------------------------------------*/

class DAOutput
{

private:
    /// Disallow default bitwise copy construct
    DAOutput(const DAOutput&);

    /// Disallow default bitwise assignment
    void operator=(const DAOutput&);

protected:
    /// name of the output
    const word outputName_;

    /// type of the output
    const word outputType_;

    /// fvMesh
    fvMesh& mesh_;

    /// DAOption object
    const DAOption& daOption_;

    /// DAIndex object
    DAModel& daModel_;

    /// DAIndex object
    const DAIndex& daIndex_;

    DAResidual& daResidual_;

    UPtrList<DAFunction>& daFunctionList_;

    /// the stateInfo_ list from DAStateInfo object
    HashTable<wordList> stateInfo_;

public:
    /// Runtime type information
    TypeName("DAOutput");

    // Declare run-time constructor selection table
    declareRunTimeSelectionTable(
        autoPtr,
        DAOutput,
        dictionary,
        (
            const word outputName,
            const word outputType,
            fvMesh& mesh,
            const DAOption& daOption,
            DAModel& daModel,
            const DAIndex& daIndex,
            DAResidual& daResidual,
            UPtrList<DAFunction>& daFunctionList),
        (
            outputName,
            outputType,
            mesh,
            daOption,
            daModel,
            daIndex,
            daResidual,
            daFunctionList));

    // Constructors

    //- Construct from components
    DAOutput(
        const word outputName,
        const word outputType,
        fvMesh& mesh,
        const DAOption& daOption,
        DAModel& daModel,
        const DAIndex& daIndex,
        DAResidual& daResidual,
        UPtrList<DAFunction>& daFunctionList);

    // Selectors

    //- Return a reference to the selected model
    static autoPtr<DAOutput> New(
        const word outputName,
        const word outputType,
        fvMesh& mesh,
        const DAOption& daOption,
        DAModel& daModel,
        const DAIndex& daIndex,
        DAResidual& daResidual,
        UPtrList<DAFunction>& daFunctionList);

    //- Destructor
    virtual ~DAOutput()
    {
    }

    virtual void run(scalarList& output) = 0;

    virtual label size() = 0;

    virtual label distributed() = 0;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
