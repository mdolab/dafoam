/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Compute objective and constraint functions and their derivatives

\*---------------------------------------------------------------------------*/

#ifndef DAFunction_H
#define DAFunction_H

#include "runTimeSelectionTables.H"
#include "fvOptions.H"
#include "DAOption.H"
#include "DAModel.H"
#include "DAIndex.H"
#include "topoSetSource.H"
#include "topoSet.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                    Class DAFunction Declaration
\*---------------------------------------------------------------------------*/

class DAFunction
{

private:
    /// Disallow default bitwise copy construct
    DAFunction(const DAFunction&);

    /// Disallow default bitwise assignment
    void operator=(const DAFunction&);

protected:
    /// fvMesh
    const fvMesh& mesh_;

    /// DAOption object
    const DAOption& daOption_;

    /// DAModel object
    const DAModel& daModel_;

    /// DAIndex object
    const DAIndex& daIndex_;

    /// the name of the objective function
    word functionName_;

    /// the part of the objective function
    word functionPart_;

    /// the type of the objective function
    word functionType_;

    /// dictionary containing the information for the objective function
    const dictionary& functionDict_;

    /// a sorted list of all face sources for the objective function
    labelList faceSources_;

    /// a sorted list of all cell sources for the objective function
    labelList cellSources_;

    /// scale of the objective function
    scalar scale_;

    /// the value of the objective function
    scalar functionValue_;

    /// whether to calculate (obj-ref)^2
    label calcRefVar_;

    /// if calcRefVar_ is True, set the reference value list
    scalarList ref_;

public:
    /// Runtime type information
    TypeName("DAFunction");

    // Declare run-time constructor selection table
    declareRunTimeSelectionTable(
        autoPtr,
        DAFunction,
        dictionary,
        (
            const fvMesh& mesh,
            const DAOption& daOption,
            const DAModel& daModel,
            const DAIndex& daIndex,
            const word functionName,
            const word functionPart,
            const dictionary& functionDict),
        (
            mesh,
            daOption,
            daModel,
            daIndex,
            functionName,
            functionPart,
            functionDict));

    // Constructors

    //- Construct from components
    DAFunction(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const word functionName,
        const word functionPart,
        const dictionary& functionDict);

    // Selectors

    //- Return a reference to the selected model
    static autoPtr<DAFunction> New(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const word functionName,
        const word functionPart,
        const dictionary& functionDict);

    //- Destructor
    virtual ~DAFunction()
    {
    }

    /// clear up members
    void clear()
    {
        faceSources_.clear();
        cellSources_.clear();
    }

    /// calculate DAFunction::faceSources_ and DAFunction::cellSources_
    void calcFunctionSources();

    /// calculate the value of objective function
    virtual void calcFunction(scalar& functionValue) = 0;

    /// calculate the value of objective function
    scalar getFunctionValue();

    /// return the name of objective function
    word getFunctionName()
    {
        return functionName_;
    }

    /// return the part of objective function
    word getFunctionPart()
    {
        return functionPart_;
    }

    /// return the part of objective function
    word getFunctionType()
    {
        return functionType_;
    }

    /// return DAFunction::faceSources_
    const labelList& getFaceSources() const
    {
        return faceSources_;
    }

    /// return DAFunction::cellSources_
    const labelList& getCellSources() const
    {
        return cellSources_;
    }

    /// whether to compute reference coefficients for special function treatment such as totalPressureRatio
    label calcRefCoeffs = 1;

    /// calculate (var-ref)^2
    void calcRefVar(scalar& functionValue);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
