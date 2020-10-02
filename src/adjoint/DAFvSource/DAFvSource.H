/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Compute the source term (e.g., actuator disk) for the primal 
        governing equations

\*---------------------------------------------------------------------------*/

#ifndef DAFvSource_H
#define DAFvSource_H

#include "runTimeSelectionTables.H"
#include "fvOptions.H"
#include "surfaceFields.H"
#include "DAOption.H"
#include "DAModel.H"
#include "DAIndex.H"
#include "topoSetSource.H"
#include "topoSet.H"
#include "regIOobject.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                    Class DAFvSource Declaration
\*---------------------------------------------------------------------------*/

class DAFvSource
:
    public regIOobject
{

private:
    /// Disallow default bitwise copy construct
    DAFvSource(const DAFvSource&);

    /// Disallow default bitwise assignment
    void operator=(const DAFvSource&);

protected:

    /// model name
    const word& modelType_;
    
    /// fvMesh
    const fvMesh& mesh_;

    /// DAOption object
    const DAOption& daOption_;

    /// DAModel object
    const DAModel& daModel_;

    /// DAIndex object
    const DAIndex& daIndex_;

public:
    /// Runtime type information
    TypeName("DAFvSource");

    // Declare run-time constructor selection table
    declareRunTimeSelectionTable(
        autoPtr,
        DAFvSource,
        dictionary,
        (const word modelType,
         const fvMesh& mesh,
         const DAOption& daOption,
         const DAModel& daModel,
         const DAIndex& daIndex),
        (modelType, mesh, daOption, daModel, daIndex));

    // Constructors

    //- Construct from components
    DAFvSource(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    // Selectors

    //- Return a reference to the selected model
    static autoPtr<DAFvSource> New(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAFvSource()
    {
    }

    /// compute the FvSource term
    virtual void calcFvSource(volVectorField& fvSource);

    /// virtual function for regIOobject
    bool writeData(Ostream& os) const;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
