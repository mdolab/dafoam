/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for uniform pressure gradient

\*---------------------------------------------------------------------------*/

#ifndef DAFvSourceUniformPressureGradient_H
#define DAFvSourceUniformPressureGradient_H

#include "DAFvSource.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAFvSourceUniformPressureGradient Declaration
\*---------------------------------------------------------------------------*/

class DAFvSourceUniformPressureGradient
    : public DAFvSource
{

protected:
public:
    TypeName("uniformPressureGradient");
    // Constructors

    //- Construct from components
    DAFvSourceUniformPressureGradient(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAFvSourceUniformPressureGradient()
    {
    }

    /// compute the FvSource term
    virtual void calcFvSource(volVectorField& fvSource);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
