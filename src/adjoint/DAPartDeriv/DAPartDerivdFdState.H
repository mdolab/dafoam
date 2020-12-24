/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for dFdState

\*---------------------------------------------------------------------------*/

#ifndef DAPartDerivdFdState_H
#define DAPartDerivdFdState_H

#include "DAPartDeriv.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAPartDerivdFdState Declaration
\*---------------------------------------------------------------------------*/

class DAPartDerivdFdState
    : public DAPartDeriv
{
public:
    TypeName("dFdState");
    // Constructors

    //- Construct from components
    DAPartDerivdFdState(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAJacCon& daJacCon,
        const DAResidual& daResidual);

    //- Destructor
    virtual ~DAPartDerivdFdState()
    {
    }

    // Member functions

    /// initialize partial derivative matrix
    virtual void initializePartDerivMat(
        const dictionary& options,
        Mat jacMat);

    /// compute the partial derivative matrix
    virtual void calcPartDerivMat(
        const dictionary& options,
        const Vec xvVec,
        const Vec wVec,
        Mat jacMat);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
