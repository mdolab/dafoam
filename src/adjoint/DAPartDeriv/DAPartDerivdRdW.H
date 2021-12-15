/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for dRdW

\*---------------------------------------------------------------------------*/

#ifndef DAPartDerivdRdW_H
#define DAPartDerivdRdW_H

#include "DAPartDeriv.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAPartDerivdRdW Declaration
\*---------------------------------------------------------------------------*/

class DAPartDerivdRdW
    : public DAPartDeriv
{

protected:
public:
    TypeName("dRdW");
    // Constructors

    //- Construct from components
    DAPartDerivdRdW(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAJacCon& daJacCon,
        const DAResidual& daResidual);

    //- Destructor
    virtual ~DAPartDerivdRdW()
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
