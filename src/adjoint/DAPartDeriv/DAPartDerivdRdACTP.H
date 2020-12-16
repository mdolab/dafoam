/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for dRdACTP

\*---------------------------------------------------------------------------*/

#ifndef DAPartDerivdRdACTP_H
#define DAPartDerivdRdACTP_H

#include "DAPartDeriv.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAPartDerivdRdACTP Declaration
\*---------------------------------------------------------------------------*/

class DAPartDerivdRdACTP
    : public DAPartDeriv
{

protected:

    /// number of design variables for actuator points
    label nActDVs_ = 9;

public:
    TypeName("dRdACTP");
    // Constructors

    //- Construct from components
    DAPartDerivdRdACTP(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAJacCon& daJacCon,
        const DAResidual& daResidual);

    //- Destructor
    virtual ~DAPartDerivdRdACTP()
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
