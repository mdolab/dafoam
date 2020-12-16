/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for dRdFFD

\*---------------------------------------------------------------------------*/

#ifndef DAPartDerivdRdFFD_H
#define DAPartDerivdRdFFD_H

#include "DAPartDeriv.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAPartDerivdRdFFD Declaration
\*---------------------------------------------------------------------------*/

class DAPartDerivdRdFFD
    : public DAPartDeriv
{

protected:
public:
    TypeName("dRdFFD");
    // Constructors

    //- Construct from components
    DAPartDerivdRdFFD(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAJacCon& daJacCon,
        const DAResidual& daResidual);

    //- Destructor
    virtual ~DAPartDerivdRdFFD()
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
