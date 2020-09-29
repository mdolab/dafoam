/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for dFdACTP

\*---------------------------------------------------------------------------*/

#ifndef DAPartDerivdFdACTP_H
#define DAPartDerivdFdACTP_H

#include "DAPartDeriv.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAPartDerivdFdACTP Declaration
\*---------------------------------------------------------------------------*/

class DAPartDerivdFdACTP
    : public DAPartDeriv
{

protected:
    /// the number of objFunc discrete elements for local procs
    label nLocalObjFuncGeoElements_;

    /// number of design variables for actuator points
    label nActPointDVs_ = 9;

public:
    TypeName("dFdACTP");
    // Constructors

    //- Construct from components
    DAPartDerivdFdACTP(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAJacCon& daJacCon,
        const DAResidual& daResidual);

    //- Destructor
    virtual ~DAPartDerivdFdACTP()
    {
    }

    // Member functions

    /// initialize partial derivative matrix
    virtual void initializePartDerivMat(
        const dictionary& options,
        Mat* jacMat);

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
