/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for dFdAOA

\*---------------------------------------------------------------------------*/

#ifndef DAPartDerivdFdAOA_H
#define DAPartDerivdFdAOA_H

#include "DAPartDeriv.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAPartDerivdFdAOA Declaration
\*---------------------------------------------------------------------------*/

class DAPartDerivdFdAOA
    : public DAPartDeriv
{

protected:
    /// the number of objFunc discrete elements for local procs
    label nLocalObjFuncGeoElements_;

public:
    TypeName("dFdAOA");
    // Constructors

    //- Construct from components
    DAPartDerivdFdAOA(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAJacCon& daJacCon,
        const DAResidual& daResidual);

    //- Destructor
    virtual ~DAPartDerivdFdAOA()
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
