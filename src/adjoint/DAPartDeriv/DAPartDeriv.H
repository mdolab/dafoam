/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Compute partial derivatives using the finite-difference method
        with coloring

\*---------------------------------------------------------------------------*/

#ifndef DAPartDeriv_H
#define DAPartDeriv_H

#include "runTimeSelectionTables.H"
#include "fvOptions.H"
#include "DAUtility.H"
#include "DAOption.H"
#include "DAIndex.H"
#include "DAModel.H"
#include "DAStateInfo.H"
#include "syncTools.H"
#include "DAJacCon.H"
#include "DAResidual.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                    Class DAPartDeriv Declaration
\*---------------------------------------------------------------------------*/

class DAPartDeriv
{

private:
    /// Disallow default bitwise copy construct
    DAPartDeriv(const DAPartDeriv&);

    /// Disallow default bitwise assignment
    void operator=(const DAPartDeriv&);

protected:
    /// the name of the jacCon matrix
    const word modelType_;

    /// fvMesh
    const fvMesh& mesh_;

    /// DAOption object
    const DAOption& daOption_;

    /// DAModel object
    const DAModel& daModel_;

    /// DAIndex object
    const DAIndex& daIndex_;

    /// DAJacCon object
    const DAJacCon& daJacCon_;

    /// DAResidual object
    const DAResidual& daResidual_;

    /// all the DAFoam option
    const dictionary& allOptions_;

    /// the stateInfo_ list from DAStateInfo object
    HashTable<wordList> stateInfo_;

    /// perturb state variables given a color index
    void perturbStates(
        const Vec jacConColors,
        const Vec normStatePerturbVec,
        const label colorI,
        const scalar delta,
        Vec wVec);

    /// set values for the partial derivative matrix
    void setPartDerivMat(
        const Vec resVec,
        const Vec coloredColumn,
        const label transposed,
        Mat jacMat,
        const scalar jacLowerBound = 1e-30) const;

public:
    // Constructors
    DAPartDeriv(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAJacCon& daJacCon,
        const DAResidual& daResidual);

    //- Destructor
    virtual ~DAPartDeriv()
    {
    }

    // Member functions

    /// clear members in parent and child objects
    void clear()
    {
    }

    /// initialize partial derivative matrix
    void initializePartDerivMat(
        const dictionary& options,
        Mat jacMat);

    /// compute the partial derivative matrix
    void calcPartDerivMat(
        const dictionary& options,
        const Vec xvVec,
        const Vec wVec,
        Mat jacMat);

    /// setup the state normalization vector
    void setNormStatePerturbVec(Vec* normStatePerturbVec);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
