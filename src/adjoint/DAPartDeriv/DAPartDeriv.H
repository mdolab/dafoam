/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

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
#include "DAObjFunc.H"
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

    /// perturb the values in the boundary condition
    void perturbBC(
        const dictionary options,
        const scalar delta);

    /// set values for the partial derivative matrix
    void setPartDerivMat(
        const Vec resVec,
        const Vec coloredColumn,
        const label transposed,
        Mat jacMat,
        const scalar jacLowerBound=1e-30) const;
    
    /// perturb the angle of attack
    void perturbAOA(
        const dictionary options,
        const scalar delta);

    /// volume mesh coordinates wrt the ffd point coordinate partials
    Mat dXvdFFDMat_;

public:
    /// Runtime type information
    TypeName("DAPartDeriv");

    // Declare run-time constructor selection table
    declareRunTimeSelectionTable(
        autoPtr,
        DAPartDeriv,
        dictionary,
        (const word modelType,
         const fvMesh& mesh,
         const DAOption& daOption,
         const DAModel& daModel,
         const DAIndex& daIndex,
         const DAJacCon& daJacCon,
         const DAResidual& daResidual),
        (modelType,
         mesh,
         daOption,
         daModel,
         daIndex,
         daJacCon,
         daResidual));

    // Constructors

    //- Construct from components
    DAPartDeriv(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAJacCon& daJacCon,
        const DAResidual& daResidual);

    // Selectors

    //- Return a reference to the selected model
    static autoPtr<DAPartDeriv> New(
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
        MatDestroy(&dXvdFFDMat_);
    }

    /// initialize partial derivative matrix
    virtual void initializePartDerivMat(
        const dictionary& options,
        Mat jacMat) = 0;

    /// compute the partial derivative matrix
    virtual void calcPartDerivMat(
        const dictionary& options,
        const Vec xvVec,
        const Vec wVec,
        Mat jacMat) = 0;

    /// setup dXvdFFD matrix
    void setdXvdFFDMat(const Mat dXvdFFDMat);

    /// setup the state normalization vector
    void setNormStatePerturbVec(Vec* normStatePerturbVec);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
