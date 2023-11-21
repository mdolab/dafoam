/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for variable volume sum

\*---------------------------------------------------------------------------*/

#ifndef DAObjFuncVariance_H
#define DAObjFuncVariance_H

#include "DAObjFunc.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAObjFuncVariance Declaration
\*---------------------------------------------------------------------------*/

class DAObjFuncVariance
    : public DAObjFunc
{

protected:

    /// name of the variable
    word varName_;

    /// type of the variable either vector or scalar
    word varType_;

    /// cell index of the probe points
    labelList refCellIndex_;

    /// component for vector at the above probe points cellI
    labelList refCellComp_;
 
    /// probe value
    List<List<scalar>> refValue_;

    /// number of reference points
    label nRefPoints_;

    /// the global reference points in parallel
    label nRefPointsGlobal_;

    /// upper bound to be considered as non-data
    scalar varUpperBound_;

public:
    TypeName("variance");
    // Constructors

    //- Construct from components
    DAObjFuncVariance(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAResidual& daResidual,
        const word objFuncName,
        const word objFuncPart,
        const dictionary& objFuncDict);

    //- Destructor
    virtual ~DAObjFuncVariance()
    {
    }

    /// calculate the value of objective function
    virtual void calcObjFunc(
        const labelList& objFuncFaceSources,
        const labelList& objFuncCellSources,
        scalarList& objFuncFaceValues,
        scalarList& objFuncCellValues,
        scalar& objFuncValue);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
