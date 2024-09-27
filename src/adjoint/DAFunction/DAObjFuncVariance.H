/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

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

    /// data mode options are probePoint surface or field
    word mode_;

    /// type of the variable either vector or scalar
    word varType_;

    /// component of the vector variable
    List<label> components_;

    /// cell index of the probe points
    labelList probeCellIndex_;

    /// coordinates of the probe points
    List<List<scalar>> probePointCoords_;

    /// names of the surfaces for ref data
    List<word> surfaceNames_;

    /// probe value
    List<List<scalar>> refValue_;

    /// total number of reference points
    label nRefPoints_ = 0;

    /// upper bound to be considered as non-data
    scalar varUpperBound_;

    /// whether we find the reference data
    label isRefData_;

    /// whether the ref data is time dependent if yes we need data in all time folders otherwise get it from the 0 folder
    label timeDependentRefData_;

#ifdef CompressibleFlow
    /// thermophysical property
    fluidThermo& thermo_;
#endif

    /// DATurbulenceModel object
    DATurbulenceModel& daTurb_;

    /// Cp used in incompressible heatFlux calculation
    scalar Cp_ = -9999.0;

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
