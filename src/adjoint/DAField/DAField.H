/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Field manipulation such as correct boundary conditions

\*---------------------------------------------------------------------------*/

#ifndef DAField_H
#define DAField_H

#include "fvOptions.H"
#include "surfaceFields.H"
#include "DAOption.H"
#include "DAUtility.H"
#include "DAStateInfo.H"
#include "DAModel.H"
#include "DAIndex.H"
#include "DAMacroFunctions.H"
#include "mixedFvPatchFields.H" // for setPrimalBoundaryCondition
#include "fixedGradientFvPatchField.H" // for setPrimalBoundaryCondition
#include "wordRe.H"
#include "wordRes.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DAField Declaration
\*---------------------------------------------------------------------------*/

class DAField
{

private:
    /// Disallow default bitwise copy construct
    DAField(const DAField&);

    /// Disallow default bitwise assignment
    void operator=(const DAField&);

protected:
    /// Foam::fvMesh object
    const fvMesh& mesh_;

    /// Foam::DAOption object
    const DAOption& daOption_;

    /// DAModel object
    const DAModel& daModel_;

    /// DAIndex object
    const DAIndex& daIndex_;

    /// the StateInfo_ list from DAStateInfo object
    HashTable<wordList> stateInfo_;

public:
    /// Constructors
    DAField(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    /// Destructor
    virtual ~DAField()
    {
    }

    // Members

    /// set the state vector based on the latest fields in OpenFOAM
    void ofField2StateVec(Vec stateVec) const;

    /// assign the fields in OpenFOAM based on the state vector
    void stateVec2OFField(const Vec stateVec) const;

    /// assign the fields in OpenFOAM based on the state array
    void state2OFField(const scalar* states) const;

    /// assign the openfoam fields to the states array
    void ofField2State(scalar* states) const;

    /// assign the points in fvMesh of OpenFOAM based on the point array
    void point2OFMesh(const scalar* volCoords) const;

    /// assign the point vector based on the points in fvMesh of OpenFOAM
    void ofMesh2PointVec(Vec xvVec) const;

    /// assign the residual vector based on the residual field in OpenFOAM
    void ofResField2ResVec(Vec resVec) const;

    /// set the boundary conditions based on parameters defined in DAOption
    void setPrimalBoundaryConditions(const label printInfo = 1);

    /// apply special treatment for boundary conditions
    void specialBCTreatment();

    /// check if we need to do special treatment for boundary conditions
    void checkSpecialBCs();

    /// a list that contains the names of detected special boundary conditions
    wordList specialBCs;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
