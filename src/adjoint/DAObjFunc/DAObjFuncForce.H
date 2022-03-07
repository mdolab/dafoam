/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for force 

\*---------------------------------------------------------------------------*/

#ifndef DAObjFuncForce_H
#define DAObjFuncForce_H

#include "DAObjFunc.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAObjFuncForce Declaration
\*---------------------------------------------------------------------------*/

class DAObjFuncForce
    : public DAObjFunc
{

protected:
    /// the direction of the force
    vector forceDir_;

    /// if dynamically adjusting the angle what mode to use
    word dirMode_;

    /// the reference patch name from the alpha design variable dict patches[0] from which we compute the flow direction
    word inoutRefPatchName_;

    /// flowAxisIndex_ for the alpha design variable with tan(U_normal/U_flow)
    label flowAxisIndex_;

    /// normalAxisIndex_ for the alpha design variable with tan(U_normal/U_flow)
    label normalAxisIndex_;

    /// dynamically adjust the force direction
    void updateForceDir(vector& forceDir);

    /// DATurbulenceModel object
    const DATurbulenceModel& daTurb_;

public:
    TypeName("force");
    // Constructors

    //- Construct from components
    DAObjFuncForce(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAResidual& daResidual,
        const word objFuncName,
        const word objFuncPart,
        const dictionary& objFuncDict);

    //- Destructor
    virtual ~DAObjFuncForce()
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
