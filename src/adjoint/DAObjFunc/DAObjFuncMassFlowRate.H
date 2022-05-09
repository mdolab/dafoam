/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for mass flow rate

\*---------------------------------------------------------------------------*/

#ifndef DAObjFuncMassFlowRate_H
#define DAObjFuncMassFlowRate_H

#include "DAObjFunc.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAObjFuncMassFlowRate Declaration
\*---------------------------------------------------------------------------*/

class DAObjFuncMassFlowRate
    : public DAObjFunc
{

protected:

    /// DATurbulenceModel object
    DATurbulenceModel& daTurb_;

    /// density 
    volScalarField rho_;
    
    /// the area of all inlet patches
    scalar areaSum_ = -9999.0;

public:
    TypeName("massFlowRate");
    // Constructors

    //- Construct from components
    DAObjFuncMassFlowRate(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAResidual& daResidual,
        const word objFuncName,
        const word objFuncPart,
        const dictionary& objFuncDict);

    //- Destructor
    virtual ~DAObjFuncMassFlowRate()
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
