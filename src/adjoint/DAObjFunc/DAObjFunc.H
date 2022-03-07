/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Compute object functions and their derivatives

\*---------------------------------------------------------------------------*/

#ifndef DAObjFunc_H
#define DAObjFunc_H

#include "runTimeSelectionTables.H"
#include "fvOptions.H"
#include "DAOption.H"
#include "DAModel.H"
#include "DAIndex.H"
#include "topoSetSource.H"
#include "topoSet.H"
#include "DAField.H"
#include "DAResidual.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                    Class DAObjFunc Declaration
\*---------------------------------------------------------------------------*/

class DAObjFunc
{

private:
    /// Disallow default bitwise copy construct
    DAObjFunc(const DAObjFunc&);

    /// Disallow default bitwise assignment
    void operator=(const DAObjFunc&);

protected:
    /// fvMesh
    const fvMesh& mesh_;

    /// DAOption object
    const DAOption& daOption_;

    /// DAModel object
    const DAModel& daModel_;

    /// DAIndex object
    const DAIndex& daIndex_;

    /// DAResidual object
    const DAResidual& daResidual_;

    /// the name of the objective function
    word objFuncName_;

    /// the part of the objective function
    word objFuncPart_;

    /// the type of the objective function
    word objFuncType_;

    /// dictionary containing the information for the objective function
    const dictionary& objFuncDict_;

    /// DAField object
    DAField daField_;

    /// a sorted list of all face sources for the objective function
    labelList objFuncFaceSources_;

    /// a sorted list of all cell sources for the objective function
    labelList objFuncCellSources_;

    /// value of the obj function on the discrete faces
    scalarList objFuncFaceValues_;

    /// value of the obj function in the discrete cells
    scalarList objFuncCellValues_;

    /// scale of the objective function
    scalar scale_;

    /// the value of the objective function
    scalar objFuncValue_;

    /// the connectivity information for the objective function
    List<List<word>> objFuncConInfo_;

public:
    /// Runtime type information
    TypeName("DAObjFunc");

    // Declare run-time constructor selection table
    declareRunTimeSelectionTable(
        autoPtr,
        DAObjFunc,
        dictionary,
        (
            const fvMesh& mesh,
            const DAOption& daOption,
            const DAModel& daModel,
            const DAIndex& daIndex,
            const DAResidual& daResidual,
            const word objFuncName,
            const word objFuncPart,
            const dictionary& objFuncDict),
        (
            mesh,
            daOption,
            daModel,
            daIndex,
            daResidual,
            objFuncName,
            objFuncPart,
            objFuncDict));

    // Constructors

    //- Construct from components
    DAObjFunc(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAResidual& daResidual,
        const word objFuncName,
        const word objFuncPart,
        const dictionary& objFuncDict);

    // Selectors

    //- Return a reference to the selected model
    static autoPtr<DAObjFunc> New(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const DAResidual& daResidual,
        const word objFuncName,
        const word objFuncPart,
        const dictionary& objFuncDict);

    //- Destructor
    virtual ~DAObjFunc()
    {
    }

    /// clear up members
    void clear()
    {
        objFuncFaceSources_.clear();
        objFuncCellSources_.clear();
        objFuncFaceValues_.clear();
        objFuncCellValues_.clear();
        objFuncConInfo_.clear();
    }

    /// calculate DAObjFunc::objFuncFaceSources_ and DAObjFunc::objFuncCellSources_
    void calcObjFuncSources(
        labelList& faceSources,
        labelList& cellSources);

    /// calculate the value of objective function
    virtual void calcObjFunc(
        const labelList& objFuncFaceSources,
        const labelList& objFuncCellSources,
        scalarList& objFuncFaceValues,
        scalarList& objFuncCellValues,
        scalar& objFuncValue) = 0;

    /// calculate the value of objective function
    scalar getObjFuncValue();

    /// the master function to compute objective function given the state and point vectors
    scalar masterFunction(
        const dictionary& options,
        const Vec xvVec,
        const Vec wVec);

    /// return the name of objective function
    word getObjFuncName()
    {
        return objFuncName_;
    }

    /// return the part of objective function
    word getObjFuncPart()
    {
        return objFuncPart_;
    }

    /// return the part of objective function
    word getObjFuncType()
    {
        return objFuncType_;
    }

    /// return DAObjFunc::objFuncFaceSources_
    const labelList& getObjFuncFaceSources() const
    {
        return objFuncFaceSources_;
    }

    /// return DAObjFunc::objFuncCellSources_
    const labelList& getObjFuncCellSources() const
    {
        return objFuncCellSources_;
    }

    /// return DAObjFunc::objFuncFaceValues_
    const scalarList& getObjFuncFaceValues() const
    {
        return objFuncFaceValues_;
    }

    /// return DAObjFunc::objFuncCellValues_
    const scalarList& getObjFuncCellValues() const
    {
        return objFuncCellValues_;
    }

    /// return DAObjFunc::objFuncConInfo_
    const List<List<word>>& getObjFuncConInfo() const
    {
        return objFuncConInfo_;
    }

    /// expSumKS stores sum[exp(coeffKS*x_i)] for KS function which will be used to scale dFdW
    scalar expSumKS = 1.0;

    /// whether to compute reference coefficients for special objFunc treatment such as totalPressureRatio
    label calcRefCoeffs = 1;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
