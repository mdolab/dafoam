/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for residual output

\*---------------------------------------------------------------------------*/

#ifndef DAOutputFunction_H
#define DAOutputFunction_H

#include "DAOutput.H"
#include "addToRunTimeSelectionTable.H"
#include "mixedFvPatchFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAOutputFunction Declaration
\*---------------------------------------------------------------------------*/

class DAOutputFunction
    : public DAOutput
{

protected:
public:
    TypeName("function");
    // Constructors

    //- Construct from components
    DAOutputFunction(
        const word outputName,
        const word outputType,
        fvMesh& mesh,
        const DAOption& daOption,
        DAModel& daModel,
        const DAIndex& daIndex,
        DAResidual& daResidual,
        UPtrList<DAFunction>& daFunctionList);

    //- Destructor
    virtual ~DAOutputFunction()
    {
    }

    /// return the index of a give functionName in daFunctionList
    label getFunctionListIndex(
        const word functionName,
        UPtrList<DAFunction>& daFunctionList)
    {
        forAll(daFunctionList, idxI)
        {
            DAFunction& daFunction = daFunctionList[idxI];
            word functionName1 = daFunction.getFunctionName();
            if (functionName1 == functionName)
            {
                return idxI;
            }
        }
        return -9999;
    }

    virtual void run(scalarList& output);

    virtual label size()
    {
        return 1;
    }

    virtual label distributed()
    {
        return 0;
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
