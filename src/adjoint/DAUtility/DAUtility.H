/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        DAUtility contains basic functions such as matrix file IO, and
        it is independent of fvMesh. All the functions in DAUtility are static,
        so they can be directly called, e.g., DAUtility::boundVar(...)

\*---------------------------------------------------------------------------*/

#ifndef DAUtility_H
#define DAUtility_H

#include <petscksp.h>
#include "Python.h"
#include "fvOptions.H"
#include "globalIndex.H"
#include "IOMRFZoneListDF.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

typedef void (*pyComputeInterface)(const double*, int, double*, int, void*);
typedef void (*pyJacVecProdInterface)(const double*, double*, int, const double*, const double*, int, void*);
typedef void (*pySetCharInterface)(const char*, void*);

/*---------------------------------------------------------------------------*\
                       Class DAUtility Declaration
\*---------------------------------------------------------------------------*/
class DAUtility
{

private:
    /// Disallow default bitwise copy construct
    DAUtility(const DAUtility&);

    /// Disallow default bitwise assignment
    void operator=(const DAUtility&);

public:
    /// Constructors
    DAUtility();

    /// Destructor
    virtual ~DAUtility();

    /// convert a python dictionary object to OpenFoam dictionary
    static void pyDict2OFDict(
        PyObject* pyDict,
        dictionary& ofDict);

    /// delete a value in the list
    template<class classType>
    static label listDeleteVal(
        List<classType>& listIn,
        const classType valDel);

    /// write petsc matrix in binary format
    static void writeMatrixBinary(
        const Mat matIn,
        const word prefix);

    /// write petsc matrix in ascii format
    static void writeMatrixASCII(
        const Mat matIn,
        const word prefix);

    /// read petsc matrix in binary format
    static void readMatrixBinary(
        Mat matIn,
        const word prefix);

    /// write petsc vector in ascii format
    static void writeVectorASCII(
        const Vec vecIn,
        const word prefix);

    /// read petsc vector in binary format
    static void readVectorBinary(
        Vec vecIn,
        const word prefix);

    /// write petsc vector in binary format
    static void writeVectorBinary(
        const Vec vecIn,
        const word prefix);

    /// bound a volScalar variable based on parametes defined in DAOption::allOptions_
    static void boundVar(
        const dictionary& allOptions,
        volScalarField& var,
        const label printToScreen);

    /// bound a volVector variable based on parametes defined in DAOption::allOptions_
    static void boundVar(
        const dictionary& allOptions,
        volVectorField& var,
        const label printToScreen);

    /// check whether a value is close to a reference value by a tolerance
    static label isValueCloseToRef(
        const scalar val,
        const scalar refVal,
        const scalar tol = 1.0e-6);

    /// generate global index numbering for local-global index transferring
    static globalIndex genGlobalIndex(const label localIndexSize);

    /// angle of attack in radian used in forward mode AD
    static scalar angleOfAttackRadForwardAD;

    /// define a function pointer template for Python call back
    static void* pyCalcBeta;
    static pyComputeInterface pyCalcBetaInterface;

    static void* pyCalcBetaJacVecProd;
    static pyJacVecProdInterface pyCalcBetaJacVecProdInterface;

    static void* pySetModelName;
    static pySetCharInterface pySetModelNameInterface;

    /// control when to print the residual and also compute the maxInitRes
    static void primalResidualControl(
        const SolverPerformance<scalar>& solverP,
        const label printToScreen,
        const word varName,
        scalar& primalMaxRes);

    /// control when to print the residual and also compute the maxInitRes
    static void primalResidualControl(
        const SolverPerformance<vector>& solverP,
        const label printToScreen,
        const word varName,
        scalar& primalMaxRes);

    static label myFindCell(
        const primitiveMesh& mesh,
        const point& point);

    static label isFieldReadable(
        const fvMesh& mesh,
        const word fieldName,
        const word fieldType);

    /// swap two lists
    template<class classType>
    static void swapLists(List<classType>& a, List<classType>& b);
};

template<class classType>
label DAUtility::listDeleteVal(
    List<classType>& listIn,
    const classType valDel)
{
    /*
    Description:
        Delete a value in the list, the sequence of other
        elements will be preserved

    Input:
        listIn: the list to delete value
        valDel: value to delete
    
    Output:
        foundVal: 1 means the requested value is found and deleted

    Example:
        List<word> listIn={"apple","orange"};
        listDeleteVal<word>(listIn,"apple");
        Now listIn will be {orange"}
        NOTE: if list has multiple val, they will be all deleted
    */

    // we will create a new list and delete the value
    List<classType> listNew;

    label foundVal = 0;
    forAll(listIn, idxI)
    {
        const classType& val = listIn[idxI];
        if (val == valDel)
        {
            foundVal = 1;
            // do not append
        }
        else
        {
            listNew.append(val);
        }
    }

    listIn.clear();
    listIn = listNew;

    return foundVal;
}

/// swap two lists
template<class classType>
void DAUtility::swapLists(List<classType>& a, List<classType>& b)
{
    List<classType> temp = a;
    a = b;
    b = temp;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
