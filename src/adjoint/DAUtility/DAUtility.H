/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

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

    /// check whether a value is in the list
    template<class classType>
    static label isInList(
        const classType value,
        const List<classType>& list);

    /// replace a value in the list
    template<class classType>
    static label listReplaceVal(
        List<classType>& listIn,
        const classType valOrig,
        const classType valNew);

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
};

template<class classType>
label DAUtility::isInList(
    const classType value,
    const List<classType>& list)
{
    /*
    Description:
        Check whether a value is in the list

    Input:
        value: a value to check with type: classType
        list: list to check the value

    Example:
        If the list reads
        list1={"test","run","compare"};
        label val=isInList<word>("run",list1);
        Here the return value: val==1 (value found in the list)
    */

    forAll(list, idxI)
    {
        if (list[idxI] == value)
        {
            return 1;
        }
    }
    return 0;
}

template<class classType>
label DAUtility::listReplaceVal(
    List<classType>& listIn,
    const classType valOrig,
    const classType valNew)
{
    /*
    Description:
        Replace a value in hte list, the list sequence is preserved

    Input:
        listIn: the list to replace
        valOrig: original value in the list
        valNew: new value to replace

    Example:
        List<word> listIn={"apple","orange"};
        listReplaceVal<word>(listIn,"apple","banana");
        Now listIn will be {"banana","orange"}
        NOTE: if list has multiple valOrig, they will be all replaced
    */

    label foundVal = 0;
    forAll(listIn, idxI)
    {
        const classType& val = listIn[idxI];
        if (val == valOrig)
        {
            listIn[idxI] = valNew;
            foundVal = 1;
        }
    }

    return foundVal;
}

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

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
