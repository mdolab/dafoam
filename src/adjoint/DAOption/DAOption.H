/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        DAOption contains an OpenFOAM dictionary that has all the options 
        defined in the python layer. It also has functions to get and set
        values from DAOption::allOptions_

\*---------------------------------------------------------------------------*/

#ifndef DAOption_H
#define DAOption_H

#include "fvOptions.H"
#include "Python.h"
#include "DAUtility.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DAOption Declaration
\*---------------------------------------------------------------------------*/

class DAOption
    : public regIOobject
{

private:
    /// Disallow default bitwise copy construct
    DAOption(const DAOption&);

    /// Disallow default bitwise assignment
    void operator=(const DAOption&);

    /// Foam::fvMesh object
    const fvMesh& mesh_;

    /// all the options defined
    dictionary allOptions_;

public:
    /// Constructors
    DAOption(
        const fvMesh& mesh,
        PyObject* pyOptions);

    /// Destructor
    virtual ~DAOption();

    /// return a reference of allOptions_ dictionary
    const dictionary& getAllOptions() const
    {
        return allOptions_;
    }

    /// get an option from subDict and key
    template<class classType>
    classType getOption(const word key) const;

    /// set an option to subDict and key
    template<class classType>
    void setOption(
        const word key,
        const classType value);

    /// get an dictionary option from subDict and key
    template<class classType>
    classType getSubDictOption(
        const word subDict,
        const word subDictKey) const;

    /// set an dictionary option to subDict and key
    template<class classType>
    void setSubDictOption(
        const word subDict,
        const word subDictKey,
        const classType value);

    /// this is a virtual function for regIOobject
    bool writeData(Ostream& os) const;

    /// update the allOptions_ dict in DAOption based on the pyOptions from pyDAFoam
    void updateDAOption(PyObject* pyOptions);
};

template<class classType>
classType DAOption::getOption(const word key) const
{
    /*
    Description:
        Get an option from allOptions_
        NOTE: the return value can not be a dictionary. If one need to get a sub
        dictionary, use DAOption::getAllOptions().subDict("subDictName") instead

    Input:
        key: the key in the DAOption::allOptions_ dictionary

    Output:
        classType: the value for the DAOption::allOptions_[key]

    Example:
        If DAOption::allOptions_ reads
        {
            flowCondition  Incompressible;
            nkTol          1.e-6;
            objFuncs       (CD CL);
        }
        Then, we can get the values by:
        {
            word val1=daOption.getOption<word>("flowCondition");
            scalar val2=daOption.getOption<scalar>("nkTol");
            List<word> val3=daOption.getOption< List<word> >("objFuncs");
        }
    */

    classType val;
    allOptions_.readEntry<classType>(key, val);
    return val;
}

template<class classType>
void DAOption::setOption(
    const word key,
    const classType value)
{
    /*
    Description:
        Set an option to DAOption::allOptions_. 
        NOTE: the value can not be a dictionary. 
        NOTE: changing allOptions_ will not change self.options in pyDAFoam

    Input:
        key: the key in the DAOption::allOptions_ dictionary
        value: the value to set

    Output:
        DAOption::allOptions_: the key value in allOptions will be replaced

    Example:
        If DAOption::allOptions_ reads
        {
            flowCondition  Incompressible;
            nkTol          1.e-6;
            objFuncs       (CD CL);
        }
        Then, we can set the values by:
        {
            daOption.setOption<word>("flowCondition","Compressible");
            daOption.setOption<scalar>("nkTol",1.0e-5);
            daOption.getOption< List<word> >("objFuncs",{CMX CMY});
        }
    */

    allOptions_.set(key, value);

    return;
}

template<class classType>
classType DAOption::getSubDictOption(
    const word subDict,
    const word subDictKey) const
{
    /*
    Description:
        Get value from a key from a sub dictionary in allOptions_

    Input:
        subDict: the name of the subDict
        subDictKey: the key in the subDict

    Output:
        classType: the value for the allOptions_[subDict][subDictKey]

    Example:
        If allOptions_ reads
        {
            flowBCs
            {
                useWallFunction true;
                value           1.0;
            };
            flowEndTime 100.0;
        }
        Then, we can get the sub dictionary flowBCs by:
        scalar val=daOption.getSubDictOption<scalar>("flowBCs","value");
    */

    dictionary subDictF = allOptions_.subDict(subDict);
    classType val;
    subDictF.readEntry<classType>(subDictKey, val);
    return val;
}

template<class classType>
void DAOption::setSubDictOption(
    const word subDict,
    const word subDictKey,
    const classType value)
{
    /*
    Description:
        Set a sub dictionary option to allOptions_
        NOTE: changing allOptions_ will not change self.options in pyDAFoam

    Input:
        subDict: the name of the subDict
        subDictKey: the key in the subDict
        value: the value to set

    Output:
        allOptions_: the key value in allOptions will be replaced

    Example:
        If allOptions_ reads
        {
            flowBCs
            {
                useWallFunction true;
                value           1.0;
            };
            flowEndTime 100.0;
        }
        Then, we can set the sub dictionary flowBCs-value by:
        daOption.setSubDictOption<scalar>("flowBCs","value",2.0);
    */

    dictionary& subDictF = allOptions_.subDict(subDict);
    subDictF.set(subDictKey, value);
    return;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
