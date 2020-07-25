/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Test functions in libDAFoamIncompressible.so

\*---------------------------------------------------------------------------*/

#ifndef TestDAFoamIncompressible_H
#define TestDAFoamIncompressible_H

#include <petscksp.h>
#include "fvCFD.H"
#include "Python.h"
#include "DAUtility.H"
#include "DAOption.H"
#include "DAStateInfo.H"
#include "DATurbulenceModel.H"
#include "simpleControl.H"
#include "DAModel.H"
#include "DAIndex.H"
#include "DAObjFunc.H"
#include "DARegDbSinglePhaseTransportModel.H"
#include "DARegDbTurbulenceModelIncompressible.H"
#include "DAField.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class TestDAFoamIncompressible Declaration
\*---------------------------------------------------------------------------*/

class TestDAFoamIncompressible
{

private:
    /// Disallow default bitwise copy construct
    TestDAFoamIncompressible(const TestDAFoamIncompressible&);

    /// Disallow default bitwise assignment
    void operator=(const TestDAFoamIncompressible&);

    /// all the arguments
    char* argsAll_;

public:
    // Constructors

    /// Construct from components
    TestDAFoamIncompressible(char* argsAll);

    /// Destructor
    virtual ~TestDAFoamIncompressible();

    /// check if the dict matches the reference
    label validateOFDict(const dictionary& ofDict1);

    /// test DAUtility
    label testDAUtility(
        PyObject* pyDict,
        PyObject* pyStrList);

    /// test DAOption
    label testDAOption(PyObject* pyDict);

    /// test DAModel
    label testDAModel(PyObject* pyDict);

    /// test DAStateInfo
    label testDAStateInfo(PyObject* pyDict);

    /// test DAObjFunc
    label testDAObjFunc(PyObject* pyDict);

    /// test DAField
    label testDAField(PyObject* pyDict);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
