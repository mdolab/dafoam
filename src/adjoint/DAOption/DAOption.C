/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAOption.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Constructors
DAOption::DAOption(
    const fvMesh& mesh,
    PyObject* pyOptions)
    : regIOobject(
        IOobject(
            "DAOption", // always use DAOption for the db name
            mesh.time().timeName(),
            mesh, // register to mesh
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            true // always register object
            )),
      mesh_(mesh)
{
    /*
    Description:
        Construct from fvMesh and a dict object from Python
    Input:
        mesh: Foam::fvMesh object
        pyOptions: dictionary from Python, which contains all
        options for DAFoam 
    */

    // now we need to convert the pyOptions<PyObject*> to allOptions_<dictionary> in OpenFOAM
    DAUtility::pyDict2OFDict(pyOptions, allOptions_);

    //Info << "All DAFoam Options:";
    //Info << this->getAllOptions() << endl;
}

DAOption::~DAOption()
{
}

void DAOption::updateDAOption(PyObject* pyOptions)
{
    /*
    Description:
        Update the allOptions_ dict in DAOption based on the pyOptions from pyDAFoam
    Input:
        pyOptions: A Python dictionary from pyDAFoam
    Output:
        allOptions_: the OpenFOAM dictionary used in DAOption
    */

    // clear up the existing allOptions_
    allOptions_.clear();
    // assign allOptions_ based on pyOptions
    DAUtility::pyDict2OFDict(pyOptions, allOptions_);
}

// this is a virtual function for regIOobject
bool DAOption::writeData(Ostream& os) const
{
    allOptions_.write(os);
    return true;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
