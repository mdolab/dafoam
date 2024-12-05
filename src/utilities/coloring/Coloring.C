/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "Coloring.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Constructors
Coloring::Coloring(
    char* argsAll,
    PyObject* pyOptions)
    : argsAll_(argsAll),
      pyOptions_(pyOptions),
      argsPtr_(nullptr),
      runTimePtr_(nullptr),
      meshPtr_(nullptr)
{
}

Coloring::~Coloring()
{
}

void Coloring::run()
{
    /*
    Description:
        Run the coloring solver and save the computed colors to disks
    */
    
#include "setArgs.H"
#include "setRootCasePython.H"
#include "createTimePython.H"
#include "createMeshPython.H"
#include "createFields.H"
#include "createAdjoint.H"

    word solverName = daOption.getOption<word>("solverName");
    autoPtr<DAStateInfo> daStateInfo(DAStateInfo::New(solverName, mesh, daOption, daModel));
    
    // dRdW
    {
        DAJacCon daJacCon("dRdW", mesh, daOption, daModel, daIndex);

        if (!daJacCon.coloringExists())
        {
            dictionary options;
            const HashTable<List<List<word>>>& stateResConInfo = daStateInfo->getStateResConInfo();
            options.set("stateResConInfo", stateResConInfo);

            // need to first setup preallocation vectors for the dRdWCon matrix
            // because directly initializing the dRdWCon matrix will use too much memory
            daJacCon.setupJacConPreallocation(options);

            // now we can initilaize dRdWCon
            daJacCon.initializeJacCon(options);

            // setup dRdWCon
            daJacCon.setupJacCon(options);
            Info << "dRdWCon Created. " << mesh.time().elapsedClockTime() << " s" << endl;

            // compute the coloring
            Info << "Calculating dRdW Coloring... " << mesh.time().elapsedClockTime() << " s" << endl;
            daJacCon.calcJacConColoring();
            Info << "Calculating dRdW Coloring... Completed! " << mesh.time().elapsedClockTime() << " s" << endl;

            // clean up
            daJacCon.clear();
        }
    }

    return;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
