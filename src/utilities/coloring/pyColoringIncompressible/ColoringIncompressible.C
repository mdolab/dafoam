/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "ColoringIncompressible.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Constructors
ColoringIncompressible::ColoringIncompressible(
    char* argsAll,
    PyObject* pyOptions)
    : argsAll_(argsAll),
      pyOptions_(pyOptions),
      argsPtr_(nullptr),
      runTimePtr_(nullptr),
      meshPtr_(nullptr)
{
}

ColoringIncompressible::~ColoringIncompressible()
{
}

void ColoringIncompressible::run()
{
    /*
    Description:
        Run the coloring solver and save the computed colors to disks
    */
    
#include "setArgs.H"
#include "setRootCasePython.H"
#include "createTimePython.H"
#include "createMeshPython.H"
// 1. This is our dumb file. It creates 
// meshPtr_ only.

    DAOption daOption_t(meshPtr_(), pyOptions_);
    bool multiMeshMode = (daOption_t.getOption<word>("multiMesh") == "yes");

    // 3. Create 'targetPtr_' (coarse mesh) *only* if in multi-mesh mode
    //    (DASolver constructor does this, but this is a
    //    SEPARATE program, so it must do it too).
    if (multiMeshMode)
    {
        Info << "Multi-mesh mode: Loading target mesh for coloring." << endl;
        const word newRegion = "target";
        targetPtr_.reset( // 'targetPtr_' is a class member
            new fvMesh(
                IOobject(
                    newRegion,
                    runTimePtr_().timeName(),
                    runTimePtr_(),
                    IOobject::MUST_READ)));
    }

    // 4. THE HIJACK
    //    Create the 'mesh' reference that subsequent files will use.
    fvMesh& mesh = (multiMeshMode ? targetPtr_() : meshPtr_());

    if (multiMeshMode)
    {
        Info << "ColoringIncompressible: Hijack successful. Coloring will run on " << mesh.name() << endl;
    }
    
    // --- END OF MODIFICATION ---


    // 5. Now, these files will use the *hijacked* 'mesh' reference
#include "createFields.H"
#include "createAdjoint.H"

    word solverName = daOption.getOption<word>("solverName");
    autoPtr<DAStateInfo> daStateInfo(DAStateInfo::New(solverName, mesh, daOption, daModel));
    
    // dRdW
    {
        autoPtr<DAJacCon> daJacCon(DAJacCon::New("dRdW", mesh, daOption, daModel, daIndex));

        if (!daJacCon->coloringExists())
        {
            dictionary options;
            const HashTable<List<List<word>>>& stateResConInfo = daStateInfo->getStateResConInfo();
            options.set("stateResConInfo", stateResConInfo);

            // need to first setup preallocation vectors for the dRdWCon matrix
            // because directly initializing the dRdWCon matrix will use too much memory
            daJacCon->setupJacConPreallocation(options);

            // now we can initilaize dRdWCon
            daJacCon->initializeJacCon(options);

            // setup dRdWCon
            daJacCon->setupJacCon(options);
            Info << "dRdWCon Created. " << mesh.time().elapsedClockTime() << " s" << endl;

            // compute the coloring
            Info << "Calculating dRdW Coloring... " << mesh.time().elapsedClockTime() << " s" << endl;
            daJacCon->calcJacConColoring();
            Info << "Calculating dRdW Coloring... Completed! " << mesh.time().elapsedClockTime() << " s" << endl;

            // clean up
            daJacCon->clear();
        }
    }

    // dFdW
    const dictionary& allOptions = daOption.getAllOptions();
    dictionary objFuncDict = allOptions.subDict("objFunc");
    // create a dummy DAResidual just for initializing DAObjFunc
    autoPtr<DAResidual> daResidual(DAResidual::New("dummy", mesh, daOption, daModel, daIndex));
    forAll(objFuncDict.toc(), idxI)
    {
        word objFuncName = objFuncDict.toc()[idxI];
        dictionary objFuncSubDict = objFuncDict.subDict(objFuncName);
        forAll(objFuncSubDict.toc(), idxJ)
        {
            word objFuncPart = objFuncSubDict.toc()[idxJ];
            dictionary objFuncSubDictPart = objFuncSubDict.subDict(objFuncPart);

            autoPtr<DAJacCon> daJacCon(DAJacCon::New("dFdW", mesh, daOption, daModel, daIndex));

            word postFix = "_" + objFuncName + "_" + objFuncPart;

            if (!daJacCon->coloringExists(postFix))
            {
                autoPtr<DAObjFunc> daObjFunc(
                    DAObjFunc::New(
                        mesh,
                        daOption,
                        daModel,
                        daIndex,
                        daResidual,
                        objFuncName,
                        objFuncPart,
                        objFuncSubDictPart));

                dictionary options;
                const List<List<word>>& objFuncConInfo = daObjFunc->getObjFuncConInfo();
                const labelList& objFuncFaceSources = daObjFunc->getObjFuncFaceSources();
                const labelList& objFuncCellSources = daObjFunc->getObjFuncCellSources();
                options.set("objFuncConInfo", objFuncConInfo);
                options.set("objFuncFaceSources", objFuncFaceSources);
                options.set("objFuncCellSources", objFuncCellSources);

                // now we can initilaize dFdWCon
                daJacCon->initializeJacCon(options);

                // setup dFdWCon
                daJacCon->setupJacCon(options);
                Info << "dFdWCon Created. " << mesh.time().elapsedClockTime() << " s" << endl;

                // compute the coloring
                Info << "Calculating dFdW " << objFuncName << "-"
                     << objFuncPart << " Coloring... "
                     << mesh.time().elapsedClockTime() << " s" << endl;

                daJacCon->calcJacConColoring(postFix);

                Info << "Calculating dFdW " << objFuncName << "-"
                     << objFuncPart << " Coloring... Completed"
                     << mesh.time().elapsedClockTime() << " s" << endl;

                // clean up
                daJacCon->clear();
            }
        }
    }

    return;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
