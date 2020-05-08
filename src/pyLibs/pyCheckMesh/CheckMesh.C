/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2.0

\*---------------------------------------------------------------------------*/

#include "CheckMesh.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Constructors
CheckMesh::CheckMesh(char* argsAll)
{
    argsAll_ = argsAll;
    meshOK_ = 1;
}

CheckMesh::~CheckMesh()
{
}

label CheckMesh::run()
{
    #include "setArgs.H"

    timeSelector::addOptions();
    #include "addRegionOption.H"
    argList::addBoolOption
    (
        "noTopology",
        "Skip checking the mesh topology"
    );
    argList::addBoolOption
    (
        "allGeometry",
        "Include bounding box checks"
    );
    argList::addBoolOption
    (
        "allTopology",
        "Include extra topology checks"
    );
    argList::addBoolOption
    (
        "writeAllFields",
        "Write volFields with mesh quality parameters"
    );
    argList::addOption
    (
        "writeFields",
        "wordList",
        "Write volFields with selected mesh quality parameters"
    );
    argList::addBoolOption
    (
        "meshQuality",
        "Read user-defined mesh quality criteria from system/meshQualityDict"
    );
    argList::addOption
    (
        "writeSets",
        "surfaceFormat",
        "Reconstruct and write all faceSets and cellSets in selected format"
    );

    #include "setRootCasePython.H"
    #include "createTime.H"
    instantList timeDirs = timeSelector::select0(runTime, args);
    #include "createNamedMesh.H"

    const bool noTopology  = args.found("noTopology");
    const bool allGeometry = args.found("allGeometry");
    const bool allTopology = args.found("allTopology");
    const bool meshQuality = args.found("meshQuality");

    const word surfaceFormat = args.opt<word>("writeSets", "");
    const bool writeSets = surfaceFormat.size();

    wordHashSet selectedFields;
    bool writeFields = args.readIfPresent
                       (
                           "writeFields",
                           selectedFields
                       );
    if (!writeFields && args.found("writeAllFields"))
    {
        selectedFields.insert("nonOrthoAngle");
        selectedFields.insert("faceWeight");
        selectedFields.insert("skewness");
        selectedFields.insert("cellDeterminant");
        selectedFields.insert("aspectRatio");
        selectedFields.insert("cellShapes");
        selectedFields.insert("cellVolume");
        selectedFields.insert("cellVolumeRatio");
        selectedFields.insert("minTetVolume");
        selectedFields.insert("cellRegion");
    }


    if (noTopology)
    {
        Info << "Disabling all topology checks." << nl << endl;
    }
    if (allTopology)
    {
        Info << "Enabling all (cell, face, edge, point) topology checks."
             << nl << endl;
    }
    if (allGeometry)
    {
        Info << "Enabling all geometry checks." << nl << endl;
    }
    if (meshQuality)
    {
        Info << "Enabling user-defined geometry checks." << nl << endl;
    }
    if (writeSets)
    {
        Info << "Reconstructing and writing " << surfaceFormat
             << " representation"
             << " of all faceSets and cellSets." << nl << endl;
    }
    if (selectedFields.size())
    {
        Info << "Writing mesh quality as fields " << selectedFields << nl
             << endl;
    }


    autoPtr<IOdictionary> qualDict;
    if (meshQuality)
    {
        qualDict.reset
        (
            new IOdictionary
            (
                IOobject
                (
                    "meshQualityDict",
                    mesh.time().system(),
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE
                )
            )
        );
    }


    autoPtr<surfaceWriter> surfWriter;
    autoPtr<writer<scalar>> setWriter;
    if (writeSets)
    {
        surfWriter = surfaceWriter::New(surfaceFormat);
        setWriter = writer<scalar>::New(vtkSetWriter<scalar>::typeName);
    }


    forAll(timeDirs, timeI)
    {
        runTime.setTime(timeDirs[timeI], timeI);

        polyMesh::readUpdateState state = mesh.readUpdate();

        if
        (
            !timeI
            || state == polyMesh::TOPO_CHANGE
            || state == polyMesh::TOPO_PATCH_CHANGE
        )
        {
            Info << "Time = " << runTime.timeName() << nl << endl;

            // Reconstruct globalMeshData
            mesh.globalData();

            printMeshStats(mesh, allTopology);

            label nFailedChecks = 0;

            if (!noTopology)
            {
                nFailedChecks += checkTopology
                                 (
                                     mesh,
                                     allTopology,
                                     allGeometry,
                                     surfWriter,
                                     setWriter
                                 );
            }

            nFailedChecks += checkGeometry
                             (
                                 mesh,
                                 allGeometry,
                                 surfWriter,
                                 setWriter
                             );

            if (meshQuality)
            {
                nFailedChecks += checkMeshQuality(mesh, qualDict(), surfWriter);
            }


            // Note: no reduction in nFailedChecks necessary since is
            //       counter of checks, not counter of failed cells,faces etc.

            if (nFailedChecks == 0)
            {
                Info << "\nMesh OK.\n" << endl;
            }
            else
            {
                Info << "\nFailed " << nFailedChecks << " mesh checks.\n"
                     << endl;
                meshOK_ = 0;
            }


            // Write selected fields
            Foam::writeFields(mesh, selectedFields);
        }
        else if (state == polyMesh::POINTS_MOVED)
        {
            Info << "Time = " << runTime.timeName() << nl << endl;

            label nFailedChecks = checkGeometry
                                  (
                                      mesh,
                                      allGeometry,
                                      surfWriter,
                                      setWriter
                                  );

            if (meshQuality)
            {
                nFailedChecks += checkMeshQuality(mesh, qualDict(), surfWriter);
            }


            if (nFailedChecks)
            {
                Info << "\nFailed " << nFailedChecks << " mesh checks.\n"
                     << endl;
            }
            else
            {
                Info << "\nMesh OK.\n" << endl;
                meshOK_ = 0;
            }


            // Write selected fields
            Foam::writeFields(mesh, selectedFields);
        }
    }

    Info << "End\n" << endl;

    return meshOK_;

}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //