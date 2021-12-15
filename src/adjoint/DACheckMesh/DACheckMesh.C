/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DACheckMesh.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Constructors
DACheckMesh::DACheckMesh(
    const DAOption& daOption,
    const Time& runTime1,
    const fvMesh& mesh1)
    : daOption_(daOption),
      runTime(runTime1),
      mesh(mesh1),
      surfWriter(nullptr),
      setWriter(nullptr)
{
    // Give an option to overwrite the default value of mesh quality check threshold
    fvMesh& meshNew = const_cast<fvMesh&>(mesh);
    maxNonOrth_ = daOption_.getSubDictOption<scalar>("checkMeshThreshold", "maxNonOrth");
    maxSkewness_ = daOption_.getSubDictOption<scalar>("checkMeshThreshold", "maxSkewness");
    maxAspectRatio_ = daOption_.getSubDictOption<scalar>("checkMeshThreshold", "maxAspectRatio");
    maxIncorrectlyOrientedFaces_ =
        daOption_.getSubDictOption<label>("checkMeshThreshold", "maxIncorrectlyOrientedFaces");
    meshNew.setNonOrthThreshold(maxNonOrth_);
    meshNew.setSkewThreshold(maxSkewness_);
    meshNew.setAspectThreshold(maxAspectRatio_);

    Info << "DACheckMesh Thresholds: " << endl;
    Info << "maxNonOrth: " << maxNonOrth_ << endl;
    Info << "maxSkewness: " << maxSkewness_ << endl;
    Info << "maxAspectRatio: " << maxAspectRatio_ << endl;
    Info << "maxIncorrectlyOrientedFaces: " << maxIncorrectlyOrientedFaces_ << endl;

    word surfaceFormat = "vtk";
    surfWriter.reset(surfaceWriter::New(surfaceFormat));
    setWriter.reset(writer<scalar>::New(vtkSetWriter<scalar>::typeName));
}

DACheckMesh::~DACheckMesh()
{
}

label DACheckMesh::run() const
{
    /*
    Description:
        Run checkMesh and return meshOK
    
    Output:
        meshOK: 1 means quality passes
    */

    label meshOK = 1;

    Info << "Checking mesh quality for time = " << runTime.timeName() << endl;

    label nFailedChecks = checkGeometry(mesh, surfWriter, setWriter, maxIncorrectlyOrientedFaces_);

    if (nFailedChecks)
    {
        Info << "\nFailed " << nFailedChecks << " mesh checks.\n"
             << endl;
        meshOK = 0;
    }
    else
    {
        Info << "\nMesh OK.\n"
             << endl;
    }

    return meshOK;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //