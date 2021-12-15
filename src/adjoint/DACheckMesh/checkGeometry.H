/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "label.H"
#include "HashSet.H"
#include "labelVector.H"
#include "polyMesh.H"
#include "cellSet.H"
#include "faceSet.H"
#include "pointSet.H"
#include "edgeHashes.H"
#include "wedgePolyPatch.H"
#include "unitConversion.H"
#include "polyMeshTetDecomposition.H"
#include "surfaceWriter.H"
#include "checkTools.H"
#include "functionObject.H"

#include "vtkSurfaceWriter.H"
#include "vtkSetWriter.H"
#include "writer.H"

#include "cyclicACMIPolyPatch.H"
#include "Time.H"
#include "PatchTools.H"

namespace Foam
{
class polyMesh;
class wedgePolyPatch;

label findOppositeWedge(const polyMesh&, const wedgePolyPatch&);

/// Check wedge orientation
bool checkWedges(
    const polyMesh&,
    const bool report,
    const Vector<label>&,
    labelHashSet*);

/// Check 0th vertex on coupled faces
bool checkCoupledPoints(const polyMesh&, const bool report, labelHashSet*);

/// check mesh quality
label checkGeometry(
    const polyMesh& mesh,
    const autoPtr<surfaceWriter>& surfWriter,
    const autoPtr<writer<scalar>>& setWriter,
    const label maxIncorrectlyOrientedFaces);

}
