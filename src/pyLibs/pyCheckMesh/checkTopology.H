#include "label.H"
#include "autoPtr.H"
#include "writer.H"

namespace Foam
{
class polyMesh;
class surfaceWriter;
class pointSet;

template<class PatchType>
void checkPatch
(
    const bool allGeometry,
    const word& name,
    const PatchType& pp,
    pointSet& points
);

label checkTopology
(
    const polyMesh&,
    const bool,
    const bool,
    const autoPtr<surfaceWriter>&,
    const autoPtr<writer<scalar>>&
);
}
