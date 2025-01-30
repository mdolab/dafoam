/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    This file is modified from OpenFOAM's source code
    applications/utilities/mesh/manipulation/checkMesh/checkMeshGeometry.C

    OpenFOAM: The Open Source CFD Toolbox

    Copyright (C): 2011-2016 OpenFOAM Foundation

    OpenFOAM License:

        OpenFOAM is free software: you can redistribute it and/or modify it
        under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.
    
        OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
        ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
        FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
        for more details.
    
        You should have received a copy of the GNU General Public License
        along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "checkGeometry.H"

namespace Foam
{
//- Default transformation behaviour for position
class transformPositionList
{
public:
    //- Transform patch-based field
    void operator()(
        const coupledPolyPatch& cpp,
        List<pointField>& pts) const
    {
        // Each element of pts is all the points in the face. Convert into
        // lists of size cpp to transform.

        List<pointField> newPts(pts.size());
        forAll(pts, facei)
        {
            newPts[facei].setSize(pts[facei].size());
        }

        label index = 0;
        while (true)
        {
            label n = 0;

            // Extract for every face the i'th position
            pointField ptsAtIndex(pts.size(), Zero);
            forAll(cpp, facei)
            {
                const pointField& facePts = pts[facei];
                if (facePts.size() > index)
                {
                    ptsAtIndex[facei] = facePts[index];
                    n++;
                }
            }

            if (n == 0)
            {
                break;
            }

            // Now ptsAtIndex will have for every face either zero or
            // the position of the i'th vertex. Transform.
            cpp.transformPosition(ptsAtIndex);

            // Extract back from ptsAtIndex into newPts
            forAll(cpp, facei)
            {
                pointField& facePts = newPts[facei];
                if (facePts.size() > index)
                {
                    facePts[index] = ptsAtIndex[facei];
                }
            }

            index++;
        }

        pts.transfer(newPts);
    }
};
}

bool Foam::checkCoupledPoints(
    const polyMesh& mesh,
    const bool report,
    labelHashSet* setPtr)
{
    const pointField& p = mesh.points();
    const faceList& fcs = mesh.faces();
    const polyBoundaryMesh& patches = mesh.boundaryMesh();

    // Zero'th point on coupled faces
    //pointField nbrZeroPoint(fcs.size()-mesh.nInternalFaces(), vector::max);
    List<pointField> nbrPoints(fcs.size() - mesh.nInternalFaces());

    // Exchange zero point
    forAll(patches, patchi)
    {
        if (patches[patchi].coupled())
        {
            const coupledPolyPatch& cpp = refCast<const coupledPolyPatch>(
                patches[patchi]);

            forAll(cpp, i)
            {
                label bFacei = cpp.start() + i - mesh.nInternalFaces();
                const face& f = cpp[i];
                nbrPoints[bFacei].setSize(f.size());
                forAll(f, fp)
                {
                    const point& p0 = p[f[fp]];
                    nbrPoints[bFacei][fp] = p0;
                }
            }
        }
    }
    syncTools::syncBoundaryFaceList(
        mesh,
        nbrPoints,
        eqOp<pointField>(),
        transformPositionList());

    // Compare to local ones. Use same tolerance as for matching
    label nErrorFaces = 0;
    scalar avgMismatch = 0;
    label nCoupledPoints = 0;

    forAll(patches, patchi)
    {
        if (patches[patchi].coupled())
        {
            const coupledPolyPatch& cpp =
                refCast<const coupledPolyPatch>(patches[patchi]);

            if (cpp.owner())
            {
                scalarField smallDist(
                    cpp.calcFaceTol(
                        //cpp.matchTolerance(),
                        cpp,
                        cpp.points(),
                        cpp.faceCentres()));

                forAll(cpp, i)
                {
                    label bFacei = cpp.start() + i - mesh.nInternalFaces();
                    const face& f = cpp[i];

                    if (f.size() != nbrPoints[bFacei].size())
                    {
                        FatalErrorInFunction
                            << "Local face size : " << f.size()
                            << " does not equal neighbour face size : "
                            << nbrPoints[bFacei].size()
                            << abort(FatalError);
                    }

                    label fp = 0;
                    forAll(f, j)
                    {
                        const point& p0 = p[f[fp]];
                        scalar d = mag(p0 - nbrPoints[bFacei][j]);

                        if (d > smallDist[i])
                        {
                            if (setPtr)
                            {
                                setPtr->insert(cpp.start() + i);
                            }
                            nErrorFaces++;

                            break;
                        }

                        avgMismatch += d;
                        nCoupledPoints++;

                        fp = f.rcIndex(fp);
                    }
                }
            }
        }
    }

    reduce(nErrorFaces, sumOp<label>());
    reduce(avgMismatch, maxOp<scalar>());
    reduce(nCoupledPoints, sumOp<label>());

    if (nCoupledPoints > 0)
    {
        avgMismatch /= nCoupledPoints;
    }

    if (nErrorFaces > 0)
    {
        if (report)
        {
            Info << "  **Error in coupled point location: "
                 << nErrorFaces
                 << " faces have their 0th or consecutive vertex not opposite"
                 << " their coupled equivalent. Average mismatch "
                 << avgMismatch << "."
                 << endl;
        }

        return true;
    }
    else
    {
        if (report)
        {
            Info << "    Coupled point location match (average "
                 << avgMismatch << ") OK." << endl;
        }

        return false;
    }
}

Foam::label Foam::checkGeometry(
    const polyMesh& mesh,
    const autoPtr<surfaceWriter>& surfWriter,
    const autoPtr<writer<scalar>>& setWriter,
    const label maxIncorrectlyOrientedFaces)
{
    label noFailedChecks = 0;

    //Info << "\nChecking geometry..." << endl;

    // Get a small relative length from the bounding box
    const boundBox& globalBb = mesh.bounds();

    Info << "    Overall domain bounding box "
         << globalBb.min() << " " << globalBb.max() << endl;

    // Geometric directions
    const Vector<label> validDirs = (mesh.geometricD() + Vector<label>::one);
    labelList tmpList(3);
    forAll(tmpList, idxI) tmpList[idxI] = validDirs[idxI] / 2;
    Info << "    Mesh has " << mesh.nGeometricD()
         << " geometric (non-empty/wedge) directions " << tmpList << endl;

    // Solution directions
    const Vector<label> solDirs = (mesh.solutionD() + Vector<label>::one);
    forAll(tmpList, idxI) tmpList[idxI] = solDirs[idxI] / 2;
    Info << "    Mesh has " << mesh.nSolutionD()
         << " solution (non-empty) directions " << tmpList << endl;

    if (mesh.nGeometricD() < 3)
    {
        FatalErrorInFunction
            << "Mesh geometric directions is less than 3 and not supported!"
            << abort(FatalError);
    }

    if (mesh.checkClosedBoundary(true))
    {
        noFailedChecks++;
    }

    {
        cellSet cells(mesh, "nonClosedCells", mesh.nCells() / 100 + 1);
        cellSet aspectCells(mesh, "highAspectRatioCells", mesh.nCells() / 100 + 1);
        if (
            mesh.checkClosedCells(
                true,
                &cells,
                &aspectCells,
                mesh.geometricD()))
        {
            noFailedChecks++;

            label nNonClosed = returnReduce(cells.size(), sumOp<label>());

            if (nNonClosed > 0)
            {
                Info << "  <<Writing " << nNonClosed
                     << " non closed cells to set " << cells.name() << endl;
                cells.instance() = mesh.pointsInstance();
                cells.write();
                if (surfWriter.valid())
                {
                    mergeAndWrite(surfWriter(), cells);
                }
            }
        }

        label nHighAspect = returnReduce(aspectCells.size(), sumOp<label>());

        if (nHighAspect > 0)
        {
            Info << "  <<Writing " << nHighAspect
                 << " cells with high aspect ratio to set "
                 << aspectCells.name() << endl;
            aspectCells.instance() = mesh.pointsInstance();
            aspectCells.write();
            if (surfWriter.valid())
            {
                mergeAndWrite(surfWriter(), aspectCells);
            }
        }
    }

    {
        faceSet faces(mesh, "zeroAreaFaces", mesh.nFaces() / 100 + 1);
        if (mesh.checkFaceAreas(true, &faces))
        {
            noFailedChecks++;

            label nFaces = returnReduce(faces.size(), sumOp<label>());

            if (nFaces > 0)
            {
                Info << "  <<Writing " << nFaces
                     << " zero area faces to set " << faces.name() << endl;
                faces.instance() = mesh.pointsInstance();
                faces.write();
                if (surfWriter.valid())
                {
                    mergeAndWrite(surfWriter(), faces);
                }
            }
        }
    }

    {
        cellSet cells(mesh, "zeroVolumeCells", mesh.nCells() / 100 + 1);
        if (mesh.checkCellVolumes(true, &cells))
        {
            noFailedChecks++;

            label nCells = returnReduce(cells.size(), sumOp<label>());

            if (nCells > 0)
            {
                Info << "  <<Writing " << nCells
                     << " zero volume cells to set " << cells.name() << endl;
                cells.instance() = mesh.pointsInstance();
                cells.write();
                if (surfWriter.valid())
                {
                    mergeAndWrite(surfWriter(), cells);
                }
            }
        }
    }

    {
        faceSet faces(mesh, "nonOrthoFaces", mesh.nFaces() / 100 + 1);
        if (mesh.checkFaceOrthogonality(true, &faces))
        {
            noFailedChecks++;
        }

        label nFaces = returnReduce(faces.size(), sumOp<label>());

        if (nFaces > 0)
        {
            Info << "  <<Writing " << nFaces
                 << " non-orthogonal faces to set " << faces.name() << endl;
            faces.instance() = mesh.pointsInstance();
            faces.write();
            if (surfWriter.valid())
            {
                mergeAndWrite(surfWriter(), faces);
            }
        }
    }

    {
        faceSet faces(mesh, "wrongOrientedFaces", mesh.nFaces() / 100 + 1);
        if (mesh.checkFacePyramids(true, -SMALL, &faces))
        {
            if (maxIncorrectlyOrientedFaces > 0)
            {
                Info << "maxIncorrectlyOrientedFaces threshold is set to " << maxIncorrectlyOrientedFaces << endl;
            }

            label nFaces = returnReduce(faces.size(), sumOp<label>());

            // if nFaces is less than maxIncorrectlyOrientedFaces
            // we do not consider it is a failed mesh. This allows
            // users to relax the mesh check for incorrectly oriented
            // mesh faces
            if (nFaces > maxIncorrectlyOrientedFaces)
            {
                noFailedChecks++;
            }

            if (nFaces > 0)
            {
                Info << "  <<Writing " << nFaces
                     << " faces with incorrect orientation to set "
                     << faces.name() << endl;
                faces.instance() = mesh.pointsInstance();
                faces.write();
                if (surfWriter.valid())
                {
                    mergeAndWrite(surfWriter(), faces);
                }
            }
        }
    }

    {
        faceSet faces(mesh, "skewFaces", mesh.nFaces() / 100 + 1);
        if (mesh.checkFaceSkewness(true, &faces))
        {
            noFailedChecks++;

            label nFaces = returnReduce(faces.size(), sumOp<label>());

            if (nFaces > 0)
            {
                Info << "  <<Writing " << nFaces
                     << " skew faces to set " << faces.name() << endl;
                faces.instance() = mesh.pointsInstance();
                faces.write();
                if (surfWriter.valid())
                {
                    mergeAndWrite(surfWriter(), faces);
                }
            }
        }
    }

    {
        faceSet faces(mesh, "coupledFaces", mesh.nFaces() / 100 + 1);
        if (checkCoupledPoints(mesh, true, &faces))
        {
            noFailedChecks++;

            label nFaces = returnReduce(faces.size(), sumOp<label>());

            if (nFaces > 0)
            {
                Info << "  <<Writing " << nFaces
                     << " faces with incorrectly matched 0th (or consecutive)"
                     << " vertex to set "
                     << faces.name() << endl;
                faces.instance() = mesh.pointsInstance();
                faces.write();
                if (surfWriter.valid())
                {
                    mergeAndWrite(surfWriter(), faces);
                }
            }
        }
    }

    return noFailedChecks;
}
