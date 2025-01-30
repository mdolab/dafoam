/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    This file is modified from OpenFOAM's source code
    applications/utilities/mesh/manipulation/checkMesh/checkTools.C

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

#include "checkTools.H"
#include "polyMesh.H"
#include "globalMeshData.H"
#include "hexMatcher.H"
#include "wedgeMatcher.H"
#include "prismMatcher.H"
#include "pyrMatcher.H"
#include "tetWedgeMatcher.H"
#include "tetMatcher.H"
#include "IOmanip.H"
#include "pointSet.H"
#include "faceSet.H"
#include "cellSet.H"
#include "Time.H"
#include "surfaceWriter.H"
#include "syncTools.H"
#include "globalIndex.H"
#include "PatchTools.H"
#include "functionObject.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void Foam::mergeAndWrite(
    const polyMesh& mesh,
    const surfaceWriter& writer,
    const word& name,
    const indirectPrimitivePatch& setPatch,
    const fileName& outputDir)
{
    if (Pstream::parRun())
    {
        labelList pointToGlobal;
        labelList uniqueMeshPointLabels;
        autoPtr<globalIndex> globalPoints;
        autoPtr<globalIndex> globalFaces;
        faceList mergedFaces;
        pointField mergedPoints;
        Foam::PatchTools::gatherAndMerge(
            mesh,
            setPatch.localFaces(),
            setPatch.meshPoints(),
            setPatch.meshPointMap(),

            pointToGlobal,
            uniqueMeshPointLabels,
            globalPoints,
            globalFaces,

            mergedFaces,
            mergedPoints);

        // Write
        if (Pstream::master())
        {
            writer.write(
                outputDir,
                name,
                meshedSurfRef(
                    mergedPoints,
                    mergedFaces));
        }
    }
    else
    {
        writer.write(
            outputDir,
            name,
            meshedSurfRef(
                setPatch.localPoints(),
                setPatch.localFaces()));
    }
}

void Foam::mergeAndWrite(
    const surfaceWriter& writer,
    const faceSet& set)
{
    const polyMesh& mesh = refCast<const polyMesh>(set.db());

    const indirectPrimitivePatch setPatch(
        IndirectList<face>(mesh.faces(), set.sortedToc()),
        mesh.points());

    fileName outputDir(
        set.time().globalPath()
        / functionObject::outputPrefix
        / mesh.pointsInstance()
        / set.name());
    outputDir.clean();

    mergeAndWrite(mesh, writer, set.name(), setPatch, outputDir);
}

void Foam::mergeAndWrite(
    const surfaceWriter& writer,
    const cellSet& set)
{
    const polyMesh& mesh = refCast<const polyMesh>(set.db());
    const polyBoundaryMesh& pbm = mesh.boundaryMesh();

    // Determine faces on outside of cellSet
    bitSet isInSet(mesh.nCells());
    forAllConstIter(cellSet, set, iter)
    {
        isInSet.set(iter.key());
    }

    boolList bndInSet(mesh.nBoundaryFaces());
    forAll(pbm, patchi)
    {
        const polyPatch& pp = pbm[patchi];
        const labelList& fc = pp.faceCells();
        forAll(fc, i)
        {
            bndInSet[pp.start() + i - mesh.nInternalFaces()] = isInSet[fc[i]];
        }
    }
    syncTools::swapBoundaryFaceList(mesh, bndInSet);

    DynamicList<label> outsideFaces(3 * set.size());
    for (label facei = 0; facei < mesh.nInternalFaces(); facei++)
    {
        const bool ownVal = isInSet[mesh.faceOwner()[facei]];
        const bool neiVal = isInSet[mesh.faceNeighbour()[facei]];

        if (ownVal != neiVal)
        {
            outsideFaces.append(facei);
        }
    }

    forAll(pbm, patchi)
    {
        const polyPatch& pp = pbm[patchi];
        const labelList& fc = pp.faceCells();
        if (pp.coupled())
        {
            forAll(fc, i)
            {
                label facei = pp.start() + i;

                const bool neiVal = bndInSet[facei - mesh.nInternalFaces()];
                if (isInSet[fc[i]] && !neiVal)
                {
                    outsideFaces.append(facei);
                }
            }
        }
        else
        {
            forAll(fc, i)
            {
                if (isInSet[fc[i]])
                {
                    outsideFaces.append(pp.start() + i);
                }
            }
        }
    }

    const indirectPrimitivePatch setPatch(
        IndirectList<face>(mesh.faces(), outsideFaces),
        mesh.points());

    fileName outputDir(
        set.time().globalPath()
        / functionObject::outputPrefix
        / mesh.pointsInstance()
        / set.name());
    outputDir.clean();

    mergeAndWrite(mesh, writer, set.name(), setPatch, outputDir);
}

// ************************************************************************* //
