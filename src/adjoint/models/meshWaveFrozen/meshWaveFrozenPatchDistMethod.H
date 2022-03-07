/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        A modified version of meshWave wall distance method
        Basically, we compute the wall distance only once and save it to y_. 
        When the mesh is deformed during optimization, we will NOT update y_.
        The reason we do this is that the meshWave fuction is not AD in 
        parallel so it will impact the adjoint derivative. Also, not updating 
        the wall distance during optimization has little impact on CFD 
    
    This file is modified from OpenFOAM's source code
    src/finiteVolume/fvMesh/wallDist/patchDistMethods/meshWave/meshWavePatchDistMethod.H

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

#ifndef meshWaveFrozenPatchDistMethod_H
#define meshWaveFrozenPatchDistMethod_H

#include "patchDistMethod.H"
#include "fvMesh.H"
#include "volFields.H"
#include "patchWave.H"
#include "patchDataWave.H"
#include "wallPointData.H"
#include "emptyFvPatchFields.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace patchDistMethods
{

/*---------------------------------------------------------------------------*\
                          Class meshWaveFrozen Declaration
\*---------------------------------------------------------------------------*/

class meshWaveFrozen
    : public patchDistMethod
{
    // Private Member Data

    //- Do accurate distance calculation for near-wall cells.
    const bool correctWalls_;

    //- Number of unset cells and faces.
    mutable label nUnset_;

    /// wall distance
    mutable volScalarField y_;

    /// wall normal vector
    mutable volVectorField n_;

    /// whether the wall distance has been computed
    mutable label isComputed_ = 0;

    // Private Member Functions

    //- No copy construct
    meshWaveFrozen(const meshWaveFrozen&) = delete;

    //- No copy assignment
    void operator=(const meshWaveFrozen&) = delete;

public:
    //- Runtime type information
    TypeName("meshWaveFrozen");

    // Constructors

    //- Construct from coefficients dictionary, mesh
    //  and fixed-value patch set
    meshWaveFrozen(
        const dictionary& dict,
        const fvMesh& mesh,
        const labelHashSet& patchIDs);

    //- Construct from mesh, fixed-value patch set and flag specifying
    //  whether or not to correct wall.
    //  Calculate for all cells. correctWalls : correct wall (face&point)
    //  cells for correct distance, searching neighbours.
    meshWaveFrozen(
        const fvMesh& mesh,
        const labelHashSet& patchIDs,
        const bool correctWalls = true);

    // Member Functions

    label nUnset() const
    {
        return nUnset_;
    }

    //- Correct the given distance-to-patch field
    virtual bool correct(volScalarField& y);

    //- Correct the given distance-to-patch and normal-to-patch fields
    virtual bool correct(volScalarField& y, volVectorField& n);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace patchDistMethods
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
