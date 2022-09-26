/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        A modified version of MRF that allows changing the rotation speed at
        the runtime to enable derivatives wrt the rotation speed

    This class is modified from OpenFOAM's source code
    src/finiteVolume/cfdTools/general/MRF

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

#ifndef MRFZoneDF_H
#define MRFZoneDF_H

#include "dictionary.H"
#include "wordList.H"
#include "labelList.H"
#include "dimensionedScalar.H"
#include "dimensionedVector.H"
#include "volFieldsFwd.H"
#include "surfaceFields.H"
#include "fvMatricesFwd.H"
#include "mapPolyMesh.H"
#include "Function1.H"
#include "autoPtr.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward declaration of classes
class fvMesh;

/*---------------------------------------------------------------------------*\
                           Class MRFZoneDF Declaration
\*---------------------------------------------------------------------------*/

class MRFZoneDF
{
    // Private data

    //- Reference to the mesh database
    const fvMesh& mesh_;

    //- Name of the MRF region
    const word name_;

    //- Coefficients dictionary
    dictionary coeffs_;

    //- MRF region active flag
    bool active_;

    //- Name of cell zone
    word cellZoneName_;

    //- Cell zone ID
    label cellZoneID_;

    const wordRes excludedPatchNames_;

    labelList excludedPatchLabels_;

    //- Internal faces that are part of MRF
    labelList internalFaces_;

    //- Outside faces (per patch) that move with the MRF
    labelListList includedFaces_;

    //- Excluded faces (per patch) that do not move with the MRF
    labelListList excludedFaces_;

    //- Origin of the axis
    const vector origin_;

    //- Axis vector
    vector axis_;

    //- Angular velocity (rad/sec)
    scalar omega_;

    // Private Member Functions

    //- Divide faces in frame according to patch
    void setMRFFaces();

    //- Make the given absolute mass/vol flux relative within the MRF region
    template<class RhoFieldType>
    void makeRelativeRhoFlux(
        const RhoFieldType& rho,
        surfaceScalarField& phi) const;

    //- Make the given absolute mass/vol flux relative within the MRF region
    template<class RhoFieldType>
    void makeRelativeRhoFlux(
        const RhoFieldType& rho,
        FieldField<fvsPatchField, scalar>& phi) const;

    //- Make the given absolute mass/vol flux relative within the MRF region
    template<class RhoFieldType>
    void makeRelativeRhoFlux(
        const RhoFieldType& rho,
        Field<scalar>& phi,
        const label patchi) const;

    //- No copy construct
    MRFZoneDF(const MRFZoneDF&) = delete;

    //- No copy assignment
    void operator=(const MRFZoneDF&) = delete;

public:
    // Declare name of the class and its debug switch
    ClassName("MRFZoneDF");

    // Constructors

    //- Construct from fvMesh
    MRFZoneDF(
        const word& name,
        const fvMesh& mesh,
        const dictionary& dict,
        const word& cellZoneName = word::null);

    //- Return clone
    autoPtr<MRFZoneDF> clone() const
    {
        NotImplemented;
        return nullptr;
    }

    // Member Functions

    //- Return const access to the MRF region name
    inline const word& name() const;

    //- Return const access to the MRF active flag
    inline bool active() const;

    //- Return the current Omega vector
    vector Omega() const;

    //- Return the reference of rotation speed omega scalar
    const scalar& getOmegaRef() const
    {
        return omega_;
    }

    //- Update the mesh corresponding to given map
    void updateMesh(const mapPolyMesh& mpm)
    {
        // Only updates face addressing
        setMRFFaces();
    }

    //- Add the Coriolis force contribution to the acceleration field
    void addCoriolis(
        const volVectorField& U,
        volVectorField& ddtU) const;

    //- Make the given absolute velocity relative within the MRF region
    void makeRelative(volVectorField& U) const;

    //- Make the given absolute flux relative within the MRF region
    void makeRelative(surfaceScalarField& phi) const;

    //- Make the given absolute boundary flux relative
    //  within the MRF region
    void makeRelative(FieldField<fvsPatchField, scalar>& phi) const;

    //- Make the given absolute patch flux relative
    //  within the MRF region
    void makeRelative(Field<scalar>& phi, const label patchi) const;

    //- Make the given absolute mass-flux relative within the MRF region
    void makeRelative(
        const surfaceScalarField& rho,
        surfaceScalarField& phi) const;

    //- Correct the boundary velocity for the rotation of the MRF region
    void correctBoundaryVelocity(volVectorField& U) const;

    //- Update MRFZoneDF faces if the mesh topology changes
    //void update();

    // I-O

    //- Write
    //void writeData(Ostream& os) const;

    //- Read MRF dictionary
    //bool read(const dictionary& dict);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
#include "MRFZoneTemplatesDF.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "MRFZoneIDF.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
