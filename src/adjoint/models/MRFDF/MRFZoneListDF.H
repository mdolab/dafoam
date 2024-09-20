/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

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

#ifndef MRFZoneListDF_H
#define MRFZoneListDF_H

#include "fvMesh.H"
#include "dictionary.H"
#include "fvMatricesFwd.H"
#include "MRFZoneDF.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward declaration of friend functions and operators
class MRFZoneListDF;
Ostream& operator<<(Ostream& os, const MRFZoneListDF& models);

/*---------------------------------------------------------------------------*\
                         Class MRFZoneListDF Declaration
\*---------------------------------------------------------------------------*/

class MRFZoneListDF
    : public PtrList<MRFZoneDF>
{
    // Private Member Functions

    //- No copy construct
    MRFZoneListDF(const MRFZoneListDF&) = delete;

    //- No copy assignment
    void operator=(const MRFZoneListDF&) = delete;

protected:
    // Protected data

    //- Reference to the mesh database
    const fvMesh& mesh_;

public:
    //- Constructor
    MRFZoneListDF(const fvMesh& mesh, const dictionary& dict);

    //- Destructor
    ~MRFZoneListDF();

    // Member Functions

    //- Return active status
    bool active(const bool warn = false) const;

    //- Reset the source list
    void reset(const dictionary& dict);

    //- Return the frame acceleration
    tmp<volVectorField> DDt(
        const volVectorField& U) const;

    //- Return the frame acceleration
    tmp<volVectorField> DDt(
        const volScalarField& rho,
        const volVectorField& U) const;

    //- Make the given absolute velocity relative within the MRF region
    void makeRelative(volVectorField& U) const;

    //- Make the given absolute flux relative within the MRF region
    void makeRelative(surfaceScalarField& phi) const;

    //- Return the given absolute flux relative within the MRF region
    tmp<surfaceScalarField> relative(
        const tmp<surfaceScalarField>& phi) const;

    //- Return the given absolute boundary flux relative within
    //  the MRF region
    tmp<FieldField<fvsPatchField, scalar>> relative(
        const tmp<FieldField<fvsPatchField, scalar>>& tphi) const;

    //- Return the given absolute patch flux relative within
    //  the MRF region
    tmp<Field<scalar>> relative(
        const tmp<Field<scalar>>& tphi,
        const label patchi) const;

    //- Make the given absolute mass-flux relative within the MRF region
    void makeRelative(
        const surfaceScalarField& rho,
        surfaceScalarField& phi) const;

    //- Correct the boundary velocity for the rotation of the MRF region
    void correctBoundaryVelocity(volVectorField& U) const;

    //- Update MRFZone faces if the mesh topology changes
    //void update();

    //- Get the reference of the rotation speed omega from MRFZone
    const scalar& getOmegaRef() const;

    // I-O

    //- Read dictionary
    //bool read(const dictionary& dict);

    //- Write data to Ostream
    //bool writeData(Ostream& os) const;

    //- Ostream operator
    friend Ostream& operator<<(
        Ostream& os,
        const MRFZoneListDF& models);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
