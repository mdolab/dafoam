/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    This file is a modified combination of the OpenFOAM's source codes
    src/TurbulenceModels/turbulenceModels/derivedFvPatchFields/wallFunctions/
    nutWallFunctions/nutUSpaldingWallFunction/nutUSpaldingWallFunctionFvPatchScalarField.H,
    src/TurbulenceModels/turbulenceModels/derivedFvPatchFields/wallFunctions/
    nutWallFunctions/nutkRoughWallFunction/nutkRoughWallFunctionFvPatchScalarField.H.

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

    Description:

        NOTE: It's a simplified nut wall function usable with any turbulence model
	(only available for flat plates for now).

        This boundary condition provides a wall constraint on the turbulent viscosity,
	i.e. \c nut, based on velocity, i.e. \c U. Using the differential form of the
	log law to obtain utau and the wall shear stress relation to get nutw

        The inherited entries are elaborated in:
          - \link nutWallFunctionFvPatchScalarField.H \endlink

    Note
        - Suffers with highly curved surfaces or cases with strong PG since
        the implementation for buffer and subviscous layer has not been included
\*---------------------------------------------------------------------------*/

#ifndef nutWallFunctionFvPatchScalarFieldDF_H
#define nutWallFunctionFvPatchScalarFieldDF_H

#include "nutWallFunctionFvPatchScalarField.H"
#include "DAModel.H"
#include "DAOption.H"
//#include "fvMesh.H"
//#include "IOdictionary.H"
#include "primitiveFieldsFwd.H"
#include "surfaceFieldsFwd.H"
#include "geometricOneField.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
          Class nutWallFunctionFvPatchScalarFieldDF Declaration
\*---------------------------------------------------------------------------*/

class nutUtauWallFunctionFvPatchScalarFieldDF
    : public nutWallFunctionFvPatchScalarField
//      public IOdictionary
{
protected:

/*        const Time& runTime_;
        const fvMesh& mesh_;*/

        //- Roughness height [m]
        scalarField Ks_;

        //- Roughness constant (dimensionless)
        scalarField Cs_;

	//- Patch-local design variable, exposed via objectRegistry for adjoint updates
	volScalarField& etaWallDV;

    // Protected Member Functions

        //- Compute the roughness function
        virtual scalar fnRough(const scalar KsPlus, const scalar Cs) const;

        //- Calculate the turbulence viscosity
        virtual tmp<scalarField> calcNut() const;

        //- Calculate the friction velocity
        virtual tmp<scalarField> calcUTau(const scalarField& magGradU) const;


public:
    //- Runtime type information
    TypeName("nutUtauWallFunctionDF");

// Constructors

        //- Construct from patch and internal field
        nutUtauWallFunctionFvPatchScalarFieldDF
        (
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&
        );

        //- Construct from patch, internal field and dictionary
        nutUtauWallFunctionFvPatchScalarFieldDF
        (
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&,
            const dictionary&
        );

        //- Construct by mapping given patch field
        nutUtauWallFunctionFvPatchScalarFieldDF
        (
            const nutUtauWallFunctionFvPatchScalarFieldDF&,
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&,
            const fvPatchFieldMapper&
        );

/*	//-Construct eta field from IOdictionary
	nutUtauWallFunctionFvPatchScalarFieldDF
	(
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&,
            const volVectorField&,
            const surfaceScalarField&,
            const surfaceScalarField&,
            const word&
	);*/

        //- Construct as copy
        nutUtauWallFunctionFvPatchScalarFieldDF
        (
            const nutUtauWallFunctionFvPatchScalarFieldDF&
        );

        //- Construct and return a clone
        virtual tmp<fvPatchScalarField> clone() const
        {
            return tmp<fvPatchScalarField>
            (
                new nutUtauWallFunctionFvPatchScalarFieldDF(*this)
            );
        }

        //- Construct as copy with new internal field
        nutUtauWallFunctionFvPatchScalarFieldDF
        (
            const nutUtauWallFunctionFvPatchScalarFieldDF&,
            const DimensionedField<scalar, volMesh>&
        );

        //- Construct and return a clone setting internal field reference
        virtual tmp<fvPatchScalarField> clone
        (
            const DimensionedField<scalar, volMesh>& iF
        ) const
        {
            return tmp<fvPatchScalarField>
            (
                new nutUtauWallFunctionFvPatchScalarFieldDF(*this, iF)
            );
        }

// Member functions

    // Access

            //- Return Ks
            scalarField& Ks()
            {
                return Ks_;
            }

            //- Return Cs
            scalarField& Cs()
            {
                return Cs_;
            }
/*	    //- Return eta
	    scalarField& etaWallDV()
	    {
		return etaWallDV_;
	    }*/

    // Evaluate

        //- Calculate and return the yPlus at the boundary
        virtual tmp<scalarField> yPlus() const;

    // Mapping

        //- Map from another patch field
        virtual void autoMap(const fvPatchFieldMapper&);

        //- Reverse map the values from a different field
        virtual void rmap(const fvPatchScalarField&, const labelList&);

    // I-O
	//- Wrap etaWallDV_ as a volScalarField and register it for access in DASolver
	virtual volScalarField& registerEtaField(const objectRegistry&);
        //- Write
	virtual void write(Ostream& os) const;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
