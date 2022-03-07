
/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    This file is modified from OpenFOAM's source code
    src/TurbulenceModels/turbulenceModels/derivedFvPatchFields/wallFunctions/nutWallFunctions
    /nutUSpaldingWallFunction/nutUSpaldingWallFunctionFvPatchScalarField.H

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

        NOTE: It's the nutUSpaldingWallFunction implementation in OpenFOAM-v2006,
        which supports two additional inputs: maxIter and tolerance. In DAFoam,
        we need to make sure this boundary condition provides as accurate as
        possible boundary values, so the default values are maxIter = 1000
        and tolerance = 1e-14. If this causes very slow simulations, 
        decrease maxIter or increate tolerance. 

        This boundary condition provides a wall constraint on the turbulent
        viscosity, i.e. \c nut, based on velocity, i.e. \c U. Using Spalding's
        law gives a continuous \c nut profile to the wall.
     
            \f[
                y^+ = u^+ + \frac{1}{E} \left[exp(\kappa u^+) - 1 - \kappa u^+\,
                    - 0.5 (\kappa u^+)^2 - \frac{1}{6} (\kappa u^+)^3\right]
            \f]
     
        where
        \vartable
            y^+     | Non-dimensional position
            u^+     | Non-dimensional velocity
            \kappa  | von Kármán constant
        \endvartable


    Usage
        Example of the boundary condition specification:
        \verbatim
        <patchName>
        {
            // Mandatory entries (unmodifiable)
            type            nutUSpaldingWallFunction;
     
            // Optional entries (unmodifiable)
            maxIter         100;
            tolerance       1e-8;
     
            // Optional (inherited) entries
            ...
        }
        \endverbatim
     
        where the entries mean:
        \table
          Property  | Description                         | Type   | Req'd  | Dflt
          type      | Type name: nutUBlendedWallFunction  | word   | yes    | -
          maxIter   | Number of Newton-Raphson iterations | label  | no     | 1000
          tolerance | Convergence tolerance               | scalar | no     | 1e-14
        \endtable
     
        The inherited entries are elaborated in:
          - \link nutWallFunctionFvPatchScalarField.H \endlink
  
    Note
        - Suffers from non-exact restart since \c correctNut() (called through
        \c turbulence->validate) returns a slightly different value every time
        it is called. This is since the seed for the Newton-Raphson iteration
        uses the current value of \c *this (\c =nut ).
        - This can be avoided by overriding the tolerance. This also switches on
        a pre-detection whether the current nut already satisfies the turbulence
        conditions and if so does not change it at all. This means that the nut
        only changes if it 'has really changed'. This probably should be used with
        a tight tolerance, to make sure to kick every iteration, e.g.
            maxIter     100;
            tolerance   1e-7;

\*---------------------------------------------------------------------------*/

#ifndef nutUSpaldingWallFunctionFvPatchScalarFieldDF_H
#define nutUSpaldingWallFunctionFvPatchScalarFieldDF_H

#include "nutWallFunctionFvPatchScalarField.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
          Class nutUSpaldingWallFunctionFvPatchScalarFieldDF Declaration
\*---------------------------------------------------------------------------*/

class nutUSpaldingWallFunctionFvPatchScalarFieldDF
    : public nutWallFunctionFvPatchScalarField
{
protected:
    // Protected Member Functions

    //- Calculate the turbulence viscosity
    virtual tmp<scalarField> calcNut() const;

    //- Calculate the friction velocity
    virtual tmp<scalarField> calcUTau(const scalarField& magGradU) const;

    //- Max iterations in calcNut
    const label maxIter_;

    //- Convergence tolerance
    const scalar tolerance_;

    //- Calculate the friction velocity and number of iterations for
    //- convergence
    virtual tmp<scalarField> calcUTau(
        const scalarField& magGradU,
        const label maxIter,
        scalarField& err) const;

    //- Write local wall function variables
    virtual void writeLocalEntries(Ostream&) const;

public:
    //- Runtime type information
    TypeName("nutUSpaldingWallFunctionDF");

    // Constructors

    //- Construct from patch and internal field
    nutUSpaldingWallFunctionFvPatchScalarFieldDF(
        const fvPatch&,
        const DimensionedField<scalar, volMesh>&);

    //- Construct from patch, internal field and dictionary
    nutUSpaldingWallFunctionFvPatchScalarFieldDF(
        const fvPatch&,
        const DimensionedField<scalar, volMesh>&,
        const dictionary&);

    //- Construct by mapping given
    //  nutUSpaldingWallFunctionFvPatchScalarFieldDF
    //  onto a new patch
    nutUSpaldingWallFunctionFvPatchScalarFieldDF(
        const nutUSpaldingWallFunctionFvPatchScalarFieldDF&,
        const fvPatch&,
        const DimensionedField<scalar, volMesh>&,
        const fvPatchFieldMapper&);

    //- Construct as copy
    nutUSpaldingWallFunctionFvPatchScalarFieldDF(
        const nutUSpaldingWallFunctionFvPatchScalarFieldDF&);

    //- Construct and return a clone
    virtual tmp<fvPatchScalarField> clone() const
    {
        return tmp<fvPatchScalarField>(
            new nutUSpaldingWallFunctionFvPatchScalarFieldDF(*this));
    }

    //- Construct as copy setting internal field reference
    nutUSpaldingWallFunctionFvPatchScalarFieldDF(
        const nutUSpaldingWallFunctionFvPatchScalarFieldDF&,
        const DimensionedField<scalar, volMesh>&);

    //- Construct and return a clone setting internal field reference
    virtual tmp<fvPatchScalarField> clone(
        const DimensionedField<scalar, volMesh>& iF) const
    {
        return tmp<fvPatchScalarField>(
            new nutUSpaldingWallFunctionFvPatchScalarFieldDF(*this, iF));
    }

    //- Destructor
    virtual ~nutUSpaldingWallFunctionFvPatchScalarFieldDF()
    {}

    // Member functions

    // Evaluation functions

    //- Calculate and return the yPlus at the boundary
    virtual tmp<scalarField> yPlus() const;

    // I-O

    //- Write
    virtual void write(Ostream& os) const;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
