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

#ifndef IOMRFZoneListDF_H
#define IOMRFZoneListDF_H

#include "IOdictionary.H"
#include "MRFZoneListDF.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                        Class IOMRFZoneListDF Declaration
\*---------------------------------------------------------------------------*/

class IOMRFZoneListDF
    : public IOdictionary,
      public MRFZoneListDF
{
private:
    // Private Member Functions

    //- Create IO object if dictionary is present
    IOobject createIOobject(const fvMesh& mesh) const;

    //- No copy construct
    IOMRFZoneListDF(const IOMRFZoneListDF&) = delete;

    //- No copy assignment
    void operator=(const IOMRFZoneListDF&) = delete;

public:
    // Constructors

    //- Construct from mesh
    IOMRFZoneListDF(const fvMesh& mesh);

    //- Destructor
    virtual ~IOMRFZoneListDF()
    {
    }

    // Member Functions

    //- Read dictionary
    virtual bool read();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
