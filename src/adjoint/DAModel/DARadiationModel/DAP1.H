/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
    Child class for the P1 radiation model

\*---------------------------------------------------------------------------*/

#ifndef DAP1_H
#define DAP1_H

#include "DARadiationModel.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DAP1 Declaration
\*---------------------------------------------------------------------------*/

class DAP1
    : public DARadiationModel
{

protected:
public:
    TypeName("P1");
    // Constructors

    //- Construct from components
    DAP1(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption);

    //- Destructor
    virtual ~DAP1()
    {
    }

    // Member functions
    virtual void addModelResidualCon(HashTable<List<List<word>>>& allCon) const;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
