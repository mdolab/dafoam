/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Class to compute intermediate variables such as the mean field, relative
        velocity field, primeToMeanField.

\*---------------------------------------------------------------------------*/

#ifndef DAIntmdVar_H
#define DAIntmdVar_H

#include "fvOptions.H"
#include "DAOption.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DAIntmdVar Declaration
\*---------------------------------------------------------------------------*/

class DAIntmdVar
{

private:
    /// Disallow default bitwise copy construct
    DAIntmdVar(const DAIntmdVar&);

    /// Disallow default bitwise assignment
    void operator=(const DAIntmdVar&);

protected:
    /// Foam::fvMesh object
    const fvMesh& mesh_;

    /// Foam::DAOption object
    const DAOption& daOption_;

    /// pointer list for volScalarfield
    UPtrList<volScalarField> volScalarList_;

    /// pointer list for volVectorfield
    UPtrList<volVectorField> volVectorList_;

    /// whether we need to compute any intermediate variables
    label hasIntmdVar_ = 0;

public:
    /// Constructors
    DAIntmdVar(
        const fvMesh& mesh,
        const DAOption& daOption);

    /// Destructor
    virtual ~DAIntmdVar()
    {
    }

    // Members

    /// Update the values for all intermediate variables defined in DAOption-intmdVar
    void update();

};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
