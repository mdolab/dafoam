/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Compute the coloring for Jacobian matrices using the parallel
        distance 2 method

\*---------------------------------------------------------------------------*/

#ifndef DAColoring_H
#define DAColoring_H

#include "fvOptions.H"
#include "surfaceFields.H"
#include "DAOption.H"
#include "DAUtility.H"
#include "DAStateInfo.H"
#include "DAModel.H"
#include "DAIndex.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DAColoring Declaration
\*---------------------------------------------------------------------------*/

class DAColoring
{

private:
    /// Disallow default bitwise copy construct
    DAColoring(const DAColoring&);

    /// Disallow default bitwise assignment
    void operator=(const DAColoring&);

protected:

   /// fvMesh object
   const fvMesh& mesh_;

   /// DAOption object
   const DAOption& daOption_;

    /// DAIndex object
   const DAIndex& daIndex_;

public:
    /// Constructors
    DAColoring(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    /// Destructor
    virtual ~DAColoring()
    {
    }

    // Members
    /// number of non-zero elements in a matrix
    void getMatNonZeros(
        const Mat conMat,
        label& maxCols,
        scalar& allNonZeros) const;

    /// find the index of a prescribed value in an array
    label find_index(
        const label target,
        const label start,
        const label size,
        const label* valArray) const;

    /// check if there is non-colored columns
    void coloringComplete(
        const Vec colors,
        label& colorCounter,
        label& notColored) const;

    /// a parallel distance-2 graph coloring function
    void parallelD2Coloring(
        const Mat conMat,
        Vec colors,
        label& nColors) const;

    /// validate if there is coloring conflict
    void validateColoring(
        Mat conMat,
        Vec colors) const;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
