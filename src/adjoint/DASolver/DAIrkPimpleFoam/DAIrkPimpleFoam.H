/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for DAIrkPimpleFoam
    
    This class is modified from OpenFOAM's source code
    applications/solvers/incompressible/pimpleFoam

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

#ifndef DAIrkPimpleFoam_H
#define DAIrkPimpleFoam_H

#include "DASolver.H"
#include "addToRunTimeSelectionTable.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
//#include "pimpleControlDF.H"    // A modified version of pimpleControl, Basically, we disable the output to the screen
#include "pimpleControl.H" // origional pimpleControl

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAIrkPimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class DAIrkPimpleFoam
    : public DASolver
{

protected:
    // Radau23 coefficients
    scalar D10;
    scalar D11;
    scalar D12;
    scalar D20;
    scalar D21;
    scalar D22;
    scalar w1;
    scalar w2;

    // SA-fv3 coefficients
    scalar sigmaNut;
    scalar kappa;
    scalar Cb1;
    scalar Cb2;
    scalar Cw1;
    scalar Cw2;
    scalar Cw3;
    scalar Cv1;
    scalar Cv2;

    // SA-fv3 functions
    tmp<volScalarField> chi(
        volScalarField& nuTilda,
        const volScalarField& nu);

    tmp<volScalarField> fv1(
        volScalarField& chi);

    tmp<volScalarField> fv2(
        volScalarField& chi,
        volScalarField& fv1);

    tmp<volScalarField> fv3(
        volScalarField& chi,
        volScalarField& fv1);

    tmp<volScalarField> fw(
        volScalarField& Stilda,
        volScalarField& nuTilda,
        const volScalarField& y);

    tmp<volScalarField> DnuTildaEff(
        volScalarField& nuTilda,
        const volScalarField& nu);

    void correctNut(
        volScalarField& nut,
        volScalarField& nuTilda,
        const volScalarField& nu);

public:
    TypeName("DAIrkPimpleFoam");
    // Constructors

    //- Construct from components
    DAIrkPimpleFoam(
        char* argsAll,
        PyObject* pyOptions);

    //- Destructor
    virtual ~DAIrkPimpleFoam()
    {
    }

    /// initialize fields and variables
    virtual void initSolver();

    /// solve the primal equations
    virtual label solvePrimal();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
