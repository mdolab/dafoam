/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    This file is modified from OpenFOAM's source code
    src/finiteVolume/cfdTools/general/solutionControl/pimpleControl/pimpleControlI.C

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

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

inline Foam::label Foam::pimpleControlDF::nCorrPIMPLE() const
{
    return nCorrPIMPLE_;
}

inline Foam::label Foam::pimpleControlDF::nCorrPISO() const
{
    return nCorrPISO_;
}

inline Foam::label Foam::pimpleControlDF::corrPISO() const
{
    return corrPISO_;
}

inline bool Foam::pimpleControlDF::SIMPLErho() const
{
    return SIMPLErho_;
}

inline bool Foam::pimpleControlDF::correct()
{
    setFirstIterFlag();

    ++corrPISO_;

    if (debug)
    {
        Info << algorithmName_ << " correct: corrPISO = " << corrPISO_ << endl;
    }

    if (corrPISO_ <= nCorrPISO_)
    {
        return true;
    }

    corrPISO_ = 0;

    setFirstIterFlag();

    return false;
}

inline bool Foam::pimpleControlDF::storeInitialResiduals() const
{
    // Start from second PIMPLE iteration
    return (corr_ == 2) && (corrPISO_ == 0) && (corrNonOrtho_ == 0);
}

inline bool Foam::pimpleControlDF::firstIter() const
{
    return corr_ == 1;
}

inline bool Foam::pimpleControlDF::finalIter() const
{
    return converged_ || (corr_ == nCorrPIMPLE_);
}

inline bool Foam::pimpleControlDF::finalInnerIter() const
{
    return corrPISO_ == nCorrPISO_
        && corrNonOrtho_ == nNonOrthCorr_ + 1;
}

inline bool Foam::pimpleControlDF::solveFlow() const
{
    return solveFlow_;
}

inline bool Foam::pimpleControlDF::turbCorr()
{
    if (turbOnFinalIterOnly_)
    {
        if (finalIter())
        {
            setFirstIterFlag(true, true);
        }
    }
    else
    {
        if (firstIter())
        {
            setFirstIterFlag(true, true);
        }
        else
        {
            setFirstIterFlag();
        }
    }

    return !turbOnFinalIterOnly_ || finalIter();
}

// ************************************************************************* //
