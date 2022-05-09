/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    This file is modified from OpenFOAM's source code
    src/finiteVolume/cfdTools/general/solutionControl/pimpleControl/pimpleControl.C

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

#include "pimpleControlDF.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
defineTypeNameAndDebug(pimpleControlDF, 0);
}

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

void Foam::pimpleControlDF::read()
{
    solutionControl::read(false);

    const dictionary pimpleDict(dict());

    solveFlow_ = pimpleDict.lookupOrDefault("solveFlow", true);
    nCorrPIMPLE_ = pimpleDict.lookupOrDefault<label>("nOuterCorrectors", 1);
    nCorrPISO_ = pimpleDict.lookupOrDefault<label>("nCorrectors", 1);
    SIMPLErho_ = pimpleDict.lookupOrDefault("SIMPLErho", false);
    turbOnFinalIterOnly_ =
        pimpleDict.lookupOrDefault("turbOnFinalIterOnly", true);
}

bool Foam::pimpleControlDF::criteriaSatisfied()
{
    // no checks on first iteration - nothing has been calculated yet
    if ((corr_ == 1) || residualControl_.empty() || finalIter())
    {
        return false;
    }

    const bool storeIni = this->storeInitialResiduals();

    bool achieved = true;
    bool checked = false; // safety that some checks were indeed performed

    const dictionary& solverDict = mesh_.solverPerformanceDict();
    forAllConstIters(solverDict, iter)
    {
        const entry& solverPerfDictEntry = *iter;

        const word& fieldName = solverPerfDictEntry.keyword();
        const label fieldi = applyToField(fieldName);

        if (fieldi != -1)
        {
            Pair<scalar> residuals = maxResidual(solverPerfDictEntry);

            checked = true;

            scalar relative = 0.0;
            bool relCheck = false;

            const bool absCheck =
                (residuals.last() < residualControl_[fieldi].absTol);

            if (storeIni)
            {
                residualControl_[fieldi].initialResidual = residuals.first();
            }
            else
            {
                const scalar iniRes =
                    (residualControl_[fieldi].initialResidual + ROOTVSMALL);

                relative = residuals.last() / iniRes;
                relCheck = (relative < residualControl_[fieldi].relTol);
            }

            achieved = achieved && (absCheck || relCheck);

            if (debug)
            {
                Info << algorithmName_ << " loop:" << endl;

                Info << "    " << fieldName
                     << " PIMPLE iter " << corr_
                     << ": ini res = "
                     << residualControl_[fieldi].initialResidual
                     << ", abs tol = " << residuals.last()
                     << " (" << residualControl_[fieldi].absTol << ")"
                     << ", rel tol = " << relative
                     << " (" << residualControl_[fieldi].relTol << ")"
                     << endl;
            }
        }
    }

    return checked && achieved;
}

void Foam::pimpleControlDF::setFirstIterFlag(const bool check, const bool force)
{
    DebugInfo
        << "corr:" << corr_
        << " corrPISO:" << corrPISO_
        << " corrNonOrtho:" << corrNonOrtho_
        << endl;

    solutionControl::setFirstIterFlag(check && corrPISO_ <= 1, force);
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::pimpleControlDF::pimpleControlDF(fvMesh& mesh, const word& dictName)
    : solutionControl(mesh, dictName),
      solveFlow_(true),
      nCorrPIMPLE_(0),
      nCorrPISO_(0),
      corrPISO_(0),
      SIMPLErho_(false),
      turbOnFinalIterOnly_(true),
      converged_(false)
{
    read();

    Info << nl
         << algorithmName_;

    if (nCorrPIMPLE_ > 1)
    {
        if (residualControl_.empty())
        {
            Info << ": no residual control data found. "
                 << "Calculations will employ " << nCorrPIMPLE_
                 << " corrector loops" << nl;
        }
        else
        {
            Info << ": max iterations = " << nCorrPIMPLE_ << nl;

            for (const fieldData& ctrl : residualControl_)
            {
                Info << "    field " << ctrl.name << token::TAB
                     << ": relTol " << ctrl.relTol
                     << ", tolerance " << ctrl.absTol
                     << nl;
            }
        }
    }
    else
    {
        Info << ": Operating solver in PISO mode" << nl;
    }

    Info << endl;
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool Foam::pimpleControlDF::loop()
{
    read();

    ++corr_;

    if (debug)
    {
        Info << algorithmName_ << " loop: corr = " << corr_ << endl;
    }

    setFirstIterFlag();

    if (corr_ == nCorrPIMPLE_ + 1)
    {
        if (!residualControl_.empty() && (nCorrPIMPLE_ != 1))
        {
            if (debug)
            {
                Info << algorithmName_ << ": not converged within "
                     << nCorrPIMPLE_ << " iterations" << endl;
            }
        }

        corr_ = 0;
        mesh_.data::remove("finalIteration");
        return false;
    }

    bool completed = false;
    if (converged_ || criteriaSatisfied())
    {
        if (converged_)
        {
            if (debug)
            {
                Info << algorithmName_ << ": converged in " << corr_ - 1
                     << " iterations" << endl;
            }

            mesh_.data::remove("finalIteration");
            corr_ = 0;
            converged_ = false;

            completed = true;
        }
        else
        {
            if (debug)
            {
                Info << algorithmName_ << ": iteration " << corr_ << endl;
            }
            storePrevIterFields();

            mesh_.data::add("finalIteration", true);
            converged_ = true;
        }
    }
    else
    {
        if (finalIter())
        {
            mesh_.data::add("finalIteration", true);
        }

        if (corr_ <= nCorrPIMPLE_)
        {
            if (debug)
            {
                Info << algorithmName_ << ": iteration " << corr_ << endl;
            }
            storePrevIterFields();
            completed = false;
        }
    }

    return !completed;
}

// ************************************************************************* //
