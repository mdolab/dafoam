/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

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
