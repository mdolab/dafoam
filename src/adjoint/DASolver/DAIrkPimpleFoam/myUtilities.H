// This function is for swapping the upper and lower of a xxEqn
// Let's move it elswhere later
void DAIrkPimpleFoam::swap(List<scalar>& a, List<scalar>& b)
{
    List<scalar> temp = a;
    a = b;
    b = temp;
}

// L2 norm of a scalar list (or the primitive field of a volScalarField)
scalar DAIrkPimpleFoam::L2norm(const List<scalar>& v)
{
    scalar L2norm_v = 0.0;

    forAll(v, cellI)
    {
        L2norm_v += sqr(v[cellI]);
    }
    L2norm_v = std::sqrt(L2norm_v);

    return L2norm_v;
}
// Scale off cell volumes
scalar DAIrkPimpleFoam::L2norm(const List<scalar>& v, const List<scalar>& V)
{
    scalar L2norm_v = 0.0;

    forAll(v, cellI)
    {
        L2norm_v += sqr(v[cellI] / V[cellI]);
    }
    L2norm_v = std::sqrt(L2norm_v);

    return L2norm_v;
}

// L2 norm of a vector list (or the primitive field of a volVectorField)
vector DAIrkPimpleFoam::L2norm(const List<vector>& U)
{
    vector L2norm_U = vector::zero;

    forAll(U, cellI)
    {
        L2norm_U[0] += sqr(U[cellI][0]);
        L2norm_U[1] += sqr(U[cellI][1]);
        L2norm_U[2] += sqr(U[cellI][2]);
    }
    L2norm_U[0] = std::sqrt(L2norm_U[0]);
    L2norm_U[1] = std::sqrt(L2norm_U[1]);
    L2norm_U[2] = std::sqrt(L2norm_U[2]);

    return L2norm_U;
}
// Scale off cell volumes
vector DAIrkPimpleFoam::L2norm(const List<vector>& U, const List<scalar>& V)
{
    vector L2norm_U = vector::zero;

    forAll(U, cellI)
    {
        L2norm_U[0] += sqr(U[cellI][0] / V[cellI]);
        L2norm_U[1] += sqr(U[cellI][1] / V[cellI]);
        L2norm_U[2] += sqr(U[cellI][2] / V[cellI]);
    }
    L2norm_U[0] = std::sqrt(L2norm_U[0]);
    L2norm_U[1] = std::sqrt(L2norm_U[1]);
    L2norm_U[2] = std::sqrt(L2norm_U[2]);

    return L2norm_U;
}

// L2 norm of a surfaceScalarField
scalar DAIrkPimpleFoam::L2norm(const surfaceScalarField& Phi)
{
    scalar L2norm_Phi = 0.0;

    forAll(Phi.primitiveField(), cellI)
    {
        L2norm_Phi += sqr(Phi.primitiveField()[cellI]);
    }
    forAll(Phi.boundaryField(), patchI)
    {
        forAll(Phi.boundaryField()[patchI], faceI)
        {
            L2norm_Phi += sqr(Phi.boundaryField()[patchI][faceI]);
        }
    }
    L2norm_Phi = std::sqrt(L2norm_Phi);

    return L2norm_Phi;
}
// Scale off surface areas
scalar DAIrkPimpleFoam::L2norm(const surfaceScalarField& Phi, const surfaceScalarField& SfArea)
{
    scalar L2norm_Phi = 0.0;

    forAll(Phi.primitiveField(), cellI)
    {
        L2norm_Phi += sqr(Phi.primitiveField()[cellI] / SfArea.primitiveField()[cellI]);
    }
    forAll(Phi.boundaryField(), patchI)
    {
        forAll(Phi.boundaryField()[patchI], faceI)
        {
            L2norm_Phi += sqr(Phi.boundaryField()[patchI][faceI] / SfArea.boundaryField()[patchI][faceI]);
        }
    }
    L2norm_Phi = std::sqrt(L2norm_Phi);

    return L2norm_Phi;
}

vector DAIrkPimpleFoam::getMaxAbs(const List<vector>& U)
{
    vector maxAbsVec = {0.0, 0.0, 0.0};
    forAll(U, cellI)
    {
        if (maxAbsVec[0] < std::abs(U[cellI][0]))
        {
            maxAbsVec[0] = std::abs(U[cellI][0]);
        }

        if (maxAbsVec[1] < std::abs(U[cellI][1]))
        {
            maxAbsVec[1] = std::abs(U[cellI][1]);
        }

        if (maxAbsVec[2] < std::abs(U[cellI][2]))
        {
            maxAbsVec[2] = std::abs(U[cellI][2]);
        }
    }

    return maxAbsVec;
}
