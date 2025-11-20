fvMesh& target = targetPtr_();
Info << "Reading field uData\n"
   << endl;
ubDataPtr_.reset(
    new volVectorField(
        IOobject(
	 "ubData",
	 runTime.timeName(),
	 target,
	 IOobject::MUST_READ,
	 IOobject::AUTO_WRITE),
        target));
volVectorField& ubData = ubDataPtr_();
target.thisDb().store( ubDataPtr_ );
