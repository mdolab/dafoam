Time& runTime = runTimePtr_();
// reset time to 0
runTime.setTime(0.0, 0);
fvMesh& mesh = meshPtr_();
simpleControl& simple = simplePtr_();
volScalarField& p = pPtr_();
volVectorField& U = UPtr_();
surfaceScalarField& phi = phiPtr_();
volScalarField& alphaPorosity = alphaPorosityPtr_();
singlePhaseTransportModel& laminarTransport = laminarTransportPtr_();
scalar& cumulativeContErr = cumulativeContErr_;
label& pRefCell = pRefCell_;
scalar& pRefValue = pRefValue_;
volScalarField& T = TPtr_();
dimensionedScalar Pr = PrPtr_();
dimensionedScalar Prt = PrtPtr_();
volScalarField& alphat = alphatPtr_();

// create a file to store the objective values
this->initializeObjFuncHistFilePtr("objFuncHist");

this->setDAObjFuncList();

volScalarField Q
(
    IOobject
    (
        "Q",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("Q", dimensionSet(0,0,-1,1,0,0,0),1000.0),
    calculatedFvPatchScalarField::typeName
);
