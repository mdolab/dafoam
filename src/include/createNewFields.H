// --- createNewFields.H ---

// Define the name of your new field
const word fieldName = "U";

// Check if the field already exists in the TARGET mesh's database
// We use targetPtr_().db() to get the registry for the *new* mesh
if (!targetPtr_->thisDb().foundObject<volVectorField>(fieldName))
{
    Info << "DASolver: Creating and registering target field '"
         << fieldName << "' on mesh '"
         << targetPtr_().name() << "'" << endl;

    // If it doesn't exist, create it.
    // The 'new' keyword makes it "live in memory," and the
    // registry (`targetPtr_().db()`) will manage it.
    // 1. Get a REFERENCE to the target mesh, just like your example
    fvMesh& targetMesh = targetPtr_();
    volVectorField* U_target_ptr = new volVectorField
    (
        IOobject
        (
            fieldName,
            runTime.timeName(), // Use the main runTime
            targetMesh,  // <-- CRITICAL: Register to the TARGET registry
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        targetMesh // <-- CRITICAL: Create it on the TARGET mesh
        //dimensionedVector("U", dimVelocity, vector::zero),
        //"zeroGradient" // <-- ADD THIS
    );
}


// --- ADD THIS BLOCK FOR "p" ---
const word fieldNameP = "p";
if (!targetPtr_().foundObject<volScalarField>(fieldNameP))
{
    Info << "DASolver: Creating and registering target field '"
         << fieldNameP << "' on mesh '"
         << targetPtr_().name() << "'" << endl;
    
    fvMesh& targetMesh = targetPtr_();
    volScalarField* p_target_ptr = new volScalarField
    (
        IOobject
        (
        	fieldNameP, 
        	runTime.timeName(), 
        	targetMesh, 
        	IOobject::MUST_READ, 
        	IOobject::AUTO_WRITE
        ),
        targetMesh
        //dimensionedScalar(fieldNameP, dimensionSet(0, 2, -2, 0, 0, 0, 0), 0.0),// <-- ADD THIS // Use correct pressure dimension
        //"zeroGradient" 
    );
}

// --- ADD THIS BLOCK FOR "nuTilda" ---
const word fieldNameNuTilda = "nuTilda";
if (!targetPtr_().foundObject<volScalarField>(fieldNameNuTilda))
{
    Info << "DASolver: Creating and registering target field '"
         << fieldNameNuTilda << "' on mesh '"
         << targetPtr_().name() << "'" << endl;
    
    fvMesh& targetMesh = targetPtr_();
    volScalarField* nuTilda_target_ptr = new volScalarField
    (
        IOobject
        (
        	fieldNameNuTilda, 
        	runTime.timeName(), 
        	targetMesh, 
        	IOobject::MUST_READ, 
        	IOobject::AUTO_WRITE
        ),
        targetMesh
        //dimensionedScalar(fieldNameNuTilda, dimensionSet(0, 2, -1, 0, 0, 0, 0), 0.0), // Use correct turbulence dimension
        //"zeroGradient" 
    );
}

const word fieldNameNut = "nut";
if (!targetPtr_().foundObject<volScalarField>(fieldNameNut))
{
    Info << "DASolver: Creating and registering target field '"
         << fieldNameNut << "' on mesh '"
         << targetPtr_().name() << "'" << endl;
    
    fvMesh& targetMesh = targetPtr_();
    volScalarField* nut_ptr = new volScalarField
    (
        IOobject
        (
            fieldNameNut,
            runTime.timeName(),
            targetMesh, // Register to target mesh
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        targetMesh
        //dimensionedScalar(fieldNameNut, dimensionSet(0, 2, -1, 0, 0, 0, 0), 0.0), // Use same dim as nuTilda
        //"zeroGradient" 
    );
}


const word fieldNamePhi = "phi";
if (!targetPtr_().foundObject<surfaceScalarField>(fieldNamePhi))
{
    Info << "DASolver: Creating and registering target field '"
         << fieldNamePhi << "' on mesh '"
         << targetPtr_().name() << "'" << endl;

    // We need the "U" field *that lives on the target mesh*
    // to initialize the flux field
    const volVectorField& U_target = 
        targetPtr_().lookupObject<volVectorField>("U");

    fvMesh& targetMesh = targetPtr_();
    
    // Create and register the new phi field
    surfaceScalarField* phi_ptr = new surfaceScalarField
    (
        IOobject
        (
            fieldNamePhi,
            runTime.timeName(),
            targetMesh, // <-- Register to the target mesh's DB
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        fvc::flux(U_target) // <-- Calculate initial flux (will be zero)
    );
}



const word fieldNameAlpha = "alphaPorosity";
if (!targetPtr_().foundObject<volScalarField>(fieldNameAlpha))
{
    Info << "DASolver: Creating and registering target field '"
         << fieldNameAlpha << "' on mesh '"
         << targetPtr_().name() << "'" << endl;
    
    fvMesh& targetMesh = targetPtr_();
    volScalarField* alpha_ptr = new volScalarField
    (
        IOobject
        (
            fieldNameAlpha,
            runTime.timeName(),
            targetMesh, // Register to target mesh
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        targetMesh,
        dimensionedScalar(fieldNameAlpha, dimensionSet(0, 0, -1, 0, 0, 0, 0), 1.0), // Porosity is 1.0 by default
        "zeroGradient" 
    );
}



const word fieldNameFvSource = "fvSource";
if (!targetPtr_().foundObject<volVectorField>(fieldNameFvSource))
{
    Info << "DASolver: Creating and registering target field '"
         << fieldNameFvSource << "' on mesh '"
         << targetPtr_().name() << "'" << endl;
    
    fvMesh& targetMesh = targetPtr_();
    volVectorField* fvSource_ptr = new volVectorField
    (
        IOobject
        (
            fieldNameFvSource,
            runTime.timeName(),
            targetMesh, // Register to target mesh
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        targetMesh
        // fvSource is U/T, which is dimAcceleration
        //dimensionedVector(fieldNameFvSource, dimensionSet(0, 1, -2, 0, 0, 0, 0), vector::zero),
        //"zeroGradient" 
    );
}



