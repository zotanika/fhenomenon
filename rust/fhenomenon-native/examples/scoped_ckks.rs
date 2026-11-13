use fhenomenon::{Runtime, ScopeOptions};
use fhenomenon_native::{NativeProvider, Operation};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::var_os("FHN_NATIVE_TEST_LIBRARY")
        .ok_or("set FHN_NATIVE_TEST_LIBRARY to your trusted CMake-built libckks_fhn.so")?;
    // This executable loads the repository's native CKKS ABI implementation.
    let provider = unsafe { NativeProvider::load(path, "{}") }?;
    let runtime = Runtime::default();
    provider.register(&runtime)?;
    provider.register_scope_kernels(&runtime)?;

    let mut client = provider.create_client()?;
    client.prepare_keys(&runtime, &[Operation::Rotate(1)])?;
    let encrypted = client.encrypt(&runtime, &[1.0, 2.0, 3.0, 4.0])?;
    let options = ScopeOptions {
        contract: Some(provider.scope_contract()),
        ..Default::default()
    };
    let output = runtime.run_scope(options, |scope| {
        let input = scope.input(&encrypted)?;
        let rotation = provider.capture(scope, Operation::Rotate(1), vec![input])?;
        let sum = provider.capture(scope, Operation::Add, vec![rotation, input])?;
        Ok(vec![sum])
    })?;

    println!(
        "decrypted result: {:?}",
        client.decrypt(&runtime, &output[0])?
    );
    println!("scope execution: {:?}", runtime.scope_stats());
    println!("charged resources: {:?}", runtime.stats());
    Ok(())
}
