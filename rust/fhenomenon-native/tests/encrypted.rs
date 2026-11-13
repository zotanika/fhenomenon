use fhenomenon::*;
use fhenomenon_native::{NativeProvider, Operation};

fn load_provider() -> NativeProvider {
    let path = std::env::var_os("FHN_NATIVE_TEST_LIBRARY").expect(
        "set FHN_NATIVE_TEST_LIBRARY to the CMake-built libckks_fhn.so; native tests never skip",
    );
    // The test library is built from this repository's ABI-1 implementation.
    unsafe { NativeProvider::load(path, "{}") }.expect("load actual CKKS native provider")
}
fn near(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }
}

#[test]
fn encrypted_scoped_graph_uses_shared_native_arithmetic_and_strict_metadata() {
    let provider = load_provider();
    let rt = Runtime::default();
    provider.register(&rt).unwrap();
    let mut client = provider.create_client().unwrap();
    client.prepare_keys(&rt, &[Operation::Rotate(1)]).unwrap();
    let x = client.encrypt(&rt, &[0.5, -1.25, 2.0, 0.125]).unwrap();
    let options = ScopeOptions {
        contract: Some(provider.scope_contract()),
        ..Default::default()
    };
    let out = rt
        .run_scope(options, |scope| {
            let a = scope.input(&x)?;
            let b = provider.capture(scope, Operation::Add, vec![a, a])?;
            let c = provider.capture(scope, Operation::Rotate(1), vec![b])?;
            let d = provider.capture(scope, Operation::AddScalar(0.25), vec![c])?;
            let e = provider.capture(scope, Operation::LevelDown(0), vec![d])?;
            assert_eq!(rt.scope_stats().executed_commands, 0);
            assert_eq!(
                client.decrypt(&rt, &x).unwrap_err().code,
                ErrorCode::InvalidRequest
            );
            Ok(vec![e, b, e])
        })
        .unwrap();
    near(
        &client.decrypt(&rt, &out[0]).unwrap(),
        &[-2.25, 4.25, 0.5, 1.25],
    );
    near(
        &client.decrypt(&rt, &out[1]).unwrap(),
        &[1.0, -2.5, 4.0, 0.25],
    );
    assert_eq!(out[0].identity(), out[2].identity());
    assert_eq!(provider.metadata(&out[0]).unwrap().level, 0);
    assert_eq!(rt.scope_stats().executed_commands, 4);
    assert_eq!(rt.scope_stats().recipe_spans, 4);
    drop(out);
    drop(x);
    assert_eq!(rt.stats().live_bytes, 0);
}

#[test]
fn late_invalid_level_is_rejected_before_native_graph_execution() {
    let provider = load_provider();
    let rt = Runtime::default();
    provider.register(&rt).unwrap();
    let mut client = provider.create_client().unwrap();
    let x = client.encrypt(&rt, &[1.0, 2.0]).unwrap();
    let options = ScopeOptions {
        contract: Some(provider.scope_contract()),
        ..Default::default()
    };
    let before = rt.stats();
    let error = rt
        .run_scope(options, |s| {
            let a = s.input(&x)?;
            let b = provider.capture(s, Operation::Negate, vec![a])?;
            Ok(vec![provider.capture(
                s,
                Operation::LevelDown(provider.fresh_level() + 1),
                vec![b],
            )?])
        })
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidRequest);
    assert_eq!(rt.scope_stats().executed_commands, 0);
    assert_eq!(rt.stats().live_bytes, before.live_bytes);
    near(&client.decrypt(&rt, &x).unwrap(), &[1.0, 2.0]);
}

#[test]
fn invalid_inputs_and_foreign_contexts_fail_without_leaks() {
    let provider = load_provider();
    let rt = Runtime::default();
    provider.register(&rt).unwrap();
    let mut client = provider.create_client().unwrap();
    for invalid in [&[][..], &[f64::NAN][..], &[1.0, 2.0, 3.0][..]] {
        assert!(client.encrypt(&rt, invalid).is_err());
        assert_eq!(rt.stats().live_resources, 0);
    }
    assert!(provider.create_client().is_err());
    let x = client.encrypt(&rt, &[1.0, 2.0]).unwrap();
    assert_eq!(
        provider
            .execute(&rt, Operation::Add, vec![x.clone()])
            .unwrap_err()
            .code,
        ErrorCode::InvalidRequest
    );
    assert_eq!(
        provider
            .execute(&rt, Operation::AddScalar(f64::INFINITY), vec![x.clone()])
            .unwrap_err()
            .code,
        ErrorCode::InvalidRequest
    );
    let other = load_provider();
    other.register(&rt).unwrap();
    assert_eq!(
        other
            .execute(&rt, Operation::Negate, vec![x.clone()])
            .unwrap_err()
            .code,
        ErrorCode::InvalidRequest
    );
    assert!(other.metadata(&x).is_err());
    let second_runtime = Runtime::default();
    assert!(client.decrypt(&second_runtime, &x).is_err());
    assert_eq!(rt.stats().live_resources, 1);
}

#[test]
fn library_context_and_registered_recipes_outlive_provider_handle() {
    let provider = load_provider();
    let rt = Runtime::default();
    provider.register(&rt).unwrap();
    let mut client = provider.create_client().unwrap();
    let x = client.encrypt(&rt, &[2.0, 3.0]).unwrap();
    let request = provider
        .request(Operation::Add, vec![x.clone(), x.clone()])
        .unwrap();
    drop(provider);
    let out = rt.submit(request).unwrap();
    near(&client.decrypt(&rt, &out[0]).unwrap(), &[4.0, 6.0]);
    drop(client);
    drop(x);
    drop(rt);
    // This handle is the last owner of the allocation, context, and library.
    // Destruction after all adapters/runtimes have gone must remain valid.
    assert!(out[0].bytes() > 0);
    drop(out);
}

#[test]
fn evaluation_key_expiration_fails_transactionally_but_keyless_compute_survives() {
    let provider = load_provider();
    let rt = Runtime::default();
    provider.register(&rt).unwrap();
    let mut client = provider.create_client().unwrap();
    client.prepare_keys(&rt, &[Operation::Rotate(1)]).unwrap();
    let x = client.encrypt(&rt, &[1.0, 2.0]).unwrap();
    let rotated = provider
        .execute(&rt, Operation::Rotate(1), vec![x.clone()])
        .unwrap();
    near(&client.decrypt(&rt, &rotated).unwrap(), &[2.0, 1.0]);
    drop(rotated);
    drop(client);
    let before = rt.stats();
    assert_eq!(
        provider
            .execute(&rt, Operation::Rotate(1), vec![x.clone()])
            .unwrap_err()
            .code,
        ErrorCode::ExecutionFailed
    );
    assert_eq!(rt.stats().live_resources, before.live_resources);
    assert!(provider
        .execute(&rt, Operation::Add, vec![x.clone(), x])
        .is_ok());
}

#[test]
fn strict_tensor_and_relinearize_preserve_correct_physical_metadata() {
    let provider = load_provider();
    let rt = Runtime::default();
    provider.register(&rt).unwrap();
    let mut client = provider.create_client().unwrap();
    client.prepare_keys(&rt, &[Operation::Relinearize]).unwrap();
    let x = client.encrypt(&rt, &[0.5, -1.25]).unwrap();
    let initial = provider.metadata(&x).unwrap();
    let options = ScopeOptions {
        contract: Some(provider.scope_contract()),
        ..Default::default()
    };
    let out = rt
        .run_scope(options, |s| {
            let a = s.input(&x)?;
            let b = provider.capture(s, Operation::TensorProduct, vec![a, a])?;
            let c = provider.capture(s, Operation::Relinearize, vec![b])?;
            Ok(vec![b, c])
        })
        .unwrap();
    near(&client.decrypt(&rt, &out[1]).unwrap(), &[0.25, 1.5625]);
    assert_eq!(provider.metadata(&out[0]).unwrap().components, 3);
    assert_eq!(provider.metadata(&out[1]).unwrap().components, 2);
    assert_eq!(
        provider.metadata(&out[1]).unwrap().scale,
        initial.scale * initial.scale
    );
}

#[test]
fn budget_failure_precedes_native_output_allocation_and_preserves_source() {
    let provider = load_provider();
    let rt = Runtime::new(RuntimeOptions {
        max_live_bytes: provider.reserved_bytes(),
        ..Default::default()
    });
    provider.register(&rt).unwrap();
    let mut client = provider.create_client().unwrap();
    let x = client.encrypt(&rt, &[1.0, 2.0]).unwrap();
    assert_eq!(
        provider
            .execute(&rt, Operation::Negate, vec![x.clone()])
            .unwrap_err()
            .code,
        ErrorCode::BudgetExceeded
    );
    assert_eq!(rt.stats().live_resources, 1);
    near(&client.decrypt(&rt, &x).unwrap(), &[1.0, 2.0]);
}

#[test]
fn loading_errors_are_reported_and_configuration_is_not_silently_ignored() {
    assert!(unsafe { NativeProvider::load("/a/nonexistent/fhenomenon/library.so", "{}") }.is_err());
    let path = std::env::var_os("FHN_NATIVE_TEST_LIBRARY").unwrap();
    assert!(unsafe { NativeProvider::load(path, "{\"not_a_parameter\":1}") }.is_err());
}

#[test]
fn native_coarse_scope_kernel_fuses_rotate_add_and_preserves_export_contract() {
    for export_rotation in [false, true] {
        let provider = load_provider();
        let rt = Runtime::default();
        provider.register(&rt).unwrap();
        provider.register_scope_kernels(&rt).unwrap();
        let mut client = provider.create_client().unwrap();
        client.prepare_keys(&rt, &[Operation::Rotate(1)]).unwrap();
        let x = client.encrypt(&rt, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let options = ScopeOptions {
            contract: Some(provider.scope_contract()),
            ..Default::default()
        };
        let output = rt
            .run_scope(options, |s| {
                let a = s.input(&x)?;
                let b = provider.capture(s, Operation::Rotate(1), vec![a])?;
                let c = provider.capture(s, Operation::Add, vec![a, b])?;
                Ok(if export_rotation { vec![c, b] } else { vec![c] })
            })
            .unwrap();
        near(
            &client.decrypt(&rt, &output[0]).unwrap(),
            &[3.0, 5.0, 7.0, 5.0],
        );
        assert_eq!(rt.scope_stats().executed_commands, 2);
        if export_rotation {
            assert_eq!(rt.scope_stats().kernel_spans, 0);
            assert_eq!(rt.scope_stats().recipe_spans, 2);
            near(
                &client.decrypt(&rt, &output[1]).unwrap(),
                &[2.0, 3.0, 4.0, 1.0],
            );
        } else {
            assert_eq!(rt.scope_stats().kernel_spans, 1);
            assert_eq!(rt.scope_stats().recipe_spans, 0);
            assert_eq!(rt.stats().peak_live_bytes, 2 * provider.reserved_bytes());
        }
    }
}
