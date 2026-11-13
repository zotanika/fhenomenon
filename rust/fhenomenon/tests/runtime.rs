use fhenomenon::*;
use std::{cell::Cell, rc::Rc, time::Duration};

fn value(rt: &Runtime, n: i64) -> Result<Value> {
    rt.make_value("integer", ResourceKind::Value, 8, || Ok(data(n)), vec![])
}
fn integer(v: &Value) -> i64 {
    *v.payload::<i64>().unwrap()
}
fn register(rt: &Runtime, calls: Rc<Cell<usize>>) {
    rt.add_recipe(Recipe::new("add", "integer.add", move |rt, req| {
        calls.set(calls.get() + 1);
        Ok(vec![value(
            rt,
            integer(&req.inputs[0]) + integer(&req.inputs[1]),
        )?])
    }))
    .unwrap();
}
struct IntegerContract;
impl ScopeContract for IntegerContract {
    fn describe(&self, v: &Value) -> Result<ScopeDescriptor> {
        Ok(ScopeDescriptor::new(
            v.format(),
            v.kind(),
            v.bytes(),
            data(()),
        ))
    }
    fn infer(
        &self,
        op: &str,
        inputs: &[ScopeDescriptor],
        _: &Data,
    ) -> Result<Vec<ScopeDescriptor>> {
        if op != "add" || inputs.len() != 2 {
            return Err(Error::invalid("invalid integer operation"));
        }
        Ok(vec![ScopeDescriptor::new(
            "integer",
            ResourceKind::Value,
            8,
            data(()),
        )])
    }
    fn validate(&self, v: &Value, _: &ScopeDescriptor) -> Result<()> {
        v.payload::<i64>().map(|_| ())
    }
}
fn strict() -> ScopeOptions {
    ScopeOptions {
        contract: Some(Rc::new(IntegerContract)),
        ..Default::default()
    }
}

#[test]
fn resource_budget_identity_and_dependency_lifetime() {
    let rt = Runtime::new(RuntimeOptions {
        max_live_bytes: 16,
        ..Default::default()
    });
    let first = value(&rt, 3).unwrap();
    let alias = first.clone();
    assert_eq!(first.identity(), alias.identity());
    assert_eq!(rt.stats().live_bytes, 8);
    let artifact = rt
        .make_value(
            "prepared",
            ResourceKind::Artifact,
            8,
            || Ok(data(())),
            vec![first.clone()],
        )
        .unwrap();
    drop(first);
    drop(alias);
    assert_eq!(rt.stats().live_resources, 2);
    let called = Cell::new(false);
    assert_eq!(
        rt.make_value(
            "x",
            ResourceKind::Value,
            1,
            || {
                called.set(true);
                Ok(data(()))
            },
            vec![]
        )
        .unwrap_err()
        .code,
        ErrorCode::BudgetExceeded
    );
    assert!(!called.get());
    drop(artifact);
    assert_eq!(rt.stats().live_bytes, 0);
    let foreign = Runtime::default();
    assert_eq!(
        rt.make_value(
            "x",
            ResourceKind::Value,
            1,
            || Ok(data(())),
            vec![value(&foreign, 1).unwrap()]
        )
        .unwrap_err()
        .code,
        ErrorCode::InvalidRequest
    );
    assert!(rt
        .make_value(
            "x",
            ResourceKind::Value,
            8,
            || Err(Error::invalid("factory")),
            vec![]
        )
        .is_err());
    assert_eq!(rt.stats().live_resources, 0);
}

#[test]
fn complete_capture_runs_once_and_preserves_export_aliases() {
    let rt = Runtime::default();
    let calls = Rc::new(Cell::new(0));
    register(&rt, calls.clone());
    let x = value(&rt, 4).unwrap();
    let out = rt
        .run_scope(strict(), |s| {
            let a = s.input(&x)?;
            assert_eq!(a, s.input(&x)?);
            let b = s.submit("add", vec![a, a], data(()), 1)?[0];
            let c = s.submit("add", vec![a, b], data(()), 1)?[0];
            assert_eq!(calls.get(), 0);
            assert_eq!(s.descriptor(c)?.bytes, 8);
            Ok(vec![c, b, c])
        })
        .unwrap();
    assert_eq!(out.iter().map(integer).collect::<Vec<_>>(), vec![12, 8, 12]);
    assert_eq!(out[0].identity(), out[2].identity());
    assert_eq!(calls.get(), 2);
    assert_eq!(rt.scope_stats().commands, 2);
    assert_eq!(rt.scope_stats().executed_commands, 2);
    assert_eq!(rt.scope_stats().imports, 1);
    drop(out);
    drop(x);
    assert_eq!(rt.stats().live_bytes, 0);
}

#[test]
fn late_contract_error_prevents_all_evaluation_and_restores_runtime() {
    let rt = Runtime::default();
    let calls = Rc::new(Cell::new(0));
    register(&rt, calls.clone());
    let x = value(&rt, 2).unwrap();
    let error = rt
        .run_scope(strict(), |s| {
            let a = s.input(&x)?;
            let b = s.submit("add", vec![a, a], data(()), 1)?[0];
            s.submit("invalid", vec![b], data(()), 1)
        })
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidRequest);
    assert_eq!(calls.get(), 0);
    assert_eq!(rt.scope_stats().commands, 1);
    assert!(value(&rt, 1).is_ok());
}

#[test]
fn capture_and_metadata_guards_reject_evaluation_and_registration() {
    let rt = Runtime::default();
    let calls = Rc::new(Cell::new(0));
    register(&rt, calls);
    rt.run_scope(strict(), |_| {
        assert_eq!(value(&rt, 1).unwrap_err().code, ErrorCode::InvalidRequest);
        assert!(rt.submit(Request::new("add", vec![], data(()))).is_err());
        assert!(rt
            .add_recipe(Recipe::new("noop", "noop", |_, _| Ok(vec![])))
            .is_err());
        assert!(rt.run_scope(strict(), |_| Ok(vec![])).is_err());
        Ok(vec![])
    })
    .unwrap();
    assert!(value(&rt, 1).is_ok());
    let result = rt.run_scope(strict(), |_| -> Result<Vec<ScopeValue>> {
        panic!("capture panic")
    });
    assert_eq!(result.unwrap_err().code, ErrorCode::ExecutionFailed);
    assert!(value(&rt, 1).is_ok());
}

#[test]
fn foreign_scope_symbol_and_capture_caps_rejected() {
    let rt = Runtime::default();
    let x = value(&rt, 2).unwrap();
    let mut stale = ScopeValue::default();
    rt.inspect_scope(
        strict(),
        |s| {
            stale = s.input(&x)?;
            Ok(vec![stale])
        },
        |_| Ok(()),
    )
    .unwrap();
    assert!(rt.run_scope(strict(), |_| Ok(vec![stale])).is_err());
    let options = ScopeOptions {
        max_input_edges: 1,
        ..strict()
    };
    assert_eq!(
        rt.run_scope(options, |s| {
            let a = s.input(&x)?;
            s.submit("add", vec![a, a], data(()), 1)
        })
        .unwrap_err()
        .code,
        ErrorCode::BudgetExceeded
    );
    assert_eq!(rt.scope_stats().commands, 0);
}

#[test]
fn inspect_exposes_descriptors_and_liveouts_without_dispatch() {
    let rt = Runtime::default();
    let calls = Rc::new(Cell::new(0));
    register(&rt, calls.clone());
    let x = value(&rt, 3).unwrap();
    rt.inspect_scope(
        strict(),
        |s| {
            let a = s.input(&x)?;
            s.submit("add", vec![a, a], data(()), 1)
        },
        |v| {
            assert_eq!(v.len(), 1);
            let cmd = v.command(0)?;
            assert_eq!(v.live_ins(1)?, vec![cmd.inputs[0]]);
            assert_eq!(v.live_outs(1)?, cmd.outputs);
            assert!(v.find(cmd.outputs[0])?.is_none());
            assert_eq!(v.descriptor(cmd.outputs[0])?.bytes, 8);
            assert!(value(&rt, 0).is_err());
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(calls.get(), 0);
}

#[test]
fn fused_span_preserves_intermediate_liveout_and_later_use() {
    let rt = Runtime::default();
    let calls = Rc::new(Cell::new(0));
    register(&rt, calls.clone());
    rt.add_scope_kernel(ScopeKernel::new(
        "two-adds",
        2,
        |v| {
            assert!(value(v.runtime(), 1).is_err());
            Ok(if v.len() == 2 {
                ScopeMatch::accept(2)
            } else {
                ScopeMatch::decline("short")
            })
        },
        |rt, v| {
            let first = v.command(0)?;
            let a = integer(v.value(first.inputs[0])?);
            let b = value(rt, a + a)?;
            let c = value(rt, a + integer(&b))?;
            Ok(vec![b, c])
        },
    ))
    .unwrap();
    let x = value(&rt, 2).unwrap();
    let out = rt
        .run_scope(strict(), |s| {
            let a = s.input(&x)?;
            let b = s.submit("add", vec![a, a], data(()), 1)?[0];
            let c = s.submit("add", vec![a, b], data(()), 1)?[0];
            let d = s.submit("add", vec![b, c], data(()), 1)?[0];
            Ok(vec![b, d])
        })
        .unwrap();
    assert_eq!(out.iter().map(integer).collect::<Vec<_>>(), vec![4, 10]);
    assert_eq!(calls.get(), 1);
    assert_eq!(rt.scope_stats().kernel_spans, 1);
    assert_eq!(rt.scope_stats().executed_commands, 3);
}

#[test]
fn invalid_output_descriptor_is_not_counted_as_executed() {
    let rt = Runtime::default();
    rt.add_recipe(Recipe::new("add", "wrong", |rt, _| {
        Ok(vec![rt.make_value(
            "wrong",
            ResourceKind::Value,
            8,
            || Ok(data(1i64)),
            vec![],
        )?])
    }))
    .unwrap();
    let x = value(&rt, 1).unwrap();
    assert_eq!(
        rt.run_scope(strict(), |s| {
            let a = s.input(&x)?;
            s.submit("add", vec![a, a], data(()), 1)
        })
        .unwrap_err()
        .code,
        ErrorCode::InvalidRequest
    );
    assert_eq!(rt.scope_stats().executed_commands, 0);
    assert_eq!(rt.stats().live_resources, 1);
}

#[test]
fn deadline_includes_capture_and_final_span() {
    let rt = Runtime::default();
    let options = ScopeOptions {
        max_duration_ms: 1,
        ..strict()
    };
    assert_eq!(
        rt.run_scope(options, |_| {
            std::thread::sleep(Duration::from_millis(4));
            Ok(vec![])
        })
        .unwrap_err()
        .code,
        ErrorCode::BudgetExceeded
    );
    rt.add_recipe(Recipe::new("add", "slow", |rt, _| {
        std::thread::sleep(Duration::from_millis(4));
        Ok(vec![value(rt, 2)?])
    }))
    .unwrap();
    let x = value(&rt, 1).unwrap();
    let options = ScopeOptions {
        max_duration_ms: 2,
        ..strict()
    };
    assert_eq!(
        rt.run_scope(options, |s| {
            let a = s.input(&x)?;
            s.submit("add", vec![a, a], data(()), 1)
        })
        .unwrap_err()
        .code,
        ErrorCode::BudgetExceeded
    );
    assert_eq!(rt.scope_stats().executed_commands, 1);
}

#[test]
fn large_graph_releases_intermediates_at_last_use() {
    let rt = Runtime::new(RuntimeOptions {
        max_live_bytes: 24,
        max_trace_events: 0,
    });
    let calls = Rc::new(Cell::new(0));
    register(&rt, calls.clone());
    let one = value(&rt, 1).unwrap();
    let options = ScopeOptions {
        max_commands: 100_000,
        max_values: 100_001,
        max_input_edges: 200_000,
        use_kernels: false,
        ..strict()
    };
    let out = rt
        .run_scope(options, |s| {
            let initial = s.input(&one)?;
            let mut current = initial;
            for _ in 0..100_000 {
                current = s.submit("add", vec![initial, current], data(()), 1)?[0];
            }
            Ok(vec![current])
        })
        .unwrap();
    assert_eq!(integer(&out[0]), 100_001);
    assert_eq!(calls.get(), 100_000);
    assert_eq!(rt.stats().peak_live_bytes, 24);
    drop(out);
    drop(one);
    assert_eq!(rt.stats().live_resources, 0);
}
