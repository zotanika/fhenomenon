mod support;
use fhenomenon::*;
use std::{
    cell::{Cell, RefCell},
    rc::{Rc, Weak},
};
use support::*;

#[test]
fn priority_declines_and_registration_order_select_one_recipe() {
    let rt = Runtime::new(RuntimeOptions {
        max_trace_events: 3,
        ..Default::default()
    });
    let mut low = Recipe::new("choose", "low", |rt, _| Ok(vec![value(rt, 1)?]));
    low.priority = -1;
    rt.add_recipe(low).unwrap();
    for (name, n) in [("first", 2), ("second", 3)] {
        let mut r = Recipe::new("choose", name, move |rt, _| Ok(vec![value(rt, n)?]));
        r.priority = 5;
        rt.add_recipe(r).unwrap();
    }
    let mut decline = Recipe::new("choose", "decline", |_, _| unreachable!());
    decline.priority = 9;
    decline.accepts = Rc::new(|_| Ok(Acceptance::Decline("unsupported layout".into())));
    rt.add_recipe(decline).unwrap();
    let out = rt.submit(Request::new("choose", vec![], data(()))).unwrap();
    assert_eq!(integer(&out[0]), 2);
    let trace = rt.trace();
    assert_eq!(
        trace.iter().map(|e| e.kind).collect::<Vec<_>>(),
        vec![
            TraceKind::Declined,
            TraceKind::Selected,
            TraceKind::Completed
        ]
    );
    assert_eq!(trace[0].detail, "unsupported layout");
    assert_eq!(trace[2].recipe, "first");
    rt.submit(Request::new("choose", vec![], data(()))).unwrap();
    assert_eq!(rt.trace().len(), 3);
}

#[test]
fn recipe_registration_during_acceptance_changes_only_future_dispatch() {
    let rt = Rc::new(Runtime::default());
    let weak = Rc::downgrade(&rt);
    let once = Cell::new(false);
    let mut registering = Recipe::new("choose", "registering", |_, _| unreachable!());
    registering.accepts = Rc::new(move |_| {
        if !once.replace(true) {
            let mut better = Recipe::new("choose", "future", |rt, _| Ok(vec![value(rt, 8)?]));
            better.priority = 10;
            weak.upgrade().unwrap().add_recipe(better)?;
        }
        Ok(Acceptance::Decline("registered future recipe".into()))
    });
    rt.add_recipe(registering).unwrap();
    rt.add_recipe(Recipe::new("choose", "fallback", |rt, _| {
        Ok(vec![value(rt, 3)?])
    }))
    .unwrap();
    let invoke = || integer(&rt.submit(Request::new("choose", vec![], data(()))).unwrap()[0]);
    assert_eq!(invoke(), 3);
    assert_eq!(invoke(), 8);
}

#[test]
fn accepted_recipe_failure_does_not_fall_back_and_releases_partial_allocations() {
    let rt = Runtime::default();
    rt.add_recipe(Recipe::new("fail", "first", |rt, _| {
        let _temporary = value(rt, 3)?;
        Err(Error::execution("provider failed"))
    }))
    .unwrap();
    rt.add_recipe(Recipe::new("fail", "second", |rt, _| {
        Ok(vec![value(rt, 99)?])
    }))
    .unwrap();
    assert_eq!(
        rt.submit(Request::new("fail", vec![], data(())))
            .unwrap_err()
            .code,
        ErrorCode::ExecutionFailed
    );
    assert_eq!(rt.stats().live_resources, 0);
    assert_eq!(
        rt.trace().iter().map(|e| e.kind).collect::<Vec<_>>(),
        vec![TraceKind::Selected, TraceKind::Failed]
    );
}

#[test]
fn scope_kernel_registry_is_snapshotted_before_capture() {
    let rt = Runtime::default();
    register(&rt, Rc::new(Cell::new(0)), false);
    let input = value(&rt, 2).unwrap();
    let graph = |s: &mut Scope<'_>| {
        let a = s.input(&input)?;
        let b = s.submit("add", vec![a, a], data(()), 1)?[0];
        s.submit("add", vec![b, a], data(()), 1)
    };
    rt.run_scope(ScopeOptions::default(), |s| {
        rt.add_scope_kernel(two_adds())?;
        graph(s)
    })
    .unwrap();
    assert_eq!(rt.scope_stats().kernel_spans, 0);
    rt.run_scope(strict(), graph).unwrap();
    assert_eq!(rt.scope_stats().kernel_spans, 1);
}

#[test]
fn kernel_priority_and_declines_preserve_tie_order() {
    let rt = Runtime::default();
    register(&rt, Rc::new(Cell::new(0)), false);
    let mut first = two_adds();
    first.name = "first".into();
    first.priority = 1;
    let mut second = two_adds();
    second.name = "second".into();
    second.priority = 1;
    rt.add_scope_kernel(first).unwrap();
    rt.add_scope_kernel(second).unwrap();
    let mut declining = ScopeKernel::new(
        "higher",
        2,
        |_| Ok(ScopeMatch::decline("shape")),
        |_, _| unreachable!(),
    );
    declining.priority = 5;
    rt.add_scope_kernel(declining).unwrap();
    let x = value(&rt, 2).unwrap();
    let out = rt
        .run_scope(strict(), |s| {
            let a = s.input(&x)?;
            let b = s.submit("add", vec![a, a], data(()), 1)?[0];
            s.submit("add", vec![b, a], data(()), 1)
        })
        .unwrap();
    assert_eq!(integer(&out[0]), 6);
    assert_eq!(
        rt.trace()
            .iter()
            .map(|e| (e.kind, e.recipe.as_str()))
            .collect::<Vec<_>>(),
        vec![
            (TraceKind::Declined, "higher"),
            (TraceKind::Selected, "first"),
            (TraceKind::Completed, "first")
        ]
    );
}

#[test]
fn metadata_callbacks_cannot_dispatch_allocate_register_or_start_nested_scope() {
    struct Guarded {
        runtime: Weak<Runtime>,
        visited: Rc<RefCell<Vec<&'static str>>>,
    }
    impl Guarded {
        fn check(&self, phase: &'static str) {
            self.visited.borrow_mut().push(phase);
            let rt = self.runtime.upgrade().unwrap();
            assert_eq!(value(&rt, 0).unwrap_err().code, ErrorCode::InvalidRequest);
            assert_eq!(
                rt.submit(Request::new("add", vec![], data(())))
                    .unwrap_err()
                    .code,
                ErrorCode::InvalidRequest
            );
            assert_eq!(
                rt.add_recipe(Recipe::new("x", "x", |_, _| Ok(vec![])))
                    .unwrap_err()
                    .code,
                ErrorCode::InvalidRequest
            );
            assert_eq!(
                rt.add_scope_kernel(two_adds()).unwrap_err().code,
                ErrorCode::InvalidRequest
            );
            assert_eq!(
                rt.run_scope(ScopeOptions::default(), |_| Ok(vec![]))
                    .unwrap_err()
                    .code,
                ErrorCode::InvalidRequest
            );
        }
    }
    impl ScopeContract for Guarded {
        fn describe(&self, v: &Value) -> Result<ScopeDescriptor> {
            self.check("describe");
            IntegerContract.describe(v)
        }
        fn infer(
            &self,
            op: &str,
            input: &[ScopeDescriptor],
            attr: &Data,
        ) -> Result<Vec<ScopeDescriptor>> {
            self.check("infer");
            IntegerContract.infer(op, input, attr)
        }
        fn validate(&self, v: &Value, d: &ScopeDescriptor) -> Result<()> {
            self.check("validate");
            IntegerContract.validate(v, d)
        }
    }
    let rt = Rc::new(Runtime::default());
    register(&rt, Rc::new(Cell::new(0)), false);
    let visited = Rc::new(RefCell::new(vec![]));
    let options = ScopeOptions {
        contract: Some(Rc::new(Guarded {
            runtime: Rc::downgrade(&rt),
            visited: visited.clone(),
        })),
        ..Default::default()
    };
    let x = value(&rt, 2).unwrap();
    let out = rt
        .run_scope(options, |s| {
            let a = s.input(&x)?;
            s.submit("add", vec![a, a], data(()), 1)
        })
        .unwrap();
    assert_eq!(integer(&out[0]), 4);
    assert_eq!(*visited.borrow(), vec!["describe", "infer", "validate"]);
    assert_eq!(rt.scope_stats().executed_commands, 1);
    assert!(rt.add_scope_kernel(two_adds()).is_ok());
}

#[test]
fn fused_outputs_are_validated_before_any_publication() {
    let rt = Runtime::default();
    register(&rt, Rc::new(Cell::new(0)), false);
    let mut bad = two_adds();
    bad.execute = Rc::new(|rt, _| {
        Ok(vec![
            value(rt, 4)?,
            rt.make_value("wrong", ResourceKind::Value, 8, || Ok(data(6i64)), vec![])?,
        ])
    });
    rt.add_scope_kernel(bad).unwrap();
    let x = value(&rt, 2).unwrap();
    let error = rt
        .run_scope(strict(), |s| {
            let a = s.input(&x)?;
            let b = s.submit("add", vec![a, a], data(()), 1)?[0];
            let c = s.submit("add", vec![b, a], data(()), 1)?[0];
            Ok(vec![b, c])
        })
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidRequest);
    assert_eq!(rt.scope_stats().executed_commands, 0);
    assert_eq!(rt.scope_stats().kernel_spans, 0);
    assert_eq!(rt.stats().live_resources, 1);
}

#[test]
fn malformed_kernel_span_and_output_count_fail_without_accounting_leaks() {
    for oversized_match in [true, false] {
        let rt = Runtime::default();
        let x = value(&rt, 1).unwrap();
        rt.add_scope_kernel(ScopeKernel::new(
            "broken",
            1,
            move |_| Ok(ScopeMatch::accept(if oversized_match { 2 } else { 1 })),
            |_, _| Ok(vec![]),
        ))
        .unwrap();
        assert_eq!(
            rt.run_scope(strict(), |s| {
                let a = s.input(&x)?;
                s.submit("add", vec![a, a], data(()), 1)
            })
            .unwrap_err()
            .code,
            ErrorCode::InvalidRequest
        );
        assert_eq!(rt.stats().live_resources, 1);
        assert_eq!(rt.scope_stats().executed_commands, 0);
    }
}

#[test]
fn panic_in_matcher_restores_guards_and_factory_panic_refunds_charge() {
    let rt = Runtime::default();
    let x = value(&rt, 1).unwrap();
    register(&rt, Rc::new(Cell::new(0)), false);
    rt.add_scope_kernel(ScopeKernel::new(
        "panic",
        1,
        |_| panic!("matcher panic"),
        |_, _| unreachable!(),
    ))
    .unwrap();
    assert_eq!(
        rt.run_scope(strict(), |s| {
            let a = s.input(&x)?;
            s.submit("add", vec![a, a], data(()), 1)
        })
        .unwrap_err()
        .code,
        ErrorCode::ExecutionFailed
    );
    let options = ScopeOptions {
        use_kernels: false,
        ..strict()
    };
    assert_eq!(
        integer(
            &rt.run_scope(options, |s| {
                let a = s.input(&x)?;
                s.submit("add", vec![a, a], data(()), 1)
            })
            .unwrap()[0]
        ),
        2
    );
    let before = rt.stats();
    assert_eq!(
        rt.make_value(
            "x",
            ResourceKind::Value,
            8,
            || panic!("factory panic"),
            vec![]
        )
        .unwrap_err()
        .code,
        ErrorCode::ExecutionFailed
    );
    assert_eq!(rt.stats().live_bytes, before.live_bytes);
    assert_eq!(rt.stats().live_resources, before.live_resources);
}

#[test]
fn later_span_failure_preserves_completed_progress_and_releases_temporaries() {
    let rt = Runtime::default();
    register(&rt, Rc::new(Cell::new(0)), false);
    let x = value(&rt, 2).unwrap();
    let error = rt
        .run_scope(strict(), |s| {
            let a = s.input(&x)?;
            let b = s.submit("add", vec![a, a], data(()), 1)?[0];
            s.submit("missing", vec![b], data(()), 1)
        })
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::Unsupported);
    assert_eq!(rt.scope_stats().commands, 2);
    assert_eq!(rt.scope_stats().executed_commands, 1);
    assert_eq!(rt.stats().live_resources, 1);
}

#[test]
fn inspection_panic_restores_guards_and_bounded_views_retain_global_liveness() {
    let rt = Runtime::default();
    let x = value(&rt, 2).unwrap();
    let error = rt
        .inspect_scope(
            strict(),
            |s| {
                let a = s.input(&x)?;
                let b = s.submit("add", vec![a, a], data(()), 1)?[0];
                let c = s.submit("add", vec![b, a], data(()), 1)?[0];
                s.submit("add", vec![b, c], data(()), 1)
            },
            |v| {
                let first = v.subview(0, 1)?;
                assert_eq!(first.live_outs(1)?, first.command(0)?.outputs);
                assert!(v.subview(3, 1).is_err());
                assert!(v.command(3).is_err());
                assert!(v.live_ins(4).is_err());
                assert_eq!(
                    rt.add_scope_kernel(two_adds()).unwrap_err().code,
                    ErrorCode::InvalidRequest
                );
                panic!("inspection panic")
            },
        )
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::ExecutionFailed);
    assert_eq!(rt.scope_stats().executed_commands, 0);
    assert!(value(&rt, 3).is_ok());
    assert!(rt.add_scope_kernel(two_adds()).is_ok());
}

#[test]
fn payload_drops_before_dependencies_and_values_outlive_runtime() {
    struct Destruction {
        label: &'static str,
        order: Rc<RefCell<Vec<&'static str>>>,
    }
    impl Drop for Destruction {
        fn drop(&mut self) {
            self.order.borrow_mut().push(self.label);
        }
    }
    let order = Rc::new(RefCell::new(vec![]));
    let dependent = {
        let rt = Runtime::default();
        let original = rt
            .make_value(
                "source",
                ResourceKind::Value,
                8,
                || {
                    Ok(data(Destruction {
                        label: "source",
                        order: order.clone(),
                    }))
                },
                vec![],
            )
            .unwrap();
        rt.make_value(
            "prepared",
            ResourceKind::Artifact,
            8,
            || {
                Ok(data(Destruction {
                    label: "prepared",
                    order: order.clone(),
                }))
            },
            vec![original],
        )
        .unwrap()
    };
    assert!(order.borrow().is_empty());
    drop(dependent);
    assert_eq!(*order.borrow(), vec!["prepared", "source"]);
}

#[test]
fn foreign_recipe_and_kernel_outputs_are_rejected() {
    let rt = Runtime::default();
    let other = Runtime::default();
    let foreign = value(&other, 7).unwrap();
    let returned = foreign.clone();
    rt.add_recipe(Recipe::new("foreign", "foreign", move |_, _| {
        Ok(vec![returned.clone()])
    }))
    .unwrap();
    assert_eq!(
        rt.submit(Request::new("foreign", vec![], data(())))
            .unwrap_err()
            .code,
        ErrorCode::InvalidRequest
    );
    rt.add_scope_kernel(ScopeKernel::new(
        "foreign",
        1,
        |_| Ok(ScopeMatch::accept(1)),
        move |_, _| Ok(vec![foreign.clone()]),
    ))
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
    assert_eq!(rt.stats().live_resources, 1);
}

#[test]
fn provider_data_plane_obeys_runtime_callback_guards() {
    let rt = Runtime::default();
    assert!(rt.check_evaluation_allowed().is_ok());
    rt.run_scope(strict(), |_| {
        assert_eq!(
            rt.check_evaluation_allowed().unwrap_err().code,
            ErrorCode::InvalidRequest
        );
        Ok(vec![])
    })
    .unwrap();
    assert!(rt.check_evaluation_allowed().is_ok());
}
