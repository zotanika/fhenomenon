mod support;
use fhenomenon::*;
use std::{cell::Cell, rc::Rc};
use support::*;

/// Language-neutral results are hand-specified in one fixture, never generated
/// by either implementation. A failing case names its graph for both runners.
#[test]
fn common_runtime_contract_cases() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../contracts/runtime-v1.json")).unwrap();
    assert_eq!(fixture["version"], 1);
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let rt = Runtime::new(RuntimeOptions {
            max_live_bytes: case["runtime_bytes"].as_u64().unwrap_or(0) as usize,
            ..Default::default()
        });
        let calls = Rc::new(Cell::new(0));
        register(&rt, calls, case["bad_output"].as_bool().unwrap_or(false));
        match case["kernel"].as_str() {
            Some("two_adds") => rt.add_scope_kernel(two_adds()).unwrap(),
            Some("decline") => rt
                .add_scope_kernel(ScopeKernel::new(
                    "decline",
                    2,
                    |_| Ok(ScopeMatch::decline("test")),
                    |_, _| unreachable!(),
                ))
                .unwrap(),
            _ => {}
        }
        let inputs = case["inputs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| value(&rt, n.as_i64().unwrap()).unwrap())
            .collect::<Vec<_>>();
        let mut options = strict();
        if let Some(caps) = case["caps"].as_object() {
            for (cap, limit) in caps {
                let limit = limit.as_u64().unwrap() as usize;
                match cap.as_str() {
                    "max_commands" => options.max_commands = limit,
                    "max_values" => options.max_values = limit,
                    "max_input_edges" => options.max_input_edges = limit,
                    "max_exports" => options.max_exports = limit,
                    "max_lookahead" => options.max_lookahead = limit,
                    "max_match_attempts" => options.max_match_attempts = limit,
                    _ => panic!("unknown fixture cap: {cap}"),
                }
            }
        }
        options.use_kernels = case["use_kernels"].as_bool().unwrap_or(true);
        let result = rt.run_scope(options, |scope| {
            let mut symbols = inputs
                .iter()
                .map(|v| scope.input(v))
                .collect::<Result<Vec<_>>>()?;
            if case["repeat_first_import"].as_bool().unwrap_or(false) {
                assert_eq!(symbols[0], scope.input(&inputs[0])?);
            }
            for c in case["commands"].as_array().unwrap() {
                let args = c["inputs"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|i| symbols[i.as_u64().unwrap() as usize])
                    .collect();
                symbols.extend(scope.submit(
                    c["op"].as_str().unwrap(),
                    args,
                    data(()),
                    c["outputs"].as_u64().unwrap() as usize,
                )?);
            }
            Ok(case["exports"]
                .as_array()
                .unwrap()
                .iter()
                .map(|i| symbols[i.as_u64().unwrap() as usize])
                .collect())
        });
        if let Some(error) = case["error"].as_str() {
            assert_eq!(format!("{:?}", result.unwrap_err().code), error, "{name}");
        } else {
            let output = result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let expected = case["expected"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_i64().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                output.iter().map(integer).collect::<Vec<_>>(),
                expected,
                "{name}"
            );
            for alias in case["aliases"].as_array().into_iter().flatten() {
                assert_eq!(
                    output[alias[0].as_u64().unwrap() as usize].identity(),
                    output[alias[1].as_u64().unwrap() as usize].identity(),
                    "{name}"
                );
            }
        }
        let s = rt.scope_stats();
        let actual = serde_json::json!({"commands":s.commands,"values":s.values,"imports":s.imports,"input_edges":s.input_edges,"exports":s.exports,"executed_commands":s.executed_commands,"recipe_spans":s.recipe_spans,"kernel_spans":s.kernel_spans});
        assert_eq!(actual, case["stats"], "{name}");
        assert_eq!(
            rt.stats().live_resources,
            inputs.len(),
            "{name}: unpublished intermediates must be released"
        );
    }
}
