#![allow(dead_code)]
use fhenomenon::*;
use std::{cell::Cell, rc::Rc};

pub fn value(rt: &Runtime, n: i64) -> Result<Value> {
    rt.make_value("integer", ResourceKind::Value, 8, || Ok(data(n)), vec![])
}
pub fn integer(value: &Value) -> i64 {
    *value.payload::<i64>().unwrap()
}
pub struct IntegerContract;
impl ScopeContract for IntegerContract {
    fn describe(&self, v: &Value) -> Result<ScopeDescriptor> {
        v.payload::<i64>()?;
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
        let (arity, outputs) = match op {
            "add" => (2, 1),
            "split" => (1, 2),
            "observe" => (1, 0),
            "missing" => (1, 1),
            _ => return Err(Error::invalid("invalid integer operation")),
        };
        if inputs.len() != arity {
            return Err(Error::invalid("incorrect integer arity"));
        }
        Ok((0..outputs)
            .map(|_| ScopeDescriptor::new("integer", ResourceKind::Value, 8, data(())))
            .collect())
    }
    fn validate(&self, value: &Value, _: &ScopeDescriptor) -> Result<()> {
        value.payload::<i64>().map(|_| ())
    }
}
pub fn strict() -> ScopeOptions {
    ScopeOptions {
        contract: Some(Rc::new(IntegerContract)),
        ..Default::default()
    }
}
pub fn register(rt: &Runtime, calls: Rc<Cell<usize>>, bad_output: bool) {
    let add_calls = calls.clone();
    rt.add_recipe(Recipe::new("add", "integer.add", move |rt, r| {
        add_calls.set(add_calls.get() + 1);
        let n = integer(&r.inputs[0]) + integer(&r.inputs[1]);
        if bad_output {
            return Ok(vec![rt.make_value(
                "wrong",
                ResourceKind::Value,
                8,
                || Ok(data(n)),
                vec![],
            )?]);
        }
        Ok(vec![value(rt, n)?])
    }))
    .unwrap();
    let split_calls = calls.clone();
    rt.add_recipe(Recipe::new("split", "integer.split", move |rt, r| {
        split_calls.set(split_calls.get() + 1);
        let n = integer(&r.inputs[0]);
        Ok(vec![value(rt, n)?, value(rt, -n)?])
    }))
    .unwrap();
    rt.add_recipe(Recipe::new("observe", "integer.observe", move |_, _| {
        calls.set(calls.get() + 1);
        Ok(vec![])
    }))
    .unwrap();
}
pub fn two_adds() -> ScopeKernel {
    ScopeKernel::new(
        "two-adds",
        2,
        |v| {
            Ok(
                if v.len() == 2
                    && v.command(0)?.operation == "add"
                    && v.command(1)?.operation == "add"
                {
                    ScopeMatch::accept(2)
                } else {
                    ScopeMatch::decline("requires two adds")
                },
            )
        },
        |rt, v| {
            let mut results = std::collections::HashMap::new();
            for offset in 0..v.len() {
                let command = v.command(offset)?;
                let number = |symbol: ScopeValue| -> Result<i64> {
                    results
                        .get(&symbol)
                        .copied()
                        .map(Ok)
                        .unwrap_or_else(|| v.value(symbol).map(integer))
                };
                results.insert(
                    command.outputs[0],
                    number(command.inputs[0])? + number(command.inputs[1])?,
                );
            }
            v.live_outs(v.len())?
                .iter()
                .map(|s| value(rt, results[s]))
                .collect()
        },
    )
}
