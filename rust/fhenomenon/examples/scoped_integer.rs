use fhenomenon::{data, Recipe, ResourceKind, Runtime, ScopeOptions};

fn main() -> fhenomenon::Result<()> {
    let runtime = Runtime::default();
    runtime.add_recipe(Recipe::new("double", "integer.double", |rt, request| {
        let result = 2 * request.inputs[0].payload::<i64>()?;
        Ok(vec![rt.make_value(
            "integer",
            ResourceKind::Value,
            8,
            || Ok(data(result)),
            vec![],
        )?])
    }))?;
    let input = runtime.make_value(
        "integer",
        ResourceKind::Value,
        8,
        || Ok(data(3_i64)),
        vec![],
    )?;
    let output = runtime.run_scope(ScopeOptions::default(), |scope| {
        let x = scope.input(&input)?;
        scope.submit("double", vec![x], data(()), 1)
    })?;
    assert_eq!(*output[0].payload::<i64>()?, 6);
    println!("result: {}", output[0].payload::<i64>()?);
    println!("scope execution: {:?}", runtime.scope_stats());
    Ok(())
}
