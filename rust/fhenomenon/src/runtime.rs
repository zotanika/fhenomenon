use crate::{
    error::callback,
    value::{identity, Accounting, Charge, Resource},
    *,
};
use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
    rc::Rc,
};

#[derive(Clone, Copy, Debug)]
pub struct RuntimeOptions {
    pub max_live_bytes: usize,
    pub max_trace_events: usize,
}
impl Default for RuntimeOptions {
    fn default() -> Self {
        Self {
            max_live_bytes: 0,
            max_trace_events: 256,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuntimeStats {
    pub live_bytes: usize,
    pub peak_live_bytes: usize,
    pub live_resources: usize,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScopeStats {
    pub commands: usize,
    pub values: usize,
    pub imports: usize,
    pub input_edges: usize,
    pub exports: usize,
    pub executed_commands: usize,
    pub recipe_spans: usize,
    pub kernel_spans: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceKind {
    Declined,
    Selected,
    Completed,
    Failed,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceEvent {
    pub kind: TraceKind,
    pub operation: String,
    pub recipe: String,
    pub detail: String,
}
pub struct Request {
    pub operation: String,
    pub inputs: Values,
    pub attributes: Data,
}
impl Request {
    pub fn new(operation: impl Into<String>, inputs: Values, attributes: Data) -> Self {
        Self {
            operation: operation.into(),
            inputs,
            attributes,
        }
    }
}
#[derive(Clone, Debug)]
pub enum Acceptance {
    Accept,
    Decline(String),
}
pub type RecipeAccept = dyn Fn(&Request) -> Result<Acceptance>;
pub type RecipeExecute = dyn Fn(&Runtime, &Request) -> Result<Values>;
pub struct Recipe {
    pub operation: String,
    pub name: String,
    pub priority: i32,
    pub accepts: Rc<RecipeAccept>,
    pub execute: Rc<RecipeExecute>,
}
impl Recipe {
    pub fn new(
        operation: impl Into<String>,
        name: impl Into<String>,
        execute: impl Fn(&Runtime, &Request) -> Result<Values> + 'static,
    ) -> Self {
        Self {
            operation: operation.into(),
            name: name.into(),
            priority: 0,
            accepts: Rc::new(|_| Ok(Acceptance::Accept)),
            execute: Rc::new(execute),
        }
    }
}
pub(crate) struct FlagGuard<'a> {
    flag: &'a Cell<bool>,
    previous: bool,
}
impl Drop for FlagGuard<'_> {
    fn drop(&mut self) {
        self.flag.set(self.previous);
    }
}
pub(crate) fn guard(flag: &Cell<bool>) -> FlagGuard<'_> {
    FlagGuard {
        previous: flag.replace(true),
        flag,
    }
}

/// Synchronous, single-threaded orchestration; handles can outlive the runtime.
pub struct Runtime {
    accounting: Rc<Accounting>,
    recipes: RefCell<Vec<Rc<Recipe>>>,
    kernels: RefCell<Vec<Rc<ScopeKernel>>>,
    events: RefCell<VecDeque<TraceEvent>>,
    pub(crate) running: Cell<bool>,
    pub(crate) capturing: Cell<bool>,
    pub(crate) matching: Cell<bool>,
    pub(crate) inspecting: Cell<bool>,
    pub(crate) strict: Cell<bool>,
    pub(crate) metadata: Cell<bool>,
    pub(crate) progress: RefCell<ScopeStats>,
}
impl Default for Runtime {
    fn default() -> Self {
        Self::new(RuntimeOptions::default())
    }
}
impl Runtime {
    pub fn new(options: RuntimeOptions) -> Self {
        Self {
            accounting: Rc::new(Accounting {
                options,
                stats: RefCell::new(RuntimeStats::default()),
            }),
            recipes: RefCell::new(vec![]),
            kernels: RefCell::new(vec![]),
            events: RefCell::new(VecDeque::new()),
            running: Cell::new(false),
            capturing: Cell::new(false),
            matching: Cell::new(false),
            inspecting: Cell::new(false),
            strict: Cell::new(false),
            metadata: Cell::new(false),
            progress: RefCell::new(ScopeStats::default()),
        }
    }
    pub fn owns(&self, value: &Value) -> bool {
        Rc::ptr_eq(&self.accounting, &value.0.charge.owner)
    }
    pub fn stats(&self) -> RuntimeStats {
        *self.accounting.stats.borrow()
    }
    pub fn scope_stats(&self) -> ScopeStats {
        *self.progress.borrow()
    }
    pub fn trace(&self) -> Vec<TraceEvent> {
        self.events.borrow().iter().cloned().collect()
    }
    pub(crate) fn record(
        &self,
        kind: TraceKind,
        operation: &str,
        recipe: &str,
        detail: impl Into<String>,
    ) {
        let limit = self.accounting.options.max_trace_events;
        if limit == 0 {
            return;
        }
        let mut events = self.events.borrow_mut();
        if events.len() == limit {
            events.pop_front();
        }
        events.push_back(TraceEvent {
            kind,
            operation: operation.into(),
            recipe: recipe.into(),
            detail: detail.into(),
        });
    }
    /// Check callback restrictions before provider data-plane work such as key
    /// preparation or trusted-client decryption. This does not authorize a trust
    /// boundary crossing; the caller must already own that capability.
    pub fn check_evaluation_allowed(&self) -> Result<()> {
        if self.capturing.get()
            || self.matching.get()
            || self.inspecting.get()
            || self.metadata.get()
        {
            return Err(Error::invalid(
                "evaluation forbidden during capture, matching, inspection or metadata callbacks",
            ));
        }
        Ok(())
    }
    fn registration_allowed(&self) -> Result<()> {
        if self.inspecting.get() || self.strict.get() {
            return Err(Error::invalid(
                "registration frozen during inspection or contracted scope",
            ));
        }
        Ok(())
    }
    pub fn make_value(
        &self,
        format: impl Into<String>,
        kind: ResourceKind,
        bytes: usize,
        factory: impl FnOnce() -> Result<Data>,
        dependencies: Values,
    ) -> Result<Value> {
        self.check_evaluation_allowed()?;
        if dependencies.iter().any(|v| !self.owns(v)) {
            return Err(Error::invalid("dependency belongs to another runtime"));
        }
        let charge = Charge::new(self.accounting.clone(), bytes)?;
        let payload = callback(factory)?;
        Ok(Value(Rc::new(Resource {
            payload,
            dependencies,
            charge,
            id: identity()?,
            format: format.into(),
            kind,
        })))
    }
    pub fn add_recipe(&self, recipe: Recipe) -> Result<()> {
        self.registration_allowed()?;
        if recipe.operation.is_empty() || recipe.name.is_empty() {
            return Err(Error::invalid("recipe requires operation and name"));
        }
        let mut recipes = self.recipes.borrow_mut();
        let pos = recipes
            .iter()
            .position(|r| r.priority < recipe.priority)
            .unwrap_or(recipes.len());
        recipes.insert(pos, Rc::new(recipe));
        Ok(())
    }
    pub fn add_scope_kernel(&self, kernel: ScopeKernel) -> Result<()> {
        self.registration_allowed()?;
        if kernel.name.is_empty() || kernel.max_commands == 0 {
            return Err(Error::invalid(
                "kernel requires a name and positive command cap",
            ));
        }
        let mut kernels = self.kernels.borrow_mut();
        let pos = kernels
            .iter()
            .position(|k| k.priority < kernel.priority)
            .unwrap_or(kernels.len());
        kernels.insert(pos, Rc::new(kernel));
        Ok(())
    }
    pub fn submit(&self, request: Request) -> Result<Values> {
        self.check_evaluation_allowed()?;
        if request.operation.is_empty() || request.inputs.iter().any(|v| !self.owns(v)) {
            return Err(Error::invalid("invalid request or foreign runtime value"));
        }
        let recipes = self.recipes.borrow().clone();
        for recipe in recipes.iter().filter(|r| r.operation == request.operation) {
            let result = callback(|| match (recipe.accepts)(&request)? {
                Acceptance::Decline(reason) => {
                    self.record(
                        TraceKind::Declined,
                        &request.operation,
                        &recipe.name,
                        reason,
                    );
                    Ok(None)
                }
                Acceptance::Accept => {
                    self.record(TraceKind::Selected, &request.operation, &recipe.name, "");
                    let output = (recipe.execute)(self, &request)?;
                    if output.iter().any(|v| !self.owns(v)) {
                        return Err(Error::invalid("recipe output belongs to another runtime"));
                    }
                    self.record(TraceKind::Completed, &request.operation, &recipe.name, "");
                    Ok(Some(output))
                }
            });
            match result {
                Ok(Some(output)) => return Ok(output),
                Ok(None) => {}
                Err(e) => {
                    self.record(
                        TraceKind::Failed,
                        &request.operation,
                        &recipe.name,
                        &e.message,
                    );
                    return Err(e);
                }
            }
        }
        self.record(
            TraceKind::Failed,
            &request.operation,
            "",
            "no applicable recipe",
        );
        Err(Error::new(
            ErrorCode::Unsupported,
            format!("no applicable recipe for {}", request.operation),
        ))
    }
    pub fn run_scope(
        &self,
        options: ScopeOptions,
        capture: impl FnOnce(&mut Scope<'_>) -> Result<Vec<ScopeValue>>,
    ) -> Result<Values> {
        if self.running.get() {
            return Err(Error::invalid("scopes cannot nest on the same runtime"));
        }
        let _running = guard(&self.running);
        let _strict = options.contract.as_ref().map(|_| guard(&self.strict));
        *self.progress.borrow_mut() = ScopeStats::default();
        let kernels = self.kernels.borrow().clone();
        callback(|| {
            let mut scope = Scope::new(self, options)?;
            let exports = {
                let _capture = guard(&self.capturing);
                capture(&mut scope)?
            };
            scope.execute(&exports, &kernels)
        })
    }
    pub fn inspect_scope(
        &self,
        options: ScopeOptions,
        capture: impl FnOnce(&mut Scope<'_>) -> Result<Vec<ScopeValue>>,
        inspect: impl FnOnce(&ScopeView<'_, '_>) -> Result<()>,
    ) -> Result<()> {
        if self.running.get() {
            return Err(Error::invalid("scopes cannot nest on the same runtime"));
        }
        let _running = guard(&self.running);
        let _inspecting = guard(&self.inspecting);
        let _strict = options.contract.as_ref().map(|_| guard(&self.strict));
        *self.progress.borrow_mut() = ScopeStats::default();
        callback(|| {
            let mut scope = Scope::new(self, options)?;
            let exports = {
                let _capture = guard(&self.capturing);
                capture(&mut scope)?
            };
            scope.finish_capture(&exports)?;
            inspect(&scope.view())?;
            scope.check_deadline()
        })
    }
}
