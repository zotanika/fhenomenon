use crate::{error::callback, runtime::guard, value::identity, *};
use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
    time::Instant,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScopeValue {
    scope: u64,
    id: usize,
}
impl Default for ScopeValue {
    fn default() -> Self {
        Self {
            scope: 0,
            id: usize::MAX,
        }
    }
}
impl ScopeValue {
    pub fn id(self) -> usize {
        self.id
    }
}
#[derive(Clone)]
pub struct ScopeDescriptor {
    pub format: String,
    pub kind: ResourceKind,
    pub bytes: usize,
    pub metadata: Data,
}
impl ScopeDescriptor {
    pub fn new(
        format: impl Into<String>,
        kind: ResourceKind,
        bytes: usize,
        metadata: Data,
    ) -> Self {
        Self {
            format: format.into(),
            kind,
            bytes,
            metadata,
        }
    }
    fn valid(&self) -> Result<()> {
        if self.format.is_empty() {
            Err(Error::invalid("descriptor requires a format"))
        } else {
            Ok(())
        }
    }
    fn validate_base(&self, v: &Value) -> Result<()> {
        self.valid()?;
        if self.format != v.format() || self.kind != v.kind() || self.bytes != v.bytes() {
            Err(Error::invalid(
                "resource differs from declared format, kind or bytes",
            ))
        } else {
            Ok(())
        }
    }
}
/// Metadata-only callbacks. Captured state must remain immutable for the scope.
pub trait ScopeContract {
    fn describe(&self, value: &Value) -> Result<ScopeDescriptor>;
    fn infer(
        &self,
        operation: &str,
        inputs: &[ScopeDescriptor],
        attributes: &Data,
    ) -> Result<Vec<ScopeDescriptor>>;
    fn validate(&self, value: &Value, descriptor: &ScopeDescriptor) -> Result<()>;
}
#[derive(Clone)]
pub struct ScopeOptions {
    pub max_commands: usize,
    pub max_values: usize,
    pub max_input_edges: usize,
    pub max_exports: usize,
    pub max_lookahead: usize,
    pub max_match_attempts: usize,
    pub use_kernels: bool,
    pub contract: Option<Rc<dyn ScopeContract>>,
    pub max_duration_ms: usize,
}
impl Default for ScopeOptions {
    fn default() -> Self {
        Self {
            max_commands: 256,
            max_values: 2048,
            max_input_edges: 4096,
            max_exports: 2048,
            max_lookahead: 32,
            max_match_attempts: 4096,
            use_kernels: true,
            contract: None,
            max_duration_ms: 0,
        }
    }
}
pub struct ScopeCommand {
    pub operation: String,
    pub inputs: Vec<ScopeValue>,
    pub attributes: Data,
    pub outputs: Vec<ScopeValue>,
}
#[derive(Clone, Debug)]
pub struct ScopeMatch {
    pub commands: usize,
    pub reason: String,
}
impl ScopeMatch {
    pub fn accept(commands: usize) -> Self {
        Self {
            commands,
            reason: String::new(),
        }
    }
    pub fn decline(reason: impl Into<String>) -> Self {
        Self {
            commands: 0,
            reason: reason.into(),
        }
    }
}
pub type ScopeMatcher = dyn Fn(&ScopeView<'_, '_>) -> Result<ScopeMatch>;
pub type ScopeExecutor = dyn Fn(&Runtime, &ScopeView<'_, '_>) -> Result<Values>;
pub struct ScopeKernel {
    pub name: String,
    pub priority: i32,
    pub max_commands: usize,
    pub matches: Rc<ScopeMatcher>,
    pub execute: Rc<ScopeExecutor>,
}
impl ScopeKernel {
    pub fn new(
        name: impl Into<String>,
        max_commands: usize,
        matches: impl Fn(&ScopeView<'_, '_>) -> Result<ScopeMatch> + 'static,
        execute: impl Fn(&Runtime, &ScopeView<'_, '_>) -> Result<Values> + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            priority: 0,
            max_commands,
            matches: Rc::new(matches),
            execute: Rc::new(execute),
        }
    }
}
struct Slot {
    concrete: Option<Value>,
    definition: Option<usize>,
    last_use: Option<usize>,
    exported: bool,
    descriptor: Option<ScopeDescriptor>,
}
pub struct Scope<'r> {
    runtime: &'r Runtime,
    options: ScopeOptions,
    identity: u64,
    capturing: bool,
    started: Instant,
    edges: usize,
    commands: Vec<ScopeCommand>,
    slots: Vec<Slot>,
    imports: HashMap<u64, usize>,
}
fn growth(existing: usize, additional: usize, cap: usize, message: &str) -> Result<()> {
    if existing > cap || additional > cap - existing {
        Err(Error::budget(message))
    } else {
        Ok(())
    }
}
impl<'r> Scope<'r> {
    pub(crate) fn new(runtime: &'r Runtime, options: ScopeOptions) -> Result<Self> {
        if [
            options.max_commands,
            options.max_values,
            options.max_input_edges,
            options.max_exports,
            options.max_lookahead,
            options.max_match_attempts,
        ]
        .contains(&0)
        {
            return Err(Error::invalid("scope numeric caps must be positive"));
        }
        Ok(Self {
            runtime,
            options,
            identity: identity()?,
            capturing: true,
            started: Instant::now(),
            edges: 0,
            commands: vec![],
            slots: vec![],
            imports: HashMap::new(),
        })
    }
    pub fn belongs_to(&self, runtime: &Runtime) -> bool {
        std::ptr::eq(self.runtime, runtime)
    }
    fn validate(&self, symbol: ScopeValue) -> Result<()> {
        if symbol.scope != self.identity || symbol.id >= self.slots.len() {
            Err(Error::invalid("invalid symbol or foreign scope"))
        } else {
            Ok(())
        }
    }
    fn recording(&self) -> Result<()> {
        if !self.capturing || self.runtime.metadata.get() {
            Err(Error::invalid(
                "recording requires active capture outside metadata callbacks",
            ))
        } else {
            Ok(())
        }
    }
    pub fn input(&mut self, value: &Value) -> Result<ScopeValue> {
        self.recording()?;
        if !self.runtime.owns(value) {
            return Err(Error::invalid("scope input belongs to another runtime"));
        }
        if let Some(&id) = self.imports.get(&value.identity()) {
            return Ok(ScopeValue {
                scope: self.identity,
                id,
            });
        }
        growth(
            self.slots.len(),
            1,
            self.options.max_values,
            "scope value cap exceeded",
        )?;
        let descriptor = if let Some(contract) = &self.options.contract {
            let _metadata = guard(&self.runtime.metadata);
            let descriptor = callback(|| contract.describe(value))?;
            descriptor.validate_base(value)?;
            Some(descriptor)
        } else {
            None
        };
        let id = self.slots.len();
        self.slots.push(Slot {
            concrete: Some(value.clone()),
            definition: None,
            last_use: None,
            exported: false,
            descriptor,
        });
        self.imports.insert(value.identity(), id);
        let mut stats = self.runtime.progress.borrow_mut();
        stats.values += 1;
        stats.imports += 1;
        Ok(ScopeValue {
            scope: self.identity,
            id,
        })
    }
    pub fn submit(
        &mut self,
        operation: impl Into<String>,
        inputs: Vec<ScopeValue>,
        attributes: Data,
        output_count: usize,
    ) -> Result<Vec<ScopeValue>> {
        self.recording()?;
        let operation = operation.into();
        if operation.is_empty() {
            return Err(Error::invalid("operation is empty"));
        }
        for &s in &inputs {
            self.validate(s)?;
        }
        growth(
            self.commands.len(),
            1,
            self.options.max_commands,
            "scope command cap exceeded",
        )?;
        growth(
            self.slots.len(),
            output_count,
            self.options.max_values,
            "scope value cap exceeded",
        )?;
        growth(
            self.edges,
            inputs.len(),
            self.options.max_input_edges,
            "scope edge cap exceeded",
        )?;
        let descriptors = if let Some(contract) = &self.options.contract {
            let arguments = inputs
                .iter()
                .map(|s| self.descriptor(*s).cloned())
                .collect::<Result<Vec<_>>>()?;
            let _metadata = guard(&self.runtime.metadata);
            let descriptors = callback(|| contract.infer(&operation, &arguments, &attributes))?;
            if descriptors.len() != output_count {
                return Err(Error::invalid(
                    "inferred output count differs from declaration",
                ));
            }
            for d in &descriptors {
                d.valid()?;
            }
            descriptors.into_iter().map(Some).collect::<Vec<_>>()
        } else {
            vec![None; output_count]
        };
        let outputs = (0..output_count)
            .map(|i| ScopeValue {
                scope: self.identity,
                id: self.slots.len() + i,
            })
            .collect::<Vec<_>>();
        let cursor = self.commands.len();
        for d in descriptors {
            self.slots.push(Slot {
                concrete: None,
                definition: Some(cursor),
                last_use: None,
                exported: false,
                descriptor: d,
            });
        }
        for s in &inputs {
            self.slots[s.id].last_use = Some(cursor);
        }
        self.edges += inputs.len();
        self.commands.push(ScopeCommand {
            operation,
            inputs,
            attributes,
            outputs: outputs.clone(),
        });
        let mut stats = self.runtime.progress.borrow_mut();
        stats.commands += 1;
        stats.values += output_count;
        stats.input_edges = self.edges;
        Ok(outputs)
    }
    pub fn descriptor(&self, symbol: ScopeValue) -> Result<&ScopeDescriptor> {
        self.validate(symbol)?;
        self.slots[symbol.id]
            .descriptor
            .as_ref()
            .ok_or_else(|| Error::invalid("descriptors require a scope contract"))
    }
    pub(crate) fn view(&self) -> ScopeView<'_, 'r> {
        ScopeView {
            scope: self,
            start: 0,
            length: self.commands.len(),
        }
    }
    pub(crate) fn check_deadline(&self) -> Result<()> {
        if self.options.max_duration_ms != 0
            && self.started.elapsed().as_millis() >= self.options.max_duration_ms as u128
        {
            Err(Error::budget("scope duration budget exceeded"))
        } else {
            Ok(())
        }
    }
    pub(crate) fn finish_capture(&mut self, exports: &[ScopeValue]) -> Result<()> {
        self.capturing = false;
        growth(
            0,
            exports.len(),
            self.options.max_exports,
            "scope export cap exceeded",
        )?;
        for &s in exports {
            self.validate(s)?;
        }
        for s in exports {
            self.slots[s.id].exported = true;
        }
        self.runtime.progress.borrow_mut().exports = exports.len();
        self.check_deadline()
    }
    pub(crate) fn execute(
        &mut self,
        exports: &[ScopeValue],
        kernels: &[Rc<ScopeKernel>],
    ) -> Result<Values> {
        self.finish_capture(exports)?;
        let mut releases = vec![vec![]; self.commands.len()];
        for (i, slot) in self.slots.iter().enumerate() {
            if !slot.exported && !releases.is_empty() {
                let last = slot.last_use.or(slot.definition).unwrap_or(0);
                releases[last].push(i);
            }
        }
        let mut attempts = 0;
        let mut cursor = 0;
        while cursor < self.commands.len() {
            let mut name = "runtime.submit".to_string();
            let mut detail = format!("start={cursor} count=0 liveins=0 liveouts=0");
            let result = callback(|| {
                self.check_deadline()?;
                let mut selected = None;
                let mut count = 1;
                if self.options.use_kernels {
                    for kernel in kernels {
                        name = kernel.name.clone();
                        if attempts == self.options.max_match_attempts {
                            return Err(Error::budget("scope match attempt cap exceeded"));
                        }
                        attempts += 1;
                        let length = kernel
                            .max_commands
                            .min(self.options.max_lookahead)
                            .min(self.commands.len() - cursor);
                        let candidate = ScopeView {
                            scope: self,
                            start: cursor,
                            length,
                        };
                        detail = candidate.detail(length)?;
                        let found = {
                            let _matching = guard(&self.runtime.matching);
                            (kernel.matches)(&candidate)?
                        };
                        if found.commands > length {
                            return Err(Error::invalid("matcher returned a span outside its view"));
                        }
                        if found.commands == 0 {
                            self.runtime.record(
                                TraceKind::Declined,
                                "scope.span",
                                &name,
                                format!("{detail} {}", found.reason),
                            );
                            continue;
                        }
                        count = found.commands;
                        selected = Some(kernel);
                        break;
                    }
                }
                if selected.is_none() {
                    name = "runtime.submit".into();
                }
                let span = ScopeView {
                    scope: self,
                    start: cursor,
                    length: count,
                };
                detail = span.detail(count)?;
                self.check_deadline()?;
                self.runtime
                    .record(TraceKind::Selected, "scope.span", &name, &detail);
                let (symbols, outputs) = if let Some(kernel) = selected {
                    (
                        span.live_outs(count)?,
                        (kernel.execute)(self.runtime, &span)?,
                    )
                } else {
                    let command = span.command(0)?;
                    let inputs = command
                        .inputs
                        .iter()
                        .map(|s| span.value(*s).cloned())
                        .collect::<Result<Vec<_>>>()?;
                    (
                        command.outputs.clone(),
                        self.runtime.submit(Request::new(
                            &command.operation,
                            inputs,
                            command.attributes.clone(),
                        ))?,
                    )
                };
                if symbols.len() != outputs.len() {
                    return Err(Error::invalid(
                        "executor returned an incorrect output count",
                    ));
                }
                if outputs.iter().any(|v| !self.runtime.owns(v)) {
                    return Err(Error::invalid("scope output belongs to another runtime"));
                }
                if let Some(contract) = &self.options.contract {
                    let _metadata = guard(&self.runtime.metadata);
                    for (symbol, output) in symbols.iter().zip(&outputs) {
                        let descriptor = self.descriptor(*symbol)?;
                        descriptor.validate_base(output)?;
                        contract.validate(output, descriptor)?;
                    }
                }
                for (symbol, output) in symbols.iter().zip(outputs) {
                    self.slots[symbol.id].concrete = Some(output);
                }
                for bucket in &releases[cursor..cursor + count] {
                    for &id in bucket {
                        self.slots[id].concrete = None;
                    }
                }
                cursor += count;
                {
                    let mut stats = self.runtime.progress.borrow_mut();
                    stats.executed_commands += count;
                    if selected.is_some() {
                        stats.kernel_spans += 1;
                    } else {
                        stats.recipe_spans += 1;
                    }
                }
                self.runtime
                    .record(TraceKind::Completed, "scope.span", &name, &detail);
                self.check_deadline()
            });
            if let Err(error) = result {
                self.runtime.record(
                    TraceKind::Failed,
                    "scope.span",
                    &name,
                    format!("{detail} {}", error.message),
                );
                return Err(error);
            }
        }
        exports
            .iter()
            .map(|s| {
                self.slots[s.id]
                    .concrete
                    .clone()
                    .ok_or_else(|| Error::execution("scope export was not produced"))
            })
            .collect()
    }
}
/// Borrowed only for the matcher/executor/inspection callback.
pub struct ScopeView<'a, 'r> {
    scope: &'a Scope<'r>,
    start: usize,
    length: usize,
}
impl ScopeView<'_, '_> {
    pub fn runtime(&self) -> &Runtime {
        self.scope.runtime
    }
    pub fn len(&self) -> usize {
        self.length
    }
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    pub fn start(&self) -> usize {
        self.start
    }
    pub fn subview(&self, offset: usize, length: usize) -> Result<Self> {
        if offset > self.length || length > self.length - offset {
            return Err(Error::invalid("subview exceeds view"));
        }
        Ok(Self {
            scope: self.scope,
            start: self.start + offset,
            length,
        })
    }
    pub fn command(&self, offset: usize) -> Result<&ScopeCommand> {
        if offset >= self.length {
            return Err(Error::invalid("command offset exceeds view"));
        }
        Ok(&self.scope.commands[self.start + offset])
    }
    pub fn find(&self, symbol: ScopeValue) -> Result<Option<&Value>> {
        self.scope.validate(symbol)?;
        Ok(self.scope.slots[symbol.id].concrete.as_ref())
    }
    pub fn value(&self, symbol: ScopeValue) -> Result<&Value> {
        self.find(symbol)?
            .ok_or_else(|| Error::invalid("symbol is not concrete at this cursor"))
    }
    pub fn descriptor(&self, symbol: ScopeValue) -> Result<&ScopeDescriptor> {
        self.scope.descriptor(symbol)
    }
    pub fn live_ins(&self, length: usize) -> Result<Vec<ScopeValue>> {
        if length > self.length {
            return Err(Error::invalid("span exceeds view"));
        }
        let mut result = vec![];
        let mut seen = HashSet::new();
        for command in &self.scope.commands[self.start..self.start + length] {
            for &symbol in &command.inputs {
                let definition = self.scope.slots[symbol.id].definition;
                if definition.is_none_or(|d| d < self.start) && seen.insert(symbol.id) {
                    result.push(symbol);
                }
            }
        }
        Ok(result)
    }
    pub fn live_outs(&self, length: usize) -> Result<Vec<ScopeValue>> {
        if length > self.length {
            return Err(Error::invalid("span exceeds view"));
        }
        let end = self.start + length;
        let mut result = vec![];
        for command in &self.scope.commands[self.start..end] {
            for &symbol in &command.outputs {
                let slot = &self.scope.slots[symbol.id];
                if slot.exported || slot.last_use.is_some_and(|i| i >= end) {
                    result.push(symbol);
                }
            }
        }
        Ok(result)
    }
    fn detail(&self, length: usize) -> Result<String> {
        Ok(format!(
            "start={} count={length} liveins={} liveouts={}",
            self.start,
            self.live_ins(length)?.len(),
            self.live_outs(length)?.len()
        ))
    }
}
