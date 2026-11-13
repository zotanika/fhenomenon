//! Shared native CKKS provider, independent of the C++ runtime.
//!
//! The compute owner retains only a context, kernel pointers and allocation
//! functions. [`NativeClient`] separately owns secret keys; no encryption or
//! decryption operation enters the recipe registry. Resources retain the context
//! and dynamic library until their native buffers have been destroyed.
//!
//! This adapter covers physical operations whose metadata can be inferred from
//! ABI 1 without private modulus tables. Rescale, normalized multiplication,
//! compiled regions, prepared artifacts and CUDA-specific APIs are not exposed.
//! The generic runtime can register other providers and coarse scope kernels.
mod abi;

pub use abi::BufferInfo;
use fhenomenon::*;
use libloading::Library;
use std::{
    collections::BTreeMap,
    ffi::{CString, OsStr},
    ptr::NonNull,
    rc::Rc,
};

const FORMAT: &str = "native.ckks.buffer";
const OPERATION: &str = "native.ckks.physical";

/// Physical operations with fully specified level, scale and component effects.
/// `TensorProduct` produces three components and does not rescale. It is not a
/// normalized CKKS multiplication. Rotation keys must be prepared by the client.
#[derive(Clone, Copy, Debug)]
pub enum Operation {
    Copy,
    Add,
    Subtract,
    Negate,
    AddScalar(f64),
    ScalarSubtract(f64),
    TensorProduct,
    Relinearize,
    Rotate(i64),
    Conjugate,
    LevelDown(i64),
    RotateAdd(i64),
    ConjugateAdd,
}
struct Spec {
    opcode: u32,
    arity: usize,
    params: [i64; 4],
    fparams: [f64; 2],
}
impl Operation {
    fn spec(self) -> Result<Spec> {
        let (opcode, arity, integer, scalar) = match self {
            Self::Copy => (0, 1, 0, 0.0),
            Self::Add => (1, 2, 0, 0.0),
            Self::Subtract => (4, 2, 0, 0.0),
            Self::Negate => (7, 1, 0, 0.0),
            Self::AddScalar(n) => (3, 1, 0, n),
            Self::ScalarSubtract(n) => (6, 1, 0, n),
            Self::TensorProduct => (8, 2, 0, 0.0),
            Self::Relinearize => (11, 1, 0, 0.0),
            Self::Rotate(n) => (13, 1, n, 0.0),
            Self::Conjugate => (14, 1, 0, 0.0),
            Self::LevelDown(n) => (17, 1, n, 0.0),
            Self::RotateAdd(n) => (20, 2, n, 0.0),
            Self::ConjugateAdd => (21, 2, 0, 0.0),
        };
        if !scalar.is_finite() {
            return Err(Error::invalid("native scalar must be finite"));
        }
        if matches!(self,Self::LevelDown(n) if n<0) {
            return Err(Error::invalid("target level must be nonnegative"));
        }
        Ok(Spec {
            opcode,
            arity,
            params: [integer, 0, 0, 0],
            fparams: [scalar, 0.0],
        })
    }
}

struct ComputeFunctions {
    destroy: abi::Destroy,
    allocate: abi::Allocate,
    free: abi::Free,
    metadata: abi::Metadata,
}
struct Context {
    ptr: NonNull<abi::Context>,
    functions: ComputeFunctions,
    kernels: BTreeMap<u32, abi::Kernel>,
    fresh_level: i64,
    reserved_bytes: usize,
    // Declared last: library unload follows all context-owned metadata.
    library: Library,
}
impl Drop for Context {
    fn drop(&mut self) {
        // The context is uniquely owned here; every buffer and client retains it.
        unsafe { (self.functions.destroy)(self.ptr.as_ptr()) };
    }
}
struct BufferLease {
    ptr: NonNull<abi::Buffer>,
    context: Rc<Context>,
}
impl Drop for BufferLease {
    fn drop(&mut self) {
        // The retained context/library is alive until after this destructor.
        unsafe { (self.context.functions.free)(self.context.ptr.as_ptr(), self.ptr.as_ptr()) };
    }
}
impl Context {
    fn buffer<'a>(&self, value: &'a Value) -> Result<&'a BufferLease> {
        if value.format() != FORMAT || value.kind() != ResourceKind::Value {
            return Err(Error::invalid("native ciphertext format mismatch"));
        }
        let buffer = value.payload::<BufferLease>()?;
        if !std::ptr::eq(self, buffer.context.as_ref()) {
            return Err(Error::invalid(
                "native ciphertext belongs to another context",
            ));
        }
        Ok(buffer)
    }
    fn info(&self, buffer: &BufferLease) -> Result<BufferInfo> {
        let mut out = BufferInfo::default();
        // buffer owns this context and therefore remains allocated for the call.
        let status =
            unsafe { (self.functions.metadata)(self.ptr.as_ptr(), buffer.ptr.as_ptr(), &mut out) };
        if status != 0
            || out.logical_slots == 0
            || out.level < 0
            || out.level > self.fresh_level
            || !out.scale.is_finite()
            || out.scale <= 0.0
            || !(2..=3).contains(&out.components)
            || usize::try_from(out.reserved_bytes).ok() != Some(self.reserved_bytes)
        {
            return Err(Error::invalid(
                "native provider returned invalid ciphertext metadata",
            ));
        }
        Ok(out)
    }
    fn allocate(
        self: &Rc<Self>,
        runtime: &Runtime,
        fill: impl FnOnce(&BufferLease) -> Result<()>,
    ) -> Result<Value> {
        runtime.make_value(
            FORMAT,
            ResourceKind::Value,
            self.reserved_bytes,
            || {
                // The runtime has reserved bytes before any native allocation.
                let ptr = NonNull::new(unsafe { (self.functions.allocate)(self.ptr.as_ptr()) })
                    .ok_or_else(|| Error::execution("native ciphertext allocation failed"))?;
                let buffer = BufferLease {
                    ptr,
                    context: self.clone(),
                };
                fill(&buffer)?;
                self.info(&buffer)?;
                Ok(data(buffer))
            },
            vec![],
        )
    }
}

// Symbol loading is private; the sole public unsafe boundary establishes that
// this library uses the declared ABI and honors its pointer/lifetime contracts.
unsafe fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
    unsafe { library.get::<T>(name) }
        .map(|s| *s)
        .map_err(|e| Error::execution(format!("native export unavailable: {e}")))
}

/// Single-threaded compute capability. Clone shares only the evaluation context.
#[derive(Clone)]
pub struct NativeProvider {
    context: Rc<Context>,
}
impl NativeProvider {
    /// Load the native CKKS ABI directly, without the C++ Runtime or Scope.
    ///
    /// # Safety
    /// The library must be trusted and implement the ABI-1 CKKS function
    /// signatures, buffer ownership and immutable-input contracts declared by
    /// this repository. A matching version alone cannot prove native code safe.
    /// All handles remain single-threaded through `Rc` ownership.
    pub unsafe fn load(path: impl AsRef<OsStr>, config_json: &str) -> Result<Self> {
        let config = CString::new(config_json)
            .map_err(|_| Error::invalid("configuration contains a NUL byte"))?;
        let library = unsafe { Library::new(path) }
            .map_err(|e| Error::execution(format!("cannot load native provider: {e}")))?;
        let version: abi::AbiVersion = unsafe { symbol(&library, b"ckks_fhn_get_abi_version\0") }?;
        if unsafe { version() } != abi::VERSION {
            return Err(Error::new(
                ErrorCode::Unsupported,
                "native ABI version mismatch",
            ));
        }
        let create: abi::Create = unsafe { symbol(&library, b"ckks_fhn_create\0") }?;
        let destroy = unsafe { symbol(&library, b"ckks_fhn_destroy\0") }?;
        let allocate = unsafe { symbol(&library, b"ckks_fhn_buffer_alloc\0") }?;
        let free = unsafe { symbol(&library, b"ckks_fhn_buffer_free\0") }?;
        let metadata = unsafe { symbol(&library, b"ckks_fhn_buffer_info\0") }?;
        let get_kernels: abi::GetKernels = unsafe { symbol(&library, b"ckks_fhn_get_kernels\0") }?;
        let fresh_level: abi::FreshLevel = unsafe { symbol(&library, b"ckks_fhn_fresh_level\0") }?;
        let level_bytes: abi::LevelBytes = unsafe { symbol(&library, b"ckks_fhn_level_bytes\0") }?;
        let ptr = NonNull::new(unsafe { create(config.as_ptr()) })
            .ok_or_else(|| Error::invalid("native context creation or configuration failed"))?;
        // Install ownership before any fallible metadata checks.
        let mut context = Context {
            ptr,
            functions: ComputeFunctions {
                destroy,
                allocate,
                free,
                metadata,
            },
            kernels: BTreeMap::new(),
            fresh_level: 0,
            reserved_bytes: 0,
            library,
        };
        context.fresh_level = unsafe { fresh_level(ptr.as_ptr()) };
        context.reserved_bytes =
            usize::try_from(unsafe { level_bytes(ptr.as_ptr(), context.fresh_level) })
                .map_err(|_| Error::invalid("native buffer reservation exceeds host size"))?;
        if context.fresh_level < 0 || context.reserved_bytes == 0 {
            return Err(Error::invalid("invalid native level or reservation"));
        }
        let table = unsafe { get_kernels(ptr.as_ptr()) };
        if table.is_null() {
            return Err(Error::new(
                ErrorCode::Unsupported,
                "native kernel table is unavailable",
            ));
        }
        let table = unsafe { &*table };
        if table.count > 4096 || (table.count != 0 && table.entries.is_null()) {
            return Err(Error::invalid("invalid native kernel table"));
        }
        for i in 0..table.count as usize {
            let entry = unsafe { &*table.entries.add(i) };
            if let Some(function) = entry.function {
                context.kernels.insert(entry.opcode, function);
            }
        }
        Ok(Self {
            context: Rc::new(context),
        })
    }
    pub fn fresh_level(&self) -> i64 {
        self.context.fresh_level
    }
    /// Charged ciphertext capacity, including unused allocated limbs/components.
    /// Key material, native scratch and immutable tables are not part of it.
    pub fn reserved_bytes(&self) -> usize {
        self.context.reserved_bytes
    }
    pub fn metadata(&self, value: &Value) -> Result<BufferInfo> {
        self.context.info(self.context.buffer(value)?)
    }
    pub fn supports(&self, operation: Operation) -> bool {
        operation
            .spec()
            .is_ok_and(|s| self.context.kernels.contains_key(&s.opcode))
    }
    pub fn scope_contract(&self) -> Rc<dyn ScopeContract> {
        Rc::new(NativeContract {
            context: self.context.clone(),
        })
    }
    pub fn request(&self, operation: Operation, inputs: Values) -> Result<Request> {
        let spec = operation.spec()?;
        if inputs.len() != spec.arity {
            return Err(Error::invalid("incorrect native operation arity"));
        }
        for value in &inputs {
            self.context.buffer(value)?;
        }
        Ok(Request::new(
            OPERATION,
            inputs,
            data(Attributes {
                operation,
                context: self.context.clone(),
            }),
        ))
    }
    pub fn execute(
        &self,
        runtime: &Runtime,
        operation: Operation,
        inputs: Values,
    ) -> Result<Value> {
        let mut output = runtime.submit(self.request(operation, inputs)?)?;
        if output.len() != 1 {
            return Err(Error::invalid("native recipe must return one ciphertext"));
        }
        Ok(output.remove(0))
    }
    pub fn capture(
        &self,
        scope: &mut Scope<'_>,
        operation: Operation,
        inputs: Vec<ScopeValue>,
    ) -> Result<ScopeValue> {
        let spec = operation.spec()?;
        if inputs.len() != spec.arity {
            return Err(Error::invalid("incorrect native operation arity"));
        }
        Ok(scope.submit(
            OPERATION,
            inputs,
            data(Attributes {
                operation,
                context: self.context.clone(),
            }),
            1,
        )?[0])
    }
    pub fn register(&self, runtime: &Runtime) -> Result<()> {
        let context = self.context.clone();
        let mut recipe = Recipe::new(OPERATION, "native.ckks.abi1", move |rt, request| {
            let attributes = attributes(&context, &request.attributes)?;
            let spec = attributes.operation.spec()?;
            let kernel = *context.kernels.get(&spec.opcode).ok_or_else(|| {
                Error::new(
                    ErrorCode::Unsupported,
                    "native physical operation unavailable",
                )
            })?;
            let buffers = request
                .inputs
                .iter()
                .map(|v| context.buffer(v))
                .collect::<Result<Vec<_>>>()?;
            let infos = buffers
                .iter()
                .map(|b| context.info(b))
                .collect::<Result<Vec<_>>>()?;
            let predicted = predict(attributes.operation, &infos)?;
            let operands = buffers
                .iter()
                .map(|b| b.ptr.as_ptr().cast_const())
                .collect::<Vec<_>>();
            let value = context.allocate(rt, |buffer| {
                // Operand slices and scalar arrays stay alive for this synchronous call.
                if unsafe {
                    kernel(
                        context.ptr.as_ptr(),
                        buffer.ptr.as_ptr(),
                        operands.as_ptr(),
                        spec.params.as_ptr(),
                        spec.fparams.as_ptr(),
                    )
                } != 0
                {
                    return Err(Error::execution(
                        "native physical operation failed (including unavailable evaluation keys)",
                    ));
                }
                validate_info(&context.info(buffer)?, &predicted)
            })?;
            Ok(vec![value])
        });
        let context = self.context.clone();
        recipe.accepts = Rc::new(move |request| {
            let attr = request
                .attributes
                .downcast_ref::<Attributes>()
                .ok_or_else(|| Error::invalid("native operation requires typed attributes"))?;
            if !Rc::ptr_eq(&attr.context, &context) {
                return Ok(Acceptance::Decline("different native context".into()));
            }
            let spec = attr.operation.spec()?;
            Ok(if context.kernels.contains_key(&spec.opcode) {
                Acceptance::Accept
            } else {
                Acceptance::Decline("native opcode unavailable".into())
            })
        });
        runtime.add_recipe(recipe)
    }
    /// Inspect a captured command without exposing provider internals or doing
    /// arithmetic. Commands for another provider/context return `None`.
    pub fn operation(&self, command: &ScopeCommand) -> Option<Operation> {
        if command.operation != OPERATION {
            return None;
        }
        let attr = command.attributes.downcast_ref::<Attributes>()?;
        Rc::ptr_eq(&attr.context, &self.context).then_some(attr.operation)
    }
    /// Add one optional coarse kernel for `rotate(x) + y`. Its native ABI call
    /// materializes only the final ciphertext. A separately exported or reused
    /// rotation result declines fusion and uses the ordinary recipes.
    pub fn register_scope_kernels(&self, runtime: &Runtime) -> Result<()> {
        let matcher = self.clone();
        let executor = self.clone();
        runtime.add_scope_kernel(ScopeKernel::new(
            "native.ckks.rotate-add",
            2,
            move |view| {
                Ok(if matcher.rotate_add_shape(view)?.is_some() {
                    ScopeMatch::accept(2)
                } else {
                    ScopeMatch::decline(
                        "requires a private rotate intermediate and one add live-out",
                    )
                })
            },
            move |rt, view| {
                let (rotation, source, addend) =
                    executor.rotate_add_shape(view)?.ok_or_else(|| {
                        Error::invalid("native rotate-add span no longer satisfies its contract")
                    })?;
                Ok(vec![executor.execute(
                    rt,
                    Operation::RotateAdd(rotation),
                    vec![view.value(source)?.clone(), view.value(addend)?.clone()],
                )?])
            },
        ))
    }
    fn rotate_add_shape(
        &self,
        view: &ScopeView<'_, '_>,
    ) -> Result<Option<(i64, ScopeValue, ScopeValue)>> {
        if view.len() != 2 {
            return Ok(None);
        }
        let rotate = view.command(0)?;
        let add = view.command(1)?;
        let Some(Operation::Rotate(rotation)) = self.operation(rotate) else {
            return Ok(None);
        };
        if !matches!(self.operation(add), Some(Operation::Add))
            || !self.supports(Operation::RotateAdd(rotation))
            || rotate.inputs.len() != 1
            || rotate.outputs.len() != 1
            || add.inputs.len() != 2
            || add.outputs.len() != 1
            || view.live_outs(2)? != add.outputs
        {
            return Ok(None);
        }
        let addend = if add.inputs[0] == rotate.outputs[0] {
            add.inputs[1]
        } else if add.inputs[1] == rotate.outputs[0] {
            add.inputs[0]
        } else {
            return Ok(None);
        };
        // Two uses of the private rotation output cannot use this ABI kernel.
        if view.find(rotate.inputs[0])?.is_none() || view.find(addend)?.is_none() {
            return Ok(None);
        }
        Ok(Some((rotation, rotate.inputs[0], addend)))
    }
    /// Construct a separate secret-owning client. The native ABI permits one
    /// client per context; compute owners never retain the resulting client.
    pub fn create_client(&self) -> Result<NativeClient> {
        let library = &self.context.library;
        let create: abi::ClientCreate = unsafe { symbol(library, b"ckks_fhn_client_create\0") }?;
        let destroy = unsafe { symbol(library, b"ckks_fhn_client_destroy\0") }?;
        let prepare = unsafe { symbol(library, b"ckks_fhn_client_prepare\0") }?;
        let encrypt = unsafe { symbol(library, b"ckks_fhn_client_encrypt_vec_f64\0") }?;
        let decrypt = unsafe { symbol(library, b"ckks_fhn_client_decrypt_vec_f64\0") }?;
        let ptr = NonNull::new(unsafe { create(self.context.ptr.as_ptr()) }).ok_or_else(|| {
            Error::execution("native client creation failed or client already exists")
        })?;
        Ok(NativeClient {
            ptr,
            destroy,
            prepare,
            encrypt,
            decrypt,
            context: self.context.clone(),
        })
    }
}

struct Attributes {
    operation: Operation,
    context: Rc<Context>,
}
fn attributes<'a>(context: &Rc<Context>, data: &'a Data) -> Result<&'a Attributes> {
    let attr = data
        .downcast_ref::<Attributes>()
        .ok_or_else(|| Error::invalid("native operation requires typed attributes"))?;
    if !Rc::ptr_eq(context, &attr.context) {
        return Err(Error::invalid(
            "native attributes belong to another context",
        ));
    }
    Ok(attr)
}
fn same_scale(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * a.max(b)
}
fn predict(operation: Operation, inputs: &[BufferInfo]) -> Result<BufferInfo> {
    let spec = operation.spec()?;
    if inputs.len() != spec.arity {
        return Err(Error::invalid("incorrect native operation arity"));
    }
    let mut out = inputs[0];
    if inputs.len() == 2 {
        let b = inputs[1];
        if out.level != b.level || out.logical_slots != b.logical_slots {
            return Err(Error::invalid(
                "native operands must have equal levels and logical slots",
            ));
        }
        if !matches!(operation, Operation::TensorProduct)
            && (out.components != b.components || !same_scale(out.scale, b.scale))
        {
            return Err(Error::invalid(
                "native linear operands must have equal components and scales",
            ));
        }
    }
    match operation {
        Operation::LevelDown(level) => {
            if level > out.level {
                return Err(Error::invalid("level-down cannot raise a ciphertext"));
            }
            out.level = level;
        }
        Operation::TensorProduct => {
            if inputs.iter().any(|i| i.components != 2) {
                return Err(Error::invalid(
                    "tensor product requires two-component inputs",
                ));
            }
            out.components = 3;
            out.scale *= inputs[1].scale;
            if !out.scale.is_finite() {
                return Err(Error::invalid("tensor scale overflow"));
            }
        }
        Operation::Relinearize => {
            if out.components != 3 {
                return Err(Error::invalid("relinearization requires three components"));
            }
            out.components = 2;
        }
        Operation::Rotate(_)
        | Operation::Conjugate
        | Operation::RotateAdd(_)
        | Operation::ConjugateAdd => {
            if out.components != 2 {
                return Err(Error::invalid("rotation requires two components"));
            }
        }
        Operation::AddScalar(scalar) | Operation::ScalarSubtract(scalar) => {
            let encoded = scalar * out.scale;
            if !encoded.is_finite() || encoded.abs() >= i64::MAX as f64 {
                return Err(Error::invalid(
                    "scalar exceeds native signed encoding range",
                ));
            }
        }
        _ => {}
    }
    Ok(out)
}
fn validate_info(actual: &BufferInfo, expected: &BufferInfo) -> Result<()> {
    if actual.reserved_bytes != expected.reserved_bytes
        || actual.level != expected.level
        || actual.components != expected.components
        || actual.logical_slots != expected.logical_slots
        || !same_scale(actual.scale, expected.scale)
    {
        return Err(Error::invalid(
            "native output differs from declared level, scale, components or slots",
        ));
    }
    Ok(())
}
struct NativeDescriptor {
    context: Rc<Context>,
    info: BufferInfo,
}
struct NativeContract {
    context: Rc<Context>,
}
impl NativeContract {
    fn descriptor(&self, info: BufferInfo) -> ScopeDescriptor {
        ScopeDescriptor::new(
            FORMAT,
            ResourceKind::Value,
            self.context.reserved_bytes,
            data(NativeDescriptor {
                context: self.context.clone(),
                info,
            }),
        )
    }
    fn info(&self, descriptor: &ScopeDescriptor) -> Result<BufferInfo> {
        let native = descriptor
            .metadata
            .downcast_ref::<NativeDescriptor>()
            .ok_or_else(|| Error::invalid("native descriptor metadata missing"))?;
        if descriptor.format != FORMAT
            || descriptor.kind != ResourceKind::Value
            || descriptor.bytes != self.context.reserved_bytes
            || !Rc::ptr_eq(&native.context, &self.context)
        {
            return Err(Error::invalid(
                "native descriptor belongs to another provider or context",
            ));
        }
        Ok(native.info)
    }
}
impl ScopeContract for NativeContract {
    fn describe(&self, value: &Value) -> Result<ScopeDescriptor> {
        Ok(self.descriptor(self.context.info(self.context.buffer(value)?)?))
    }
    fn infer(
        &self,
        operation: &str,
        inputs: &[ScopeDescriptor],
        attrs: &Data,
    ) -> Result<Vec<ScopeDescriptor>> {
        if operation != OPERATION {
            return Err(Error::new(
                ErrorCode::Unsupported,
                "operation is not in the native contract",
            ));
        }
        let attrs = attributes(&self.context, attrs)?;
        let spec = attrs.operation.spec()?;
        if !self.context.kernels.contains_key(&spec.opcode) {
            return Err(Error::new(
                ErrorCode::Unsupported,
                "native opcode unavailable",
            ));
        }
        let infos = inputs
            .iter()
            .map(|d| self.info(d))
            .collect::<Result<Vec<_>>>()?;
        Ok(vec![self.descriptor(predict(attrs.operation, &infos)?)])
    }
    fn validate(&self, value: &Value, descriptor: &ScopeDescriptor) -> Result<()> {
        validate_info(
            &self.context.info(self.context.buffer(value)?)?,
            &self.info(descriptor)?,
        )
    }
}

/// Trusted client capability. Secret material stays in its native client object.
/// Dropping it expires native evaluation keys without invalidating buffers.
pub struct NativeClient {
    ptr: NonNull<abi::Client>,
    destroy: abi::ClientDestroy,
    prepare: abi::Prepare,
    encrypt: abi::Encrypt,
    decrypt: abi::Decrypt,
    context: Rc<Context>,
}
impl Drop for NativeClient {
    fn drop(&mut self) {
        // The client is destroyed before its retained evaluation context/library.
        unsafe { (self.destroy)(self.ptr.as_ptr()) };
    }
}
impl NativeClient {
    pub fn prepare_keys(&mut self, runtime: &Runtime, operations: &[Operation]) -> Result<()> {
        runtime.check_evaluation_allowed()?;
        let count = u32::try_from(operations.len())
            .map_err(|_| Error::invalid("too many key preparation operations"))?;
        let instructions = operations
            .iter()
            .map(|op| {
                let spec = op.spec()?;
                Ok(abi::Instruction {
                    opcode: spec.opcode,
                    result_id: 0,
                    operands: [0; 4],
                    params: spec.params,
                    fparams: spec.fparams,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let program = abi::Program {
            version: abi::VERSION,
            count,
            instructions: instructions.as_ptr(),
            num_inputs: 0,
            input_ids: std::ptr::null(),
            num_outputs: 0,
            output_ids: std::ptr::null(),
        };
        if unsafe { (self.prepare)(self.ptr.as_ptr(), &program) } != 0 {
            return Err(Error::execution("native evaluation key preparation failed"));
        }
        Ok(())
    }
    pub fn encrypt(&mut self, runtime: &Runtime, values: &[f64]) -> Result<Value> {
        runtime.check_evaluation_allowed()?;
        if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
            return Err(Error::invalid("encryption requires nonempty finite input"));
        }
        let count = u32::try_from(values.len())
            .map_err(|_| Error::invalid("plaintext has too many logical slots"))?;
        self.context.allocate(runtime, |buffer| {
            // Both slices and client/context owners remain alive through the call.
            if unsafe {
                (self.encrypt)(
                    self.ptr.as_ptr(),
                    buffer.ptr.as_ptr(),
                    values.as_ptr(),
                    count,
                )
            } != 0
            {
                return Err(Error::invalid(
                    "native encryption rejected input count or range",
                ));
            }
            Ok(())
        })
    }
    pub fn decrypt(&self, runtime: &Runtime, value: &Value) -> Result<Vec<f64>> {
        runtime.check_evaluation_allowed()?;
        if !runtime.owns(value) {
            return Err(Error::invalid("ciphertext belongs to another runtime"));
        }
        let buffer = self.context.buffer(value)?;
        let info = self.context.info(buffer)?;
        let mut plaintext = vec![0.0; info.logical_slots as usize];
        if unsafe {
            (self.decrypt)(
                self.ptr.as_ptr(),
                buffer.ptr.as_ptr(),
                plaintext.as_mut_ptr(),
                info.logical_slots,
            )
        } != 0
        {
            return Err(Error::execution("native decryption failed"));
        }
        Ok(plaintext)
    }
}
