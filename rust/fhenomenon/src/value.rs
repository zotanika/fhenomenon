use crate::{Error, Result, RuntimeOptions, RuntimeStats};
use std::{
    any::Any,
    cell::RefCell,
    fmt,
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
};

/// An owned public attribute/metadata snapshot or provider-owned payload.
/// Interior-mutability types are possible; providers must preserve immutability.
pub type Data = Rc<dyn Any>;
pub fn data<T: Any>(value: T) -> Data {
    Rc::new(value)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceKind {
    Value,
    Artifact,
}
pub type Values = Vec<Value>;
pub(crate) struct Accounting {
    pub options: RuntimeOptions,
    pub stats: RefCell<RuntimeStats>,
}
pub(crate) struct Charge {
    pub owner: Rc<Accounting>,
    pub bytes: usize,
}
impl Charge {
    pub fn new(owner: Rc<Accounting>, bytes: usize) -> Result<Self> {
        {
            let mut stats = owner.stats.borrow_mut();
            let total = stats
                .live_bytes
                .checked_add(bytes)
                .ok_or_else(|| Error::budget("retained byte count overflow"))?;
            if owner.options.max_live_bytes != 0 && total > owner.options.max_live_bytes {
                return Err(Error::budget("retained resource budget exceeded"));
            }
            let count = stats
                .live_resources
                .checked_add(1)
                .ok_or_else(|| Error::budget("resource count overflow"))?;
            stats.live_bytes = total;
            stats.live_resources = count;
            stats.peak_live_bytes = stats.peak_live_bytes.max(total);
        }
        Ok(Self { owner, bytes })
    }
}
impl Drop for Charge {
    fn drop(&mut self) {
        let mut stats = self.owner.stats.borrow_mut();
        stats.live_bytes -= self.bytes;
        stats.live_resources -= 1;
    }
}
// Keep the atomic API available at the workspace MSRV. Newer Rust renames it
// to try_update, which is not available on all supported toolchains.
#[allow(deprecated)]
pub(crate) fn identity() -> Result<u64> {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| Error::execution("resource identities exhausted"))
}
// Rust drops fields in declaration order. The payload may need dependencies
// during destruction, and the charge must remain until both are destroyed.
pub(crate) struct Resource {
    pub payload: Data,
    pub dependencies: Values,
    pub charge: Charge,
    pub id: u64,
    pub format: String,
    pub kind: ResourceKind,
}
#[derive(Clone)]
pub struct Value(pub(crate) Rc<Resource>);
impl Value {
    pub fn identity(&self) -> u64 {
        self.0.id
    }
    pub fn format(&self) -> &str {
        &self.0.format
    }
    pub fn kind(&self) -> ResourceKind {
        self.0.kind
    }
    pub fn bytes(&self) -> usize {
        self.0.charge.bytes
    }
    pub fn payload<T: Any>(&self) -> Result<&T> {
        self.0
            .payload
            .downcast_ref()
            .ok_or_else(|| Error::invalid("provider payload type mismatch"))
    }
    pub fn dependencies(&self) -> &[Value] {
        &self.0.dependencies
    }
}
impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Value")
            .field("identity", &self.identity())
            .field("format", &self.format())
            .field("bytes", &self.bytes())
            .finish_non_exhaustive()
    }
}
