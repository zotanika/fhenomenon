//! Independent, synchronous Fhenomenon runtime with complete scoped execution.
//!
//! Values and runtimes are single-threaded. Providers own immutable payloads;
//! kernels may specialize only on public metadata. Native arithmetic is supplied
//! separately, so this crate neither links C++ nor implements cryptography.
mod error;
mod runtime;
mod scope;
mod value;

pub use error::{Error, ErrorCode, Result};
pub use runtime::*;
pub use scope::*;
pub use value::{data, Data, ResourceKind, Value, Values};
