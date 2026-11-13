use std::{
    any::Any,
    fmt,
    panic::{catch_unwind, AssertUnwindSafe},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorCode {
    Unsupported,
    InvalidRequest,
    BudgetExceeded,
    ExecutionFailed,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub code: ErrorCode,
    pub message: String,
}
pub type Result<T> = std::result::Result<T, Error>;
impl Error {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidRequest, message)
    }
    pub fn budget(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::BudgetExceeded, message)
    }
    pub fn execution(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::ExecutionFailed, message)
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.code, self.message)
    }
}
impl std::error::Error for Error {}
pub(crate) fn callback<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|panic: Box<dyn Any + Send>| {
        let message = panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()));
        Err(Error::execution(
            message.unwrap_or_else(|| "callback panicked".into()),
        ))
    })
}
