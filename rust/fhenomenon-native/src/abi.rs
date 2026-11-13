//! ABI-1 layouts from cpp/include/FHN/{fhn_backend_api,fhn_native_api}.h.
//! Numeric opcodes use integers so unrecognized C values never create invalid
//! Rust enum discriminants. Only opaque handles cross the boundary.
use std::ffi::{c_char, c_int, c_void};

pub const VERSION: u32 = 1;
pub type Context = c_void;
pub type Buffer = c_void;
pub type Client = c_void;
pub type Kernel = unsafe extern "C" fn(
    *mut Context,
    *mut Buffer,
    *const *const Buffer,
    *const i64,
    *const f64,
) -> c_int;
#[repr(C)]
pub struct KernelEntry {
    pub opcode: u32,
    pub function: Option<Kernel>,
    pub name: *const c_char,
}
#[repr(C)]
pub struct KernelTable {
    pub count: u32,
    pub entries: *const KernelEntry,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BufferInfo {
    pub reserved_bytes: u64,
    pub live_payload_bytes: u64,
    pub scratch_capacity_bytes: u64,
    pub scratch_high_water_bytes: u64,
    pub level: i64,
    pub scale: f64,
    pub components: u32,
    pub logical_slots: u32,
}
#[repr(C)]
pub struct Instruction {
    pub opcode: u32,
    pub result_id: u32,
    pub operands: [u32; 4],
    pub params: [i64; 4],
    pub fparams: [f64; 2],
}
#[repr(C)]
pub struct Program {
    pub version: u32,
    pub count: u32,
    pub instructions: *const Instruction,
    pub num_inputs: u32,
    pub input_ids: *const u32,
    pub num_outputs: u32,
    pub output_ids: *const u32,
}
pub type AbiVersion = unsafe extern "C" fn() -> u32;
pub type Create = unsafe extern "C" fn(*const c_char) -> *mut Context;
pub type Destroy = unsafe extern "C" fn(*mut Context);
pub type GetKernels = unsafe extern "C" fn(*mut Context) -> *const KernelTable;
pub type Allocate = unsafe extern "C" fn(*mut Context) -> *mut Buffer;
pub type Free = unsafe extern "C" fn(*mut Context, *mut Buffer);
pub type Metadata = unsafe extern "C" fn(*mut Context, *const Buffer, *mut BufferInfo) -> c_int;
pub type FreshLevel = unsafe extern "C" fn(*mut Context) -> i64;
pub type LevelBytes = unsafe extern "C" fn(*mut Context, i64) -> u64;
pub type ClientCreate = unsafe extern "C" fn(*mut Context) -> *mut Client;
pub type ClientDestroy = unsafe extern "C" fn(*mut Client);
pub type Prepare = unsafe extern "C" fn(*mut Client, *const Program) -> c_int;
pub type Encrypt = unsafe extern "C" fn(*mut Client, *mut Buffer, *const f64, u32) -> c_int;
pub type Decrypt = unsafe extern "C" fn(*mut Client, *const Buffer, *mut f64, u32) -> c_int;

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn abi_one_field_offsets_match_native_headers_on_64_bit_targets() {
        assert_eq!(
            size_of::<usize>(),
            8,
            "native provider currently targets 64-bit hosts"
        );
        assert_eq!(size_of::<BufferInfo>(), 56);
        assert_eq!(align_of::<BufferInfo>(), 8);
        assert_eq!(offset_of!(BufferInfo, level), 32);
        assert_eq!(offset_of!(BufferInfo, scale), 40);
        assert_eq!(offset_of!(BufferInfo, logical_slots), 52);
        assert_eq!(size_of::<KernelEntry>(), 24);
        assert_eq!(offset_of!(KernelEntry, function), 8);
        assert_eq!(size_of::<KernelTable>(), 16);
        assert_eq!(offset_of!(KernelTable, entries), 8);
        assert_eq!(size_of::<Instruction>(), 72);
        assert_eq!(offset_of!(Instruction, params), 24);
        assert_eq!(offset_of!(Instruction, fparams), 56);
        assert_eq!(size_of::<Program>(), 48);
        assert_eq!(offset_of!(Program, instructions), 8);
        assert_eq!(offset_of!(Program, input_ids), 24);
        assert_eq!(offset_of!(Program, output_ids), 40);
    }
}
