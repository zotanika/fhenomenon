use std::ffi::{c_void, c_int};
use tfhe::{ConfigBuilder, ServerKey, ClientKey, FheInt32};
use tfhe::prelude::*;
use tfhe::set_server_key;

// Opaque handles for C++
pub struct TfheContext {
    client_key: ClientKey,
    server_key: ServerKey,
}

pub struct CiphertextHandle {
    ct: FheInt32,
}

#[no_mangle]
pub extern "C" fn tfhe_context_create() -> *mut TfheContext {
    // Generate keys (insecure/short-term for now to speed up init, 
    // or standard default params if user requested "full complexity")
    // The user asked for "x64 size inflation", so we should use standard params 
    // or at least parameters that produce LWE ciphertexts.
    // Let's use default "block parameters" for integers.
    let config = ConfigBuilder::default().build();
    let (client_key, server_key) = tfhe::generate_keys(config);

    let context = Box::new(TfheContext {
        client_key,
        server_key,
    });
    Box::into_raw(context)
}

#[no_mangle]
pub extern "C" fn tfhe_context_destroy(ctx: *mut TfheContext) {
    if !ctx.is_null() {
        unsafe { drop(Box::from_raw(ctx)) };
    }
}

#[no_mangle]
pub extern "C" fn tfhe_ciphertext_destroy(handle: *mut CiphertextHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

#[no_mangle]
pub extern "C" fn tfhe_encrypt_int32(ctx: *mut TfheContext, value: i32) -> *mut CiphertextHandle {
    let ctx = unsafe { &*ctx };
    let ct = FheInt32::encrypt(value, &ctx.client_key);
    Box::into_raw(Box::new(CiphertextHandle { ct }))
}

#[no_mangle]
pub extern "C" fn tfhe_decrypt_int32(ctx: *mut TfheContext, handle: *mut CiphertextHandle) -> i32 {
    let ctx = unsafe { &*ctx };
    let handle = unsafe { &*handle };
    handle.ct.decrypt(&ctx.client_key)
}

#[no_mangle]
pub extern "C" fn tfhe_add_int32(ctx: *mut TfheContext, a: *mut CiphertextHandle, b: *mut CiphertextHandle) -> *mut CiphertextHandle {
    let ctx = unsafe { &*ctx };
    let a = unsafe { &*a };
    let b = unsafe { &*b };

    // Set server key for homomorphic operations
    set_server_key(ctx.server_key.clone());

    let result = &a.ct + &b.ct;
    Box::into_raw(Box::new(CiphertextHandle { ct: result }))
}

#[no_mangle]
pub extern "C" fn tfhe_mul_int32(ctx: *mut TfheContext, a: *mut CiphertextHandle, b: *mut CiphertextHandle) -> *mut CiphertextHandle {
    let ctx = unsafe { &*ctx };
    let a = unsafe { &*a };
    let b = unsafe { &*b };

    set_server_key(ctx.server_key.clone());

    let result = &a.ct * &b.ct;
    Box::into_raw(Box::new(CiphertextHandle { ct: result }))
}

#[no_mangle]
pub extern "C" fn tfhe_bitand_int32(ctx: *mut TfheContext, a: *mut CiphertextHandle, b: *mut CiphertextHandle) -> *mut CiphertextHandle {
    let ctx = unsafe { &*ctx };
    let a = unsafe { &*a };
    let b = unsafe { &*b };

    set_server_key(ctx.server_key.clone());

    let result = &a.ct & &b.ct;
    Box::into_raw(Box::new(CiphertextHandle { ct: result }))
}

#[no_mangle]
pub extern "C" fn tfhe_bitor_int32(ctx: *mut TfheContext, a: *mut CiphertextHandle, b: *mut CiphertextHandle) -> *mut CiphertextHandle {
    let ctx = unsafe { &*ctx };
    let a = unsafe { &*a };
    let b = unsafe { &*b };

    set_server_key(ctx.server_key.clone());

    let result = &a.ct | &b.ct;
    Box::into_raw(Box::new(CiphertextHandle { ct: result }))
}

#[no_mangle]
pub extern "C" fn tfhe_bitxor_int32(ctx: *mut TfheContext, a: *mut CiphertextHandle, b: *mut CiphertextHandle) -> *mut CiphertextHandle {
    let ctx = unsafe { &*ctx };
    let a = unsafe { &*a };
    let b = unsafe { &*b };

    set_server_key(ctx.server_key.clone());

    let result = &a.ct ^ &b.ct;
    Box::into_raw(Box::new(CiphertextHandle { ct: result }))
}

#[no_mangle]
pub extern "C" fn tfhe_eq_int32(ctx: *mut TfheContext, a: *mut CiphertextHandle, b: *mut CiphertextHandle) -> *mut CiphertextHandle {
    let ctx = unsafe { &*ctx };
    let a = unsafe { &*a };
    let b = unsafe { &*b };

    set_server_key(ctx.server_key.clone());

    let result_bool = (&a.ct).eq(&b.ct);
    // Cast FheBool to FheInt32 (1 if true, 0 if false)
    let result: FheInt32 = result_bool.cast_into();
    Box::into_raw(Box::new(CiphertextHandle { ct: result }))
}

#[no_mangle]
pub extern "C" fn tfhe_lt_int32(ctx: *mut TfheContext, a: *mut CiphertextHandle, b: *mut CiphertextHandle) -> *mut CiphertextHandle {
    let ctx = unsafe { &*ctx };
    let a = unsafe { &*a };
    let b = unsafe { &*b };

    set_server_key(ctx.server_key.clone());

    let result_bool = (&a.ct).lt(&b.ct);
    let result: FheInt32 = result_bool.cast_into();
    Box::into_raw(Box::new(CiphertextHandle { ct: result }))
}

#[no_mangle]
pub extern "C" fn tfhe_le_int32(ctx: *mut TfheContext, a: *mut CiphertextHandle, b: *mut CiphertextHandle) -> *mut CiphertextHandle {
    let ctx = unsafe { &*ctx };
    let a = unsafe { &*a };
    let b = unsafe { &*b };

    set_server_key(ctx.server_key.clone());

    let result_bool = (&a.ct).le(&b.ct);
    let result: FheInt32 = result_bool.cast_into();
    Box::into_raw(Box::new(CiphertextHandle { ct: result }))
}
