//! Apple Neural Engine programs for IronMLX's long-prompt prefill offload.
//!
//! The ANE is reached through the private `AppleNeuralEngine.framework`
//! (there is no public API for it): a program is a MIL text plus a weights
//! blob, compiled and loaded by the system service, evaluated with
//! IOSurface-backed tensors. Every framework method is checked against its
//! expected Objective-C type encoding when the runtime opens; any mismatch or
//! missing class makes [`Runtime::open`] report the ANE unavailable, so callers
//! keep their GPU path. Nothing here touches the GPU.
use anyhow::{anyhow, Result};
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::{Path, PathBuf};

extern "C" {
    fn ironmlx_ane_open(err: *mut c_char, cap: usize) -> i32;
    fn ironmlx_ane_surface_create(
        rows: usize,
        width: usize,
        elem: usize,
        format: u32,
        row_bytes: *mut usize,
        alloc_bytes: *mut usize,
        err: *mut c_char,
        cap: usize,
    ) -> *mut c_void;
    fn ironmlx_ane_surface_base(surface: *mut c_void) -> *mut c_void;
    fn ironmlx_ane_surface_release(surface: *mut c_void);
    fn ironmlx_ane_program_load(
        dir: *const c_char,
        key: *const c_char,
        err: *mut c_char,
        cap: usize,
    ) -> *mut c_void;
    fn ironmlx_ane_procedure_count(program: *mut c_void) -> i32;
    fn ironmlx_ane_procedure_index(program: *mut c_void, name: *const c_char) -> i32;
    fn ironmlx_ane_evaluate(
        program: *mut c_void,
        procedure: i32,
        names: *const *const c_char,
        surfaces: *const *mut c_void,
        count: i32,
        output: *mut c_void,
        err: *mut c_char,
        cap: usize,
    ) -> i32;
    fn ironmlx_ane_program_release(program: *mut c_void);
}

fn call_error(buffer: &[c_char]) -> String {
    // SAFETY: the native side writes a NUL-terminated message into the buffer.
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// Whether the ANE can be programmed on this machine; the reason when not.
pub fn availability() -> std::result::Result<(), String> {
    let mut err = [0 as c_char; 512];
    // SAFETY: valid output buffer of the stated capacity.
    if unsafe { ironmlx_ane_open(err.as_mut_ptr(), err.len()) } == 1 {
        Ok(())
    } else {
        Err(call_error(&err))
    }
}

/// Element type of an ANE tensor surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elem {
    I8,
    F16,
}

impl Elem {
    fn bytes(self) -> usize {
        match self {
            Self::I8 => 1,
            Self::F16 => 2,
        }
    }
    fn pixel_format(self) -> u32 {
        match self {
            Self::I8 => 0x4c30_3038,  // 'L008'
            Self::F16 => 0x4c30_3068, // 'L00h'
        }
    }
    /// MIL element type name.
    pub fn mil(self) -> &'static str {
        match self {
            Self::I8 => "int8",
            Self::F16 => "fp16",
        }
    }
}

/// A `rows x width` tensor in an IOSurface (rows padded to 64 bytes), shared
/// by the CPU, the GPU and the ANE without copies.
pub struct Surface {
    ptr: *mut c_void,
    pub rows: usize,
    pub width: usize,
    pub elem: Elem,
    pub row_bytes: usize,
    pub alloc_bytes: usize,
}

// SAFETY: an IOSurface is a reference-counted kernel object usable from any thread.
unsafe impl Send for Surface {}
unsafe impl Sync for Surface {}

impl Surface {
    pub fn new(rows: usize, width: usize, elem: Elem) -> Result<Self> {
        let mut err = [0 as c_char; 512];
        let (mut row_bytes, mut alloc_bytes) = (0usize, 0usize);
        // SAFETY: valid output pointers and buffer.
        let ptr = unsafe {
            ironmlx_ane_surface_create(
                rows,
                width,
                elem.bytes(),
                elem.pixel_format(),
                &mut row_bytes,
                &mut alloc_bytes,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if ptr.is_null() {
            return Err(anyhow!(call_error(&err)));
        }
        Ok(Self {
            ptr,
            rows,
            width,
            elem,
            row_bytes,
            alloc_bytes,
        })
    }

    /// Base address of the surface memory (page aligned, `alloc_bytes` long).
    pub fn base(&self) -> *mut u8 {
        // SAFETY: the surface is alive for `self`'s lifetime.
        unsafe { ironmlx_ane_surface_base(self.ptr) as *mut u8 }
    }

    /// Stride in elements between rows (what a MIL `tensor_buffer` names).
    pub fn stride(&self) -> usize {
        self.row_bytes / self.elem.bytes()
    }

    /// MIL `tensor_buffer` type of the first `rows` rows of this surface.
    pub fn buffer_type(&self, rows: usize) -> String {
        let stride = self.stride();
        let plane = self.rows * stride;
        format!(
            "tensor_buffer<{}, shape=[1, 1, {rows}, {}], strides=[{plane}, {plane}, {stride}, 1], interleave_factors=[1, 1, 1, 1]>",
            self.elem.mil(),
            self.width
        )
    }

    fn raw(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        // SAFETY: releases the reference taken at creation.
        unsafe { ironmlx_ane_surface_release(self.ptr) }
    }
}

fn fnv1a(data: &[u8], mut hash: u64) -> u64 {
    for &b in data {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    hash
}

fn cache_root() -> PathBuf {
    let mut buffer = vec![0 as c_char; 1024];
    // SAFETY: valid buffer; confstr writes a NUL-terminated path.
    let len = unsafe { libc_confstr(buffer.as_mut_ptr(), buffer.len()) };
    if len == 0 {
        return std::env::temp_dir().join("ironmlx-ane");
    }
    // SAFETY: NUL-terminated by confstr.
    let dir = unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    Path::new(&dir).join("ironmlx-ane")
}

extern "C" {
    fn confstr(name: i32, buf: *mut c_char, len: usize) -> usize;
}

/// `_CS_DARWIN_USER_CACHE_DIR`.
unsafe fn libc_confstr(buf: *mut c_char, len: usize) -> usize {
    const CS_DARWIN_USER_CACHE_DIR: i32 = 65538;
    confstr(CS_DARWIN_USER_CACHE_DIR, buf, len)
}

/// A compiled and loaded ANE program (one or more MIL functions).
pub struct Program {
    ptr: *mut c_void,
}

// SAFETY: the service handle is used from one evaluating thread at a time
// (callers serialize evaluations), and the object itself is thread-safe to hold.
unsafe impl Send for Program {}
unsafe impl Sync for Program {}

impl Program {
    /// Compile (or reuse the service's cached compile) and load `mil` with the
    /// constant `weights` blob. The cache key hashes both.
    pub fn load(mil: &str, weights: &[u8]) -> Result<Self> {
        let key = format!(
            "{:016x}",
            fnv1a(weights, fnv1a(mil.as_bytes(), 14_695_981_039_346_656_037))
        );
        let dir = cache_root().join(&key);
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join("model.mil"), mil)?;
        std::fs::write(dir.join("weights.bin"), weights)?;
        let dir_c = CString::new(dir.to_string_lossy().as_bytes())?;
        let key_c = CString::new(key)?;
        let mut err = [0 as c_char; 1024];
        // SAFETY: valid NUL-terminated strings and error buffer.
        let ptr = unsafe {
            ironmlx_ane_program_load(dir_c.as_ptr(), key_c.as_ptr(), err.as_mut_ptr(), err.len())
        };
        if ptr.is_null() {
            return Err(anyhow!(call_error(&err)));
        }
        Ok(Self { ptr })
    }

    pub fn procedures(&self) -> usize {
        // SAFETY: live program handle.
        unsafe { ironmlx_ane_procedure_count(self.ptr) as usize }
    }

    /// Index of the MIL function called `name`.
    pub fn procedure(&self, name: &str) -> Option<usize> {
        let name = CString::new(name).ok()?;
        // SAFETY: live program handle, NUL-terminated name.
        let index = unsafe { ironmlx_ane_procedure_index(self.ptr, name.as_ptr()) };
        (index >= 0).then_some(index as usize)
    }

    /// Run procedure `procedure` synchronously with its inputs bound by MIL
    /// parameter name, writing `output`.
    pub fn evaluate(
        &self,
        procedure: usize,
        inputs: &[(&str, &Surface)],
        output: &Surface,
    ) -> Result<()> {
        let names: Vec<CString> = inputs
            .iter()
            .map(|(n, _)| CString::new(*n))
            .collect::<std::result::Result<_, _>>()?;
        let name_ptrs: Vec<*const c_char> = names.iter().map(|n| n.as_ptr()).collect();
        let surfaces: Vec<*mut c_void> = inputs.iter().map(|(_, s)| s.raw()).collect();
        let mut err = [0 as c_char; 1024];
        // SAFETY: arrays of `inputs.len()` valid entries; surfaces alive for the call.
        let ok = unsafe {
            ironmlx_ane_evaluate(
                self.ptr,
                procedure as i32,
                name_ptrs.as_ptr(),
                surfaces.as_ptr(),
                inputs.len() as i32,
                output.raw(),
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if ok == 1 {
            Ok(())
        } else {
            Err(anyhow!(call_error(&err)))
        }
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        // SAFETY: unloads and frees the program handle exactly once.
        unsafe { ironmlx_ane_program_release(self.ptr) }
    }
}

/// An empty constant blob (programs whose tensors are all runtime inputs).
pub fn empty_weights() -> Vec<u8> {
    let mut blob = vec![0u8; 128];
    blob[4..8].copy_from_slice(&2u32.to_le_bytes());
    blob
}

/// Offset of the one constant of [`fp16_constant_weights`], for
/// `BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64))`.
pub const CONSTANT_OFFSET: u64 = 64;

/// A constant blob (MIL weight storage version 2) holding one fp16 tensor
/// (`values` as IEEE half bit patterns), its record at [`CONSTANT_OFFSET`].
pub fn fp16_constant_weights(values: &[u16]) -> Vec<u8> {
    const DATA: usize = 128;
    let mut blob = vec![0u8; DATA + values.len() * 2];
    blob[0..4].copy_from_slice(&1u32.to_le_bytes()); // blobs
    blob[4..8].copy_from_slice(&2u32.to_le_bytes()); // storage version
    let record = CONSTANT_OFFSET as usize;
    blob[record..record + 4].copy_from_slice(&0xdead_beefu32.to_le_bytes());
    blob[record + 4..record + 8].copy_from_slice(&1u32.to_le_bytes()); // fp16
    blob[record + 8..record + 16].copy_from_slice(&((values.len() * 2) as u64).to_le_bytes());
    blob[record + 16..record + 24].copy_from_slice(&(DATA as u64).to_le_bytes());
    for (i, v) in values.iter().enumerate() {
        blob[DATA + 2 * i..DATA + 2 * i + 2].copy_from_slice(&v.to_le_bytes());
    }
    blob
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Needs an ANE: a 64 x 64 fp16 add through two named input surfaces.
    #[test]
    #[ignore]
    fn adds_two_fp16_surfaces() -> Result<()> {
        if let Err(reason) = availability() {
            eprintln!("ANE unavailable: {reason}");
            return Ok(());
        }
        let a = Surface::new(64, 64, Elem::F16)?;
        let b = Surface::new(64, 64, Elem::F16)?;
        let y = Surface::new(64, 64, Elem::F16)?;
        let fill = |s: &Surface, f: &dyn Fn(usize, usize) -> f32| {
            for r in 0..64 {
                for c in 0..64 {
                    // SAFETY: inside the surface allocation.
                    unsafe {
                        *(s.base().add(r * s.row_bytes) as *mut u16).add(c) = f16_bits(f(r, c));
                    }
                }
            }
        };
        fill(&a, &|r, _c| (r as f32) * 0.5);
        fill(&b, &|_r, c| (c as f32) * 0.25);
        let mil = format!(
            "program(1.3)\n{{\n    func f<ios18>({} a, {} b) {{\n        tensor<fp16, [1, 1, 64, 64]> at = tensor_buffer_to_tensor<ios17>(input = a);\n        tensor<fp16, [1, 1, 64, 64]> bt = tensor_buffer_to_tensor<ios17>(input = b);\n        tensor<fp16, [1, 1, 64, 64]> s = add(x = at, y = bt);\n        {} y = tensor_to_tensor_buffer<ios17>(input = s, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>([{p}, {p}, {st}, 1]));\n    }} -> (y);\n}}\n",
            a.buffer_type(64),
            b.buffer_type(64),
            y.buffer_type(64),
            p = y.rows * y.stride(),
            st = y.stride()
        );
        let program = Program::load(&mil, &empty_weights())?;
        assert_eq!(program.procedures(), 1);
        program.evaluate(0, &[("b", &b), ("a", &a)], &y)?;
        for r in [0usize, 7, 63] {
            for c in [0usize, 5, 63] {
                // SAFETY: inside the surface allocation.
                let v = f16_value(unsafe { *(y.base().add(r * y.row_bytes) as *const u16).add(c) });
                let want = r as f32 * 0.5 + c as f32 * 0.25;
                assert!((v - want).abs() < 1e-2, "y[{r}][{c}] = {v}, want {want}");
            }
        }
        Ok(())
    }

    fn f16_bits(v: f32) -> u16 {
        // Exact for the small multiples of 1/4 used here.
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        if v == 0.0 {
            return sign;
        }
        let mant = (bits >> 13) & 0x3ff;
        sign | (((exp - 127 + 15) as u16) << 10) | mant as u16
    }

    fn f16_value(h: u16) -> f32 {
        let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exp = ((h >> 10) & 0x1f) as i32;
        let mant = (h & 0x3ff) as f32;
        if exp == 0 {
            return sign * mant * 2f32.powi(-24);
        }
        sign * (1.0 + mant / 1024.0) * 2f32.powi(exp - 15)
    }
}
