use mlx_sys as sys;

use crate::array::Array;
use crate::error::{check, Result};

// ── VectorArray ─────────────────────────────────────────────────────────────

/// A dynamically-sized vector of [`Array`] values, wrapping `mlx_vector_array`.
pub struct VectorArray(sys::mlx_vector_array);

impl VectorArray {
    pub fn new() -> Self {
        VectorArray(unsafe { sys::mlx_vector_array_new() })
    }

    /// Build from a slice of `Array` references.
    pub fn from_arrays(arrays: &[&Array]) -> Self {
        let raws: Vec<sys::mlx_array> = arrays.iter().map(|a| a.as_raw()).collect();
        VectorArray(unsafe { sys::mlx_vector_array_new_data(raws.as_ptr(), raws.len()) })
    }

    /// Append a single array.
    pub fn push(&self, a: &Array) -> Result<()> {
        check(unsafe { sys::mlx_vector_array_append_value(self.0, a.as_raw()) })
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        unsafe { sys::mlx_vector_array_size(self.0) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get element at index.
    pub fn get(&self, idx: usize) -> Result<Array> {
        let mut res = Array::new_empty();
        check(unsafe { sys::mlx_vector_array_get(res.as_raw_mut(), self.0, idx) })?;
        Ok(res)
    }

    pub(crate) fn as_raw(&self) -> sys::mlx_vector_array {
        self.0
    }

    /// Wrap an already-owned raw `mlx_vector_array` handle.
    pub(crate) fn from_raw(raw: sys::mlx_vector_array) -> Self {
        VectorArray(raw)
    }
}

impl Drop for VectorArray {
    fn drop(&mut self) {
        unsafe { sys::mlx_vector_array_free(self.0) };
    }
}

impl Default for VectorArray {
    fn default() -> Self {
        Self::new()
    }
}

// ── MapStringToArray ────────────────────────────────────────────────────────

/// A string-keyed map of [`Array`] values, wrapping `mlx_map_string_to_array`.
pub struct MapStringToArray(sys::mlx_map_string_to_array);

impl MapStringToArray {
    pub fn new() -> Self {
        MapStringToArray(unsafe { sys::mlx_map_string_to_array_new() })
    }

    pub fn insert(&self, key: &str, value: &Array) -> Result<()> {
        let c_key =
            std::ffi::CString::new(key).map_err(|e| crate::error::Error::Mlx(e.to_string()))?;
        check(unsafe {
            sys::mlx_map_string_to_array_insert(self.0, c_key.as_ptr(), value.as_raw())
        })
    }

    pub fn get(&self, key: &str) -> Result<Array> {
        let c_key =
            std::ffi::CString::new(key).map_err(|e| crate::error::Error::Mlx(e.to_string()))?;
        let mut res = Array::new_empty();
        check(unsafe {
            sys::mlx_map_string_to_array_get(res.as_raw_mut(), self.0, c_key.as_ptr())
        })?;
        Ok(res)
    }

    /// Iterate over all (key, array) pairs.
    pub fn iter(&self) -> MapStringToArrayIter<'_> {
        MapStringToArrayIter {
            inner: unsafe { sys::mlx_map_string_to_array_iterator_new(self.0) },
            _marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn as_raw(&self) -> sys::mlx_map_string_to_array {
        self.0
    }

    pub(crate) fn as_raw_mut(&mut self) -> *mut sys::mlx_map_string_to_array {
        &mut self.0
    }
}

impl Drop for MapStringToArray {
    fn drop(&mut self) {
        unsafe { sys::mlx_map_string_to_array_free(self.0) };
    }
}

impl Default for MapStringToArray {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over [`MapStringToArray`].
pub struct MapStringToArrayIter<'a> {
    inner: sys::mlx_map_string_to_array_iterator,
    _marker: std::marker::PhantomData<&'a MapStringToArray>,
}

impl<'a> Iterator for MapStringToArrayIter<'a> {
    type Item = (String, Array);

    fn next(&mut self) -> Option<Self::Item> {
        let mut key: *const std::os::raw::c_char = std::ptr::null();
        let mut value = Array::new_empty();
        let rc = unsafe {
            sys::mlx_map_string_to_array_iterator_next(&mut key, value.as_raw_mut(), self.inner)
        };
        if rc != 0 || key.is_null() {
            return None;
        }
        let key_str = unsafe { std::ffi::CStr::from_ptr(key) }
            .to_string_lossy()
            .into_owned();
        Some((key_str, value))
    }
}

impl Drop for MapStringToArrayIter<'_> {
    fn drop(&mut self) {
        unsafe { sys::mlx_map_string_to_array_iterator_free(self.inner) };
    }
}

// ── MapStringToString ───────────────────────────────────────────────────────

/// A string-keyed map of string values, wrapping `mlx_map_string_to_string`.
pub struct MapStringToString(sys::mlx_map_string_to_string);

impl MapStringToString {
    pub fn new() -> Self {
        MapStringToString(unsafe { sys::mlx_map_string_to_string_new() })
    }

    pub fn insert(&self, key: &str, value: &str) -> Result<()> {
        let c_key =
            std::ffi::CString::new(key).map_err(|e| crate::error::Error::Mlx(e.to_string()))?;
        let c_val =
            std::ffi::CString::new(value).map_err(|e| crate::error::Error::Mlx(e.to_string()))?;
        check(unsafe {
            sys::mlx_map_string_to_string_insert(self.0, c_key.as_ptr(), c_val.as_ptr())
        })
    }

    pub fn get(&self, key: &str) -> Result<String> {
        let c_key =
            std::ffi::CString::new(key).map_err(|e| crate::error::Error::Mlx(e.to_string()))?;
        let mut val: *const std::os::raw::c_char = std::ptr::null();
        check(unsafe { sys::mlx_map_string_to_string_get(&mut val, self.0, c_key.as_ptr()) })?;
        if val.is_null() {
            return Err(crate::error::Error::Mlx("null string value".into()));
        }
        Ok(unsafe { std::ffi::CStr::from_ptr(val) }
            .to_string_lossy()
            .into_owned())
    }

    /// Iterate over all (key, value) pairs.
    pub fn iter(&self) -> MapStringToStringIter<'_> {
        MapStringToStringIter {
            inner: unsafe { sys::mlx_map_string_to_string_iterator_new(self.0) },
            _marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn as_raw(&self) -> sys::mlx_map_string_to_string {
        self.0
    }

    pub(crate) fn as_raw_mut(&mut self) -> *mut sys::mlx_map_string_to_string {
        &mut self.0
    }
}

impl Drop for MapStringToString {
    fn drop(&mut self) {
        unsafe { sys::mlx_map_string_to_string_free(self.0) };
    }
}

impl Default for MapStringToString {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over [`MapStringToString`].
pub struct MapStringToStringIter<'a> {
    inner: sys::mlx_map_string_to_string_iterator,
    _marker: std::marker::PhantomData<&'a MapStringToString>,
}

impl<'a> Iterator for MapStringToStringIter<'a> {
    type Item = (String, String);

    fn next(&mut self) -> Option<Self::Item> {
        let mut key: *const std::os::raw::c_char = std::ptr::null();
        let mut val: *const std::os::raw::c_char = std::ptr::null();
        let rc =
            unsafe { sys::mlx_map_string_to_string_iterator_next(&mut key, &mut val, self.inner) };
        if rc != 0 || key.is_null() {
            return None;
        }
        let key_str = unsafe { std::ffi::CStr::from_ptr(key) }
            .to_string_lossy()
            .into_owned();
        let val_str = if val.is_null() {
            String::new()
        } else {
            unsafe { std::ffi::CStr::from_ptr(val) }
                .to_string_lossy()
                .into_owned()
        };
        Some((key_str, val_str))
    }
}

impl Drop for MapStringToStringIter<'_> {
    fn drop(&mut self) {
        unsafe { sys::mlx_map_string_to_string_iterator_free(self.inner) };
    }
}
