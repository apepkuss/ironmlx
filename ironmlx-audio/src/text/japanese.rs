use crate::{error::invalid, Result, SessionControl};
use mecab_sys::ffi;
use std::{
    ffi::{CStr, CString},
    os::unix::ffi::OsStrExt,
    path::Path,
    ptr::NonNull,
};
// Owned mutable MeCab tagger. No Send/Sync implementation; stays on its worker.
pub(super) struct Japanese {
    tagger: NonNull<ffi::mecab_t>,
}
impl Drop for Japanese {
    fn drop(&mut self) {
        // SAFETY: this instance exclusively owns the tagger created by mecab_new.
        unsafe {
            ffi::mecab_destroy(self.tagger.as_ptr());
        }
    }
}
fn native_error(tagger: *mut ffi::mecab_t) -> crate::AudioError {
    // SAFETY: tagger is either a live owned tagger or null (global creation error).
    let ptr = unsafe { ffi::mecab_strerror(tagger) };
    let message = if ptr.is_null() {
        "MeCab failed".into()
    } else {
        // SAFETY: MeCab returns a NUL-terminated error, copied before another call.
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    };
    invalid("japanese", message)
}
impl Japanese {
    pub fn load(root: &Path) -> Result<Self> {
        for name in ["sys.dic", "unk.dic", "matrix.bin", "char.bin", "dicrc"] {
            super::verify_resource(
                &root.join(name),
                "unidic-lite",
                &format!("unidic-lite-1.0.8/unidic_lite/dicdir/{name}"),
            )?;
        }
        let root = root.canonicalize()?;
        // Use argc/argv directly: MeCab's string parser has no quoting support.
        let args = [
            b"ironmlx-audio".to_vec(),
            b"-d".to_vec(),
            root.as_os_str().as_bytes().to_vec(),
            b"-r".to_vec(),
            root.join("dicrc").as_os_str().as_bytes().to_vec(),
        ];
        let args: std::result::Result<Vec<_>, _> = args.into_iter().map(CString::new).collect();
        let args = args.map_err(|e| invalid("dictionary", e.to_string()))?;
        let mut pointers: Vec<_> = args.iter().map(|s| s.as_ptr().cast_mut()).collect();
        // SAFETY: argv points to live NUL-terminated strings. MeCab reads/copies
        // options during creation, does not mutate argv, and returns owned storage.
        let raw = unsafe { ffi::mecab_new(pointers.len() as i32, pointers.as_mut_ptr()) };
        let tagger = NonNull::new(raw).ok_or_else(|| native_error(std::ptr::null_mut()))?;
        Ok(Self { tagger })
    }
    /// Preserve literal-space spans, tokenize all other spans, and join surface forms.
    pub fn process(&mut self, text: &str, control: &dyn SessionControl) -> Result<String> {
        let mut output = String::new();
        let mut start = 0;
        let mut was_space = text.starts_with(' ');
        for (index, ch) in text
            .char_indices()
            .chain(std::iter::once((text.len(), ' ')))
        {
            let space = ch == ' ';
            if (space != was_space || index == text.len()) && index > start {
                control.check()?;
                let part = &text[start..index];
                if part.trim().is_empty() {
                    output.push_str(part);
                } else {
                    // SAFETY: part remains live while traversing nodes; exclusive
                    // access prevents another parse from invalidating native storage.
                    let mut node = unsafe {
                        ffi::mecab_sparse_tonode2(
                            self.tagger.as_ptr(),
                            part.as_ptr().cast(),
                            part.len(),
                        )
                    };
                    if node.is_null() {
                        return Err(native_error(self.tagger.as_ptr()));
                    }
                    let mut surfaces = Vec::new();
                    while !node.is_null() {
                        // SAFETY: node belongs to the current MeCab lattice, valid
                        // until next parse or tagger destruction. Neither occurs here.
                        let n = unsafe { &*node };
                        if u32::from(n.stat) != ffi::MECAB_BOS_NODE
                            && u32::from(n.stat) != ffi::MECAB_EOS_NODE
                        {
                            // SAFETY: MeCab surfaces have exactly length bytes (no NUL).
                            let bytes = unsafe {
                                std::slice::from_raw_parts(
                                    n.surface.cast::<u8>(),
                                    usize::from(n.length),
                                )
                            };
                            surfaces.push(
                                std::str::from_utf8(bytes)
                                    .map_err(|e| invalid("japanese", e.to_string()))?
                                    .to_owned(),
                            );
                        }
                        node = n.next;
                    }
                    output.push_str(&surfaces.join(" "));
                }
                start = index;
                was_space = space;
            }
        }
        Ok(output)
    }
}
