//! Stub for typestate dispatch builder. Full implementation in T5.

#![allow(dead_code)]

use std::sync::Arc;

use crate::Dtype;

pub struct Unset;
pub struct Set;

#[derive(Debug, Clone)]
pub enum TemplateArg {
    Int(i32),
    Bool(bool),
    Dtype(Dtype),
}

pub struct DispatchBuilder<I, OS, OD, G, TG> {
    _kernel: Arc<super::MetalKernelInner>,
    _markers: std::marker::PhantomData<(I, OS, OD, G, TG)>,
}

impl DispatchBuilder<Unset, Unset, Unset, Unset, Unset> {
    pub(crate) fn new(kernel: Arc<super::MetalKernelInner>) -> Self {
        Self {
            _kernel: kernel,
            _markers: std::marker::PhantomData,
        }
    }
}
