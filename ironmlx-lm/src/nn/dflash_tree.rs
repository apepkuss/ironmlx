//! Diagnostic B1 flat tree: compute each node once, commit only its accepted path.
use crate::Result;
use mlx::{Array, MetalKernel, Shape, StreamOrDevice};
use std::{
    cell::{OnceCell, RefCell},
    rc::Rc,
    sync::OnceLock,
};

pub(crate) struct Plan {
    pub paths: Vec<Vec<i32>>,
    parents: Array,
    attention_groups: OnceCell<(i32, Vec<AttentionGroup>)>,
    lane_attention: OnceCell<super::m5_tree_attention::Layout>,
}
pub(crate) struct AttentionGroup {
    pub rows: Vec<usize>,
    pub queries: Array,
    pub keys: Array,
}
impl Plan {
    pub fn new(parents: &[i32]) -> Result<Self> {
        anyhow::ensure!(
            !parents.is_empty() && parents.len() <= 16 && parents[0] == -1,
            "flat tree requires 1..16 rows and one root"
        );
        let mut paths: Vec<Vec<i32>> = Vec::new();
        for (i, &p) in parents.iter().enumerate() {
            anyhow::ensure!(
                i == 0 || (p >= 0 && (p as usize) < i),
                "flat tree parents must precede children"
            );
            let mut path = if p < 0 {
                Vec::new()
            } else {
                paths[p as usize].clone()
            };
            path.push(i as i32);
            paths.push(path);
        }
        Ok(Self {
            paths,
            parents: (parents, &[parents.len() as i32][..]).try_into()?,
            attention_groups: OnceCell::new(),
            lane_attention: OnceCell::new(),
        })
    }
    pub fn lane_attention(&self, base: i32) -> Result<&super::m5_tree_attention::Layout> {
        if self.lane_attention.get().is_none() {
            let _ = self
                .lane_attention
                .set(super::m5_tree_attention::Layout::new(&self.paths, base)?);
        }
        Ok(self.lane_attention.get().unwrap())
    }
    pub fn attention_groups(&self, base: i32) -> Result<&[AttentionGroup]> {
        if self.attention_groups.get().is_none() {
            let mut depths = std::collections::BTreeMap::<usize, Vec<usize>>::new();
            for (row, path) in self.paths.iter().enumerate() {
                depths.entry(path.len()).or_default().push(row);
            }
            let mut groups = Vec::new();
            for (depth, rows) in depths {
                let query_indices = rows.iter().map(|&r| r as i32).collect::<Vec<_>>();
                let mut key_indices = Vec::new();
                for &row in &rows {
                    key_indices.extend(0..base);
                    key_indices.extend(self.paths[row].iter().map(|&r| base + r));
                }
                groups.push(AttentionGroup {
                    queries: (query_indices.as_slice(), &[rows.len() as i32, 1, 1, 1][..])
                        .try_into()?,
                    keys: (
                        key_indices.as_slice(),
                        &[rows.len() as i32, 1, base + depth as i32, 1][..],
                    )
                        .try_into()?,
                    rows,
                });
            }
            let _ = self.attention_groups.set((base, groups));
        }
        let (cached_base, groups) = self.attention_groups.get().unwrap();
        anyhow::ensure!(
            *cached_base == base,
            "tree attention offsets differ across layers"
        );
        Ok(groups)
    }
    pub fn positions(&self, start: i32) -> Result<Array> {
        let values = self
            .paths
            .iter()
            .map(|p| start + p.len() as i32 - 1)
            .collect::<Vec<_>>();
        let ids: Array = (values.as_slice(), &[1, 1, values.len() as i32][..]).try_into()?;
        Ok(mlx::ops::shape::broadcast_to(
            &ids,
            &[3, 1, values.len() as i32][..],
        )?)
    }
    pub fn conv_windows(&self, input: &Array, keep: i32, target: StreamOrDevice) -> Result<Array> {
        let mut indices = Vec::new();
        for path in &self.paths {
            let rows = (0..keep)
                .chain(path.iter().map(|&r| keep + r))
                .collect::<Vec<_>>();
            indices.extend_from_slice(&rows[rows.len() - keep as usize - 1..]);
        }
        let indices: Array = (indices.as_slice(), &[indices.len() as i32][..]).try_into()?;
        let x = mlx::ops::indexing::take_on(input, &indices, 1, target)?;
        Ok(x.reshape_on(
            (
                self.paths.len() as i32,
                keep + 1,
                input.shape().as_slice()[2],
            ),
            target,
        )?)
    }
}
thread_local! { static PLAN: RefCell<Option<Rc<Plan>>> = const { RefCell::new(None) }; }
pub(crate) struct Scope;
impl Drop for Scope {
    fn drop(&mut self) {
        PLAN.with(|p| *p.borrow_mut() = None);
    }
}
pub(crate) fn enter(plan: Plan) -> Result<Scope> {
    PLAN.with(|p| {
        anyhow::ensure!(p.borrow().is_none(), "nested flat tree scope");
        *p.borrow_mut() = Some(Rc::new(plan));
        Ok(Scope)
    })
}
pub(crate) fn current() -> Option<Rc<Plan>> {
    PLAN.with(|p| p.borrow().clone())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn recurrent(
    plan: &Plan,
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    target: StreamOrDevice,
) -> Result<Array> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = KERNEL.get_or_init(|| {
        MetalKernel::builder("ironmlx_flat_tree_gdn_v1")
            .inputs(&["q", "k", "v", "g", "beta", "state_in", "parents", "nodes"])
            .outputs(&["y"])
            .source(include_str!("dflash_tree_gdn.metal"))
            .build()
            .expect("build experimental tree GDN")
    });
    let (w, hk, dk) = (
        q.shape().as_slice()[1],
        q.shape().as_slice()[2],
        q.shape().as_slice()[3],
    );
    let (hv, dv) = (v.shape().as_slice()[2], v.shape().as_slice()[3]);
    let arrays = [q, k, v, g, beta, state]
        .iter()
        .map(|a| mlx::ops::shape::contiguous_on(a, false, target))
        .collect::<mlx::Result<Vec<_>>>()?;
    let nodes: Array = (&[w][..], &[1][..]).try_into()?;
    let mut inputs = arrays.iter().collect::<Vec<_>>();
    inputs.extend([&plan.parents, &nodes]);
    Ok(kernel
        .dispatch_builder()
        .inputs(&inputs)
        .output_shapes(&[Shape::from(&[1, w, hv, dv][..])])
        .output_dtypes(&[v.dtype()])
        .grid(32, dv, hv)
        .threadgroup(32, 4, 1)
        .template_int("Dk", dk)
        .template_int("Dv", dv)
        .template_int("Hk", hk)
        .template_int("Hv", hv)
        .template_dtype("InT", v.dtype())
        .stream(target)
        .dispatch()?
        .take_at(0)?)
}
