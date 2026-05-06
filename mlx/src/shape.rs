//! `Shape` newtype + `IntoShape` trait.

use smallvec::SmallVec;

/// A tensor shape: ordered list of non-negative dimension sizes. Rank-0 is a scalar.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Shape(SmallVec<[i32; 8]>);

impl Shape {
    /// Empty shape — represents a scalar (rank 0).
    pub fn new() -> Self {
        Self(SmallVec::new())
    }

    /// Number of dimensions (= 0 for scalar).
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Total element count: product of dims. Rank-0 (scalar) returns 1.
    ///
    /// Assumes every dim is non-negative. If the shape contains a negative
    /// placeholder (e.g. `-1` for reshape inference), the result is meaningless
    /// — callers must validate dims before invoking `numel()`.
    pub fn numel(&self) -> usize {
        self.0.iter().map(|&d| d as usize).product()
    }

    /// True iff rank is 0.
    pub fn is_scalar(&self) -> bool {
        self.0.is_empty()
    }

    /// View as a `&[i32]` slice.
    pub fn as_slice(&self) -> &[i32] {
        self.0.as_slice()
    }
}

impl std::ops::Deref for Shape {
    type Target = [i32];
    fn deref(&self) -> &[i32] {
        self.0.as_slice()
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{d}")?;
        }
        f.write_str("]")
    }
}

/// Owned iterator over a [`Shape`]'s dimensions.
///
/// Returned by `<Shape as IntoIterator>::into_iter`. The underlying
/// representation is intentionally hidden so the storage type (currently
/// `SmallVec`) can change without breaking downstream code.
pub struct ShapeIntoIter(smallvec::IntoIter<[i32; 8]>);

impl Iterator for ShapeIntoIter {
    type Item = i32;
    fn next(&mut self) -> Option<i32> {
        self.0.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl ExactSizeIterator for ShapeIntoIter {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl IntoIterator for Shape {
    type Item = i32;
    type IntoIter = ShapeIntoIter;
    fn into_iter(self) -> Self::IntoIter {
        ShapeIntoIter(self.0.into_iter())
    }
}

impl<'a> IntoIterator for &'a Shape {
    type Item = &'a i32;
    type IntoIter = std::slice::Iter<'a, i32>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl From<Vec<i32>> for Shape {
    fn from(v: Vec<i32>) -> Self {
        Self(SmallVec::from_vec(v))
    }
}
impl From<&[i32]> for Shape {
    fn from(s: &[i32]) -> Self {
        Self(SmallVec::from_slice(s))
    }
}
impl<const N: usize> From<[i32; N]> for Shape {
    fn from(a: [i32; N]) -> Self {
        Self(SmallVec::from_slice(&a))
    }
}
impl<const N: usize> From<&[i32; N]> for Shape {
    fn from(a: &[i32; N]) -> Self {
        Self(SmallVec::from_slice(a))
    }
}
impl From<()> for Shape {
    fn from(_: ()) -> Self {
        Shape::new()
    }
}
impl From<i32> for Shape {
    fn from(d: i32) -> Self {
        Self(SmallVec::from_slice(&[d]))
    }
}
impl From<(i32,)> for Shape {
    fn from(t: (i32,)) -> Self {
        Self(SmallVec::from_slice(&[t.0]))
    }
}
impl From<(i32, i32)> for Shape {
    fn from(t: (i32, i32)) -> Self {
        Self(SmallVec::from_slice(&[t.0, t.1]))
    }
}
impl From<(i32, i32, i32)> for Shape {
    fn from(t: (i32, i32, i32)) -> Self {
        Self(SmallVec::from_slice(&[t.0, t.1, t.2]))
    }
}
impl From<(i32, i32, i32, i32)> for Shape {
    fn from(t: (i32, i32, i32, i32)) -> Self {
        Self(SmallVec::from_slice(&[t.0, t.1, t.2, t.3]))
    }
}

/// Anything convertible to a [`Shape`]. Implemented for tuples, arrays, slices, `Vec<i32>`, and `i32`.
pub trait IntoShape {
    /// Consume `self` and produce a `Shape`.
    fn into_shape(self) -> Shape;
}

impl<T: Into<Shape>> IntoShape for T {
    fn into_shape(self) -> Shape {
        self.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_shape_is_scalar() {
        let s = Shape::new();
        assert!(s.is_scalar());
        assert_eq!(s.rank(), 0);
        assert_eq!(s.numel(), 1);
        assert_eq!(s.as_slice(), &[] as &[i32]);
    }

    #[test]
    fn rank_and_numel() {
        let s: Shape = (2, 3, 4).into();
        assert_eq!(s.rank(), 3);
        assert_eq!(s.numel(), 24);
        assert_eq!(s.as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn display_format() {
        assert_eq!(format!("{}", Shape::new()), "[]");
        assert_eq!(format!("{}", Shape::from((5,))), "[5]");
        assert_eq!(format!("{}", Shape::from((2, 3, 4))), "[2, 3, 4]");
    }

    #[test]
    fn into_shape_blanket_covers_common_inputs() {
        fn take<S: IntoShape>(s: S) -> Shape {
            s.into_shape()
        }
        assert_eq!(take(()).as_slice(), &[]);
        assert_eq!(take(5).as_slice(), &[5]);
        assert_eq!(take((2, 3)).as_slice(), &[2, 3]);
        assert_eq!(take([2, 3, 4]).as_slice(), &[2, 3, 4]);
        assert_eq!(take(&[2, 3][..]).as_slice(), &[2, 3]);
        assert_eq!(take(vec![2, 3]).as_slice(), &[2, 3]);
    }

    #[test]
    fn deref_to_slice() {
        let s: Shape = (2, 3, 4).into();
        let slice: &[i32] = &s;
        assert_eq!(slice, &[2, 3, 4]);
        assert_eq!(s.iter().sum::<i32>(), 9);
    }

    #[test]
    fn iterates() {
        let s: Shape = (1, 2, 3).into();
        let collected: Vec<i32> = s.into_iter().collect();
        assert_eq!(collected, vec![1, 2, 3]);
    }
}
