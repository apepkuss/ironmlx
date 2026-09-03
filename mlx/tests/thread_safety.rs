use mlx::Array;

fn assert_send<T: Send>() {}

#[test]
fn array_is_send() {
    assert_send::<Array>();
}

#[test]
fn array_is_not_sync() {
    use static_assertions::assert_not_impl_any;
    assert_not_impl_any!(Array: Sync);
}
