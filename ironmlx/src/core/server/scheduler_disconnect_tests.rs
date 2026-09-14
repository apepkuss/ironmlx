//! HTTP disconnect contract against the native scheduler actor.
use crate::core::scheduler_actor::test_support::{
    mk_req, wait_for_scheduler_resources_to_be_released, SchedulerActorFakeModel,
};
use crate::core::scheduler_actor::{spawn_scheduler_actor, SchedulerActorHandle, SchedulerCommand};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use std::time::Duration;
use tokio::sync::{oneshot, Mutex};

#[derive(Clone)]
struct SseDisconnectContractState {
    scheduler: SchedulerActorHandle,
    terminal_events: Arc<AtomicU64>,
}

async fn scheduler_disconnect_contract_stream(
    axum::extract::State(state): axum::extract::State<SseDisconnectContractState>,
) -> axum::response::Response {
    let (reply_tx, reply_rx) = oneshot::channel();
    state
        .scheduler
        .cmd_tx
        .send(SchedulerCommand::Admit {
            request: mk_req(11),
            reply_tx,
        })
        .await
        .expect("send disconnect-contract admission");
    let mut event_rx = reply_rx
        .await
        .expect("disconnect-contract admission reply")
        .expect("disconnect-contract admission accepted")
        .event_rx;

    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if state.scheduler.b_active.load(Ordering::Relaxed) == 1
                && state
                    .scheduler
                    .kv_cache_active_bytes
                    .load(Ordering::Relaxed)
                    > 0
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("scheduler resources must be live before returning SSE response");

    let (tx, rx, disconnect) = crate::core::server::api_transport::disconnect_aware_sse_channel(2);
    let terminal_events = state.terminal_events;
    tokio::spawn(async move {
        if tx
            .send(Ok(axum::body::Bytes::from_static(
                b"data: {\"type\":\"started\"}\n\n",
            )))
            .await
            .is_err()
        {
            return;
        }

        while let Some(event) =
            crate::core::server::api_transport::recv_or_disconnect(&disconnect, &mut event_rx).await
        {
            let terminal = event.finish_reason.is_some();
            if terminal {
                terminal_events.fetch_add(1, Ordering::Relaxed);
            }
            let frame = format!("data: {{\"token\":{}}}\n\n", event.token);
            if tx.send(Ok(axum::body::Bytes::from(frame))).await.is_err() {
                return;
            }
            if terminal {
                return;
            }
        }
    });

    crate::core::server::api_transport::disconnect_aware_sse_response(rx)
}

async fn disconnect_tcp_client_after_first_sse_frame(address: std::net::SocketAddr, path: &str) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut stream = tokio::net::TcpStream::connect(address)
        .await
        .expect("connect contract client");
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{{}}"
    );
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write contract request");

    let response = tokio::time::timeout(Duration::from_secs(2), async {
        let mut response = Vec::new();
        let mut buffer = [0_u8; 512];
        loop {
            let read = stream.read(&mut buffer).await.expect("read SSE response");
            assert!(read > 0, "SSE response closed before its first frame");
            response.extend_from_slice(&buffer[..read]);
            if response
                .windows(b"data:".len())
                .any(|part| part == b"data:")
            {
                return response;
            }
        }
    })
    .await
    .expect("first SSE frame timeout");
    let response = String::from_utf8_lossy(&response);
    assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");

    drop(stream);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn http_sse_disconnect_releases_scheduler_resources_for_all_public_protocols() {
    use axum::{routing::post, Router};

    let model = Arc::new(Mutex::new(SchedulerActorFakeModel::with_forward_delay(
        Duration::from_millis(100),
    )));
    let handle = spawn_scheduler_actor(
        model,
        1,
        Duration::from_millis(1),
        1,
        32,
        256,
        crate::core::memory_budget::test_meta_qwen35(),
    )
    .expect("spawn disconnect-contract scheduler");
    let terminal_events = Arc::new(AtomicU64::new(0));
    let state = SseDisconnectContractState {
        scheduler: handle.clone(),
        terminal_events: terminal_events.clone(),
    };
    let router = Router::new()
        .route(
            "/v1/chat/completions",
            post(scheduler_disconnect_contract_stream),
        )
        .route("/v1/responses", post(scheduler_disconnect_contract_stream))
        .route("/v1/messages", post(scheduler_disconnect_contract_stream))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .await
        .expect("bind disconnect-contract server");
    let address = listener.local_addr().expect("contract server address");
    let server = tokio::spawn(async move {
        axum::serve(listener, router)
            .await
            .expect("serve disconnect-contract router");
    });

    for path in ["/v1/chat/completions", "/v1/responses", "/v1/messages"] {
        let terminal_before = terminal_events.load(Ordering::Relaxed);
        disconnect_tcp_client_after_first_sse_frame(address, path).await;
        assert_eq!(handle.b_active.load(Ordering::Relaxed), 1);
        assert!(handle.kv_cache_active_bytes.load(Ordering::Relaxed) > 0);

        wait_for_scheduler_resources_to_be_released(&handle).await;
        assert_eq!(
            terminal_events.load(Ordering::Relaxed),
            terminal_before,
            "{path} emitted a terminal SSE event after disconnect"
        );
    }

    assert_eq!(handle.admit_count.load(Ordering::Relaxed), 3);
    server.abort();
    let _ = server.await;
}
