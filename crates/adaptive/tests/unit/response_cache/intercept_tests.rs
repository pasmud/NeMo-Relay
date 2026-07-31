// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for response-cache streaming commit behavior.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use nemo_relay::api::llm::LlmRequest;
use nemo_relay::api::runtime::{LlmExecutionNextFn, LlmJsonStream, LlmStreamExecutionNextFn};
use nemo_relay::error::FlowError;
use serde_json::json;
use tokio::sync::{oneshot, watch};
use tokio_stream::StreamExt;

use super::*;

#[derive(Default)]
struct FailingGetStore {
    get_calls: AtomicUsize,
    set_calls: AtomicUsize,
}

impl CacheStore for FailingGetStore {
    fn get<'a>(
        &'a self,
        _key: &'a str,
    ) -> crate::response_cache::store::BoxCacheFuture<'a, Option<Arc<CacheEntry>>> {
        self.get_calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async {
            Err(crate::error::AdaptiveError::Storage(
                "cache read unavailable".to_string(),
            ))
        })
    }

    fn set<'a>(
        &'a self,
        _key: &'a str,
        _entry: CacheEntry,
        _ttl: Duration,
    ) -> crate::response_cache::store::BoxCacheFuture<'a, ()> {
        self.set_calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok(()) })
    }

    fn health<'a>(&'a self) -> crate::response_cache::store::BoxCacheFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }

    fn backend_kind(&self) -> &'static str {
        "failing_test"
    }
}

fn cache_config() -> Arc<ResponseCacheConfig> {
    Arc::new(ResponseCacheConfig {
        namespace: "response-cache-unit-tests".to_string(),
        cache_nondeterministic: true,
        ..ResponseCacheConfig::default()
    })
}

fn chat_request(prompt: &str) -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }),
    }
}

fn terminal_chat_stream() -> LlmJsonStream {
    LlmJsonStream::new(tokio_stream::iter(vec![
        Ok::<_, FlowError>(json!({
            "id": "chatcmpl-unit-test",
            "object": "chat.completion.chunk",
            "created": 1_700_000_000_u64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": "cached"},
                "finish_reason": null,
            }],
        })),
        Ok(json!({
            "id": "chatcmpl-unit-test",
            "object": "chat.completion.chunk",
            "created": 1_700_000_000_u64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        })),
    ]))
}

#[test]
fn chat_stream_fidelity_gate_rejects_every_uncollected_non_null_shape() {
    let supported = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000_u64,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": "hello",
                "tool_calls": [{
                    "index": 0,
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"}
                }]
            },
            "finish_reason": null,
            "logprobs": null
        }],
        "usage": null
    });
    assert!(!chunk_has_uncollected_response_fields(&supported));

    for unsafe_chunk in [
        json!({"choices": [], "system_fingerprint": "fp_123"}),
        json!({"choices": [{"index": 0, "delta": {"content": ["not", "text"]}}]}),
        json!({"choices": [{"index": 0, "delta": {"tool_calls": "not-an-array"}}]}),
        json!({"choices": [{"index": 0, "delta": {"tool_calls": [{
            "index": 0,
            "function": {"name": "lookup", "arguments": "{}", "extension": true}
        }]}}]}),
    ] {
        assert!(
            chunk_has_uncollected_response_fields(&unsafe_chunk),
            "unsupported response data must veto aggregate storage: {unsafe_chunk}"
        );
    }
}

#[test]
fn malformed_stream_shapes_are_not_aggregated() {
    for malformed in [
        json!(null),
        json!({"choices": {}}),
        json!({"choices": [null]}),
        json!({"choices": [{"index": "first"}]}),
        json!({"choices": [{"finish_reason": 1}]}),
        json!({"choices": [{"unsupported": true}]}),
        json!({"choices": [{"delta": "not-an-object"}]}),
        json!({"choices": [{"delta": {"tool_calls": [null]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"id": 1}]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"unsupported": true}]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"function": "not-an-object"}]}}]}),
    ] {
        assert!(
            chunk_has_uncollected_response_fields(&malformed),
            "malformed stream chunk must not be cached: {malformed}"
        );
    }

    assert!(!chunk_has_uncollected_response_fields(&json!({
        "type": "message_delta"
    })));
    assert!(!chunk_has_uncollected_response_fields(&json!({
        "choices": null
    })));
    assert!(
        !chunk_has_uncollected_response_fields(&json!({
            "choices": [{"delta": {"tool_calls": [{"id": null}]}}]
        })),
        "null tool-call metadata is harmless when no uncollectable fields are present"
    );
}

#[tokio::test]
async fn cache_read_errors_fail_open_for_buffered_and_streaming_calls() {
    let store = Arc::new(FailingGetStore::default());
    let buffered_calls = Arc::new(AtomicUsize::new(0));
    let next: LlmExecutionNextFn = {
        let buffered_calls = Arc::clone(&buffered_calls);
        Arc::new(move |_request| {
            let buffered_calls = Arc::clone(&buffered_calls);
            Box::pin(async move {
                buffered_calls.fetch_add(1, Ordering::SeqCst);
                Ok(json!({"answer": "live"}))
            })
        })
    };

    let response = run_cache(
        "openai".to_string(),
        chat_request("buffered cache failure"),
        next,
        store.clone(),
        cache_config(),
    )
    .await
    .expect("a cache read error must not fail a buffered call");
    assert_eq!(response, json!({"answer": "live"}));
    assert_eq!(buffered_calls.load(Ordering::SeqCst), 1);

    let streaming_calls = Arc::new(AtomicUsize::new(0));
    let next: LlmStreamExecutionNextFn = {
        let streaming_calls = Arc::clone(&streaming_calls);
        Arc::new(move |_request| {
            let streaming_calls = Arc::clone(&streaming_calls);
            Box::pin(async move {
                streaming_calls.fetch_add(1, Ordering::SeqCst);
                Ok(terminal_chat_stream())
            })
        })
    };
    let mut stream = run_cache_stream(
        "openai".to_string(),
        chat_request("streaming cache failure"),
        next,
        store.clone(),
        cache_config(),
    )
    .await
    .expect("a cache read error must not fail a streaming call");
    assert_eq!(
        stream
            .next()
            .await
            .expect("stream must yield a live chunk")
            .expect("live chunk must succeed")["choices"][0]["delta"]["content"],
        json!("cached")
    );
    stream
        .close()
        .await
        .expect("live stream cleanup must succeed");
    assert_eq!(streaming_calls.load(Ordering::SeqCst), 1);
    assert_eq!(store.get_calls.load(Ordering::SeqCst), 2);
    assert_eq!(store.set_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn streaming_cache_bypasses_stateful_and_sampled_calls_before_reading() {
    let store = Arc::new(FailingGetStore::default());
    let calls = Arc::new(AtomicUsize::new(0));
    let next: LlmStreamExecutionNextFn = {
        let calls = Arc::clone(&calls);
        Arc::new(move |_request| {
            let calls = Arc::clone(&calls);
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(terminal_chat_stream())
            })
        })
    };

    let mut stateful_request = chat_request("stateful cache bypass");
    stateful_request
        .content
        .as_object_mut()
        .expect("chat request is an object")
        .insert("store".to_string(), json!(true));
    let mut stateful = run_cache_stream(
        "openai".to_string(),
        stateful_request,
        Arc::clone(&next),
        store.clone(),
        cache_config(),
    )
    .await
    .expect("stateful request must run live");
    assert!(stateful.next().await.is_some());
    stateful.close().await.expect("stateful stream cleanup");

    let sampled_config = Arc::new(ResponseCacheConfig {
        bypass_rate: 1.0,
        ..(*cache_config()).clone()
    });
    let mut sampled = run_cache_stream(
        "openai".to_string(),
        chat_request("sampled cache bypass"),
        next,
        store.clone(),
        sampled_config,
    )
    .await
    .expect("sampled request must run live");
    while sampled.next().await.is_some() {}
    sampled.close().await.expect("sampled stream cleanup");

    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(store.get_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn write_behind_returns_eof_before_cache_commit_completes() {
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    let (cancel, _) = watch::channel(false);
    let (_, closed) = watch::channel(None::<FlowResult<()>>);
    let (release, wait) = oneshot::channel();
    let (commit_done, committed) = oneshot::channel();
    let commit: CacheCommit = Box::pin(async move {
        let _ = wait.await;
        let _ = commit_done.send(());
    });
    assert!(tx.send(TeeMessage::Commit(commit)).await.is_ok());

    let mut stream = ResponseCacheReceiver {
        receiver: ReceiverStream::new(rx),
        cancel,
        closed,
        finished: false,
    };
    assert!(
        tokio::time::timeout(Duration::from_secs(1), stream.next())
            .await
            .expect("write-behind cache publication must not delay stream completion")
            .is_none()
    );
    release
        .send(())
        .expect("detached cache commit must still be waiting");
    tokio::time::timeout(Duration::from_secs(1), committed)
        .await
        .expect("detached cache commit must resume after release")
        .expect("detached cache commit must run to completion");
}
