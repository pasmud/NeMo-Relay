// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error-classification tests for the tool-result response cache.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use nemo_relay::api::runtime::ToolExecutionNextFn;

use super::*;
use crate::config::ResponseCacheConfig;
use crate::response_cache::config::{ToolCacheConfig, ToolClass};
use crate::response_cache::store::{CacheEntry, CacheStore, InMemoryCacheStore};

#[test]
fn conventional_tool_error_detection_is_deliberately_narrow() {
    assert!(is_error_shaped_tool_result(&serde_json::json!({
        "error": "upstream unavailable"
    })));
    assert!(is_error_shaped_tool_result(&serde_json::json!({
        "isError": true
    })));
    assert!(is_error_shaped_tool_result(&serde_json::json!({
        "is_error": true
    })));
    assert!(!is_error_shaped_tool_result(&serde_json::json!({
        "error": null
    })));
    assert!(!is_error_shaped_tool_result(&serde_json::json!({
        "status": "failed"
    })));
}

#[tokio::test]
async fn stale_error_entries_are_not_replayed_when_error_caching_is_disabled() {
    let store: Arc<dyn CacheStore> = Arc::new(InMemoryCacheStore::new(1 << 20));
    let response_cache = Arc::new(ResponseCacheConfig {
        namespace: "tool-cache-stale-error-test".to_string(),
        ..ResponseCacheConfig::default()
    });
    let tools = Arc::new(ToolCacheConfig {
        enabled: true,
        default: ToolClass {
            cacheable: true,
            ..ToolClass::default()
        },
        ..ToolCacheConfig::default()
    });
    let args = serde_json::json!({"query": "relay"});
    let key = match build_tool_cache_key(
        &response_cache.namespace,
        "docs_lookup",
        None,
        &args,
        &[],
        false,
    ) {
        KeyOutcome::Key(key) => key,
        other => panic!("expected a cache key, got {other:?}"),
    };
    let ttl = Duration::from_secs(60);
    store
        .set(
            &key,
            CacheEntry::new(
                serde_json::json!({"is_error": true, "content": "stale"}),
                ttl,
                key.clone(),
                None,
                None,
            ),
            ttl,
        )
        .await
        .unwrap();

    let calls = Arc::new(AtomicUsize::new(0));
    let next: ToolExecutionNextFn = Arc::new({
        let calls = Arc::clone(&calls);
        move |_args| {
            let calls = Arc::clone(&calls);
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(serde_json::json!({"answer": "fresh"}))
            })
        }
    });
    let result = run_tool_cache(
        "docs_lookup".to_string(),
        args.clone(),
        Arc::clone(&next),
        Arc::clone(&store),
        Arc::clone(&response_cache),
        Arc::clone(&tools),
    )
    .await
    .unwrap();

    assert_eq!(result.result, serde_json::json!({"answer": "fresh"}));
    let hit = run_tool_cache(
        "docs_lookup".to_string(),
        args,
        next,
        store,
        response_cache,
        tools,
    )
    .await
    .unwrap();
    assert_eq!(hit.result, serde_json::json!({"answer": "fresh"}));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
