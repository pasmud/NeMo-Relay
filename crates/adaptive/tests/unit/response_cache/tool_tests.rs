// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Additional behavior tests for the tool-result response cache.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use nemo_relay::api::runtime::ToolExecutionNextFn;
use serde_json::Value as Json;

use super::*;
use crate::config::ResponseCacheConfig;
use crate::response_cache::config::{ToolCacheConfig, ToolClass};
use crate::response_cache::store::InMemoryCacheStore;

fn cache_config() -> Arc<ResponseCacheConfig> {
    Arc::new(ResponseCacheConfig {
        namespace: "tool-cache-unit-tests".to_string(),
        ttl_seconds: 60,
        ..ResponseCacheConfig::default()
    })
}

fn counting_next(calls: Arc<AtomicUsize>, result: Json) -> ToolExecutionNextFn {
    Arc::new(move |_args| {
        let calls = Arc::clone(&calls);
        let result = result.clone();
        Box::pin(async move {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(result)
        })
    })
}

#[test]
fn conventional_tool_error_detection_is_deliberately_narrow() {
    assert!(is_error_shaped_tool_result(&serde_json::json!({
        "error": "upstream unavailable"
    })));
    assert!(is_error_shaped_tool_result(&serde_json::json!({
        "isError": true
    })));
    assert!(!is_error_shaped_tool_result(&serde_json::json!({
        "error": null
    })));
    assert!(!is_error_shaped_tool_result(&serde_json::json!({
        "status": "failed"
    })));
}

#[tokio::test]
async fn sampled_error_result_preserves_the_prior_successful_entry_by_default() {
    let store = Arc::new(InMemoryCacheStore::new(1 << 20));
    let response_cache = cache_config();
    let regular_tools = Arc::new(ToolCacheConfig {
        enabled: true,
        default: ToolClass {
            cacheable: true,
            ..ToolClass::default()
        },
        ..ToolCacheConfig::default()
    });
    let sampled_tools = Arc::new(ToolCacheConfig {
        enabled: true,
        default: ToolClass {
            cacheable: true,
            bypass_rate: Some(1.0),
            ..ToolClass::default()
        },
        ..ToolCacheConfig::default()
    });

    let calls = Arc::new(AtomicUsize::new(0));
    let args = serde_json::json!({"query": "relay"});
    let first = run_tool_cache(
        "docs_lookup".to_string(),
        args.clone(),
        counting_next(Arc::clone(&calls), serde_json::json!({"answer": "cached"})),
        store.clone(),
        Arc::clone(&response_cache),
        Arc::clone(&regular_tools),
    )
    .await
    .unwrap();
    assert_eq!(first.result, serde_json::json!({"answer": "cached"}));

    let refresh = run_tool_cache(
        "docs_lookup".to_string(),
        args.clone(),
        counting_next(
            Arc::clone(&calls),
            serde_json::json!({"error": "temporary upstream outage"}),
        ),
        store.clone(),
        Arc::clone(&response_cache),
        sampled_tools,
    )
    .await
    .unwrap();
    assert_eq!(
        refresh.result,
        serde_json::json!({"error": "temporary upstream outage"})
    );

    let hit = run_tool_cache(
        "docs_lookup".to_string(),
        args,
        counting_next(
            Arc::clone(&calls),
            serde_json::json!({"answer": "unexpected"}),
        ),
        store,
        response_cache,
        regular_tools,
    )
    .await
    .unwrap();
    assert_eq!(hit.result, serde_json::json!({"answer": "cached"}));
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "the final call must use the original entry rather than re-running live"
    );
}
