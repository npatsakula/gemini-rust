use crate::{FinishReason, FunctionCall, GenerationResponse, Model, Part};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[test]
fn test_model_deserialization() {
    #[derive(Serialize, Deserialize)]
    struct Response {
        model: Model,
    }

    let response = Response {
        model: Model::Custom("models/custom_gemini_model".to_string()),
    };
    let serialized = serde_json::to_string(&response).unwrap();
    let deserialized: Response = serde_json::from_str(&serialized).unwrap();
    assert_eq!(deserialized.model, response.model);

    let response = Response {
        model: Model::Gemini25Flash,
    };
    let serialized = serde_json::to_string(&response).unwrap();
    let deserialized: Response = serde_json::from_str(&serialized).unwrap();
    assert_eq!(deserialized.model, response.model);
}

#[test]
fn test_thought_signature_deserialization() {
    // Test JSON that includes thoughtSignature like in the provided API response
    let json_response = json!({
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_current_weather",
                                "args": {
                                    "location": "Kaohsiung Zuoying District"
                                }
                            },
                            "thoughtSignature": "CtwFAVSoXO4WSz0Ri3HddDzPQzsB8EaYsiQobiBKOzGOaAPM0d4DewrzUmhCnZbdboz+n+6v503fcy4epZC2bomn247laY6RHtKTc0UA8scj1DW/Y8w9AsfvjDX1adpIi043qjivTtowjxKAIesKoO69mFj6HTmGRI6sE1hamsIblZGZypowxnBQmxqJftl1aebB7kQN+MoYSeX+OU1z/8G+RXE+cb9cvwdAGIZjHXoGgEaIigYlrjTkZjRGBiI+gC2AcLNe32MHVla2/dmV8O7k8Cl45ksH+4srYABtmXLxjxwQK6s2bjVngvaRcBTCK4AUHiDb1j54n3Fls5J1i9k2sd6OcJYJuRlfwuxv2RMZ+V8zLdNthfSWtZwuJslkOD3uZCkEhO/hI6nAKcyuSokdAKtOw9g6LWORnEQoUJ+BaTVymN1tuJzbzrS9kPP5d3QJfFdQaILkk8CUdnGOEcngvlINN4MGNTQYN+0Au/JFWDWj33T5LZWkbDMp+yIpqFkZuRYwjW/9KOR6qFbxzvJyQcAKTxf0Sq7UfHTYBXTVp0/N4cDWRv+5DF0UOp+6emnPslCmaRK8JEGkmKkYXCzR6PpopfdzHHSDQHbNjjwr0h9ADZKehiB/cB1Jjy0oyBOM3HSHyuzcP8CO4NoAXOUM/VP5P41ys9TdeaPZAZ1E3cGQI4pifFVPdy3o33QSYqS1ce5Wxbeud06+d+sz2O7jJrfHMdgYpcO/2RcXQyK/GVIlDkWyxpYtBZhlkh3vLxPVmV/JJv5DQSS3YNTFSbfbwC8DtrI6YNFK5Vo07cl6mAY+U8b4ziFJk2HGuO27jq5EnhJE6v39HCfXTa8cKaLzpIURJSOs12S1rc3pqXdv4VBL6dp+Yjr8eQPxYRP93QzZMFXcYZ+Vc2H5mbnXbvTxVdYT7Qpu7aK1o6csSOMOx47NzZnOnlTWNJUxtU5UIZJ2JelOt/NsWnVJZY8D"
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 70,
            "candidatesTokenCount": 21,
            "totalTokenCount": 255,
            "thoughtsTokenCount": 164
        },
        "modelVersion": "gemini-2.5-pro",
        "responseId": "CCm8aJjzBaWh1MkP_cLEgQo"
    });

    // Test deserialization
    let response: GenerationResponse = serde_json::from_value(json_response).unwrap();

    // Verify basic structure
    assert_eq!(response.candidates.len(), 1);
    let candidate = &response.candidates[0];
    assert_eq!(candidate.finish_reason, Some(FinishReason::Stop));

    // Check content parts
    let parts = candidate.content.parts.as_ref().unwrap();
    assert_eq!(parts.len(), 1);

    // Verify the part is a function call with thought signature
    match &parts[0] {
        Part::FunctionCall {
            function_call,
            thought_signature,
        } => {
            assert_eq!(function_call.name, "get_current_weather");
            assert_eq!(function_call.args["location"], "Kaohsiung Zuoying District");

            // Verify thought signature is present and not empty
            assert!(thought_signature.is_some());
            let signature = thought_signature.as_ref().unwrap();
            assert!(!signature.is_empty());
            assert!(signature.starts_with("CtwFAVSoXO4WSz0Ri3HddDzPQzsB8EaYsiQobiBKOzGOaAPM"));
        }
        _ => panic!("Expected FunctionCall part"),
    }

    // Test the function_calls_with_thoughts method
    let function_calls_with_thoughts = response.function_calls_with_thoughts();
    assert_eq!(function_calls_with_thoughts.len(), 1);

    let (function_call, thought_signature) = &function_calls_with_thoughts[0];
    assert_eq!(function_call.name, "get_current_weather");
    assert!(thought_signature.is_some());

    // Test usage metadata with thinking tokens
    assert!(response.usage_metadata.is_some());
    let usage = response.usage_metadata.as_ref().unwrap();
    assert_eq!(usage.thoughts_token_count, Some(164));
}

#[test]
fn test_function_call_with_thought_signature() {
    // Test creating a FunctionCall with thought signature
    let function_call = FunctionCall::with_thought_signature(
        "test_function",
        json!({"param": "value"}),
        "test_thought_signature",
    );

    assert_eq!(function_call.name, "test_function");
    assert_eq!(function_call.args["param"], "value");
    assert_eq!(
        function_call.thought_signature,
        Some("test_thought_signature".to_string())
    );

    // Test serialization
    let serialized = serde_json::to_string(&function_call).unwrap();
    println!("Serialized FunctionCall: {serialized}");
    let json: serde_json::Value = serde_json::from_str(&serialized).unwrap();
    assert_eq!(json["thought_signature"], "test_thought_signature");
    assert!(json.get("thoughtSignature").is_none());

    // Test deserialization
    let deserialized: FunctionCall = serde_json::from_str(&serialized).unwrap();
    assert_eq!(deserialized, function_call);
}

#[test]
fn test_function_call_without_thought_signature() {
    // Test creating a FunctionCall without thought signature (backward compatibility)
    let function_call = FunctionCall::new("test_function", json!({"param": "value"}));

    assert_eq!(function_call.name, "test_function");
    assert_eq!(function_call.args["param"], "value");
    assert_eq!(function_call.thought_signature, None);

    // Test serialization should not include thought_signature field when None
    let serialized = serde_json::to_string(&function_call).unwrap();
    println!("Serialized FunctionCall without thought: {serialized}");
    assert!(!serialized.contains("thought_signature"));
}

#[test]
fn test_multi_turn_content_structure() {
    // Test that we can create proper multi-turn content structure for maintaining thought context
    use crate::{Content, Part, Role};

    // Simulate a function call with thought signature from first turn
    let function_call = FunctionCall::with_thought_signature(
        "get_weather",
        json!({"location": "Tokyo"}),
        "sample_thought_signature",
    );

    // Create model content with function call and thought signature
    let model_content = Content {
        parts: Some(vec![Part::FunctionCall {
            function_call: function_call.clone(),
            thought_signature: Some("sample_thought_signature".to_string()),
        }]),
        role: Some(Role::Model),
    };

    // Verify structure
    assert!(model_content.parts.is_some());
    assert_eq!(model_content.role, Some(Role::Model));

    // Test serialization of the complete structure first
    let serialized = serde_json::to_string(&model_content).unwrap();
    println!("Serialized multi-turn content: {serialized}");

    // Verify it contains the thought signature
    assert!(serialized.contains("thoughtSignature"));
    assert!(serialized.contains("sample_thought_signature"));

    let parts = model_content.parts.unwrap();
    assert_eq!(parts.len(), 1);

    match &parts[0] {
        Part::FunctionCall {
            function_call,
            thought_signature,
        } => {
            assert_eq!(function_call.name, "get_weather");
            assert_eq!(
                thought_signature.as_ref().unwrap(),
                "sample_thought_signature"
            );
        }
        _ => panic!("Expected FunctionCall part"),
    }

    let content_from_helper = Content::function_call(function_call);
    let helper_json = serde_json::to_value(&content_from_helper).unwrap();
    assert_eq!(
        helper_json["parts"][0]["thoughtSignature"],
        "sample_thought_signature"
    );
    assert!(helper_json["parts"][0]["functionCall"]
        .get("thoughtSignature")
        .is_none());
    assert!(helper_json["parts"][0]["functionCall"]
        .get("thought_signature")
        .is_none());
}

#[test]
fn test_text_with_thought_signature() {
    use crate::GenerationResponse;

    // Test JSON similar to the provided API response
    let json_response = json!({
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "**Okay, here's what I'm thinking:**\n\nThe user wants me to show them...",
                            "thought": true
                        },
                        {
                            "text": "The following functions are available in the environment: `chat.get_message_count()`",
                            "thoughtSignature": "Cs4BA.../Yw="
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 36,
            "candidatesTokenCount": 18,
            "totalTokenCount": 96,
            "thoughtsTokenCount": 42
        },
        "modelVersion": "gemini-2.5-flash",
        "responseId": "gIC..."
    });

    // Test deserialization
    let response: GenerationResponse = serde_json::from_value(json_response).unwrap();

    // Verify basic structure
    assert_eq!(response.candidates.len(), 1);
    let candidate = &response.candidates[0];

    // Check content parts
    let parts = candidate.content.parts.as_ref().unwrap();
    assert_eq!(parts.len(), 2);

    // Check first part (thought without signature)
    match &parts[0] {
        Part::Text {
            text,
            thought,
            thought_signature,
        } => {
            assert_eq!(*thought, Some(true));
            assert_eq!(*thought_signature, None);
            assert!(text.contains("here's what I'm thinking"));
        }
        _ => panic!("Expected Text part for first element"),
    }

    // Check second part (text with thought signature)
    match &parts[1] {
        Part::Text {
            text,
            thought,
            thought_signature,
        } => {
            assert_eq!(*thought, None);
            assert!(thought_signature.is_some());
            assert_eq!(thought_signature.as_ref().unwrap(), "Cs4BA.../Yw=");
            assert!(text.contains("chat.get_message_count"));
        }
        _ => panic!("Expected Text part for second element"),
    }

    // Test the new text_with_thoughts method
    let text_with_thoughts = response.text_with_thoughts();
    assert_eq!(text_with_thoughts.len(), 2);

    let (first_text, is_thought, thought_sig) = &text_with_thoughts[0];
    assert!(*is_thought);
    assert!(thought_sig.is_none());
    assert!(first_text.contains("here's what I'm thinking"));

    let (second_text, is_thought, thought_sig) = &text_with_thoughts[1];
    assert!(!(*is_thought));
    assert!(thought_sig.is_some());
    assert_eq!(thought_sig.unwrap(), "Cs4BA.../Yw=");
    assert!(second_text.contains("chat.get_message_count"));
}

#[test]
fn test_content_creation_with_thought_signature() {
    // Test creating content with thought signature
    use crate::Content;
    let content = Content::text_with_thought_signature("Test response", "test_signature_123");

    let parts = content.parts.as_ref().unwrap();
    assert_eq!(parts.len(), 1);

    match &parts[0] {
        Part::Text {
            text,
            thought,
            thought_signature,
        } => {
            assert_eq!(text, "Test response");
            assert_eq!(*thought, None);
            assert_eq!(thought_signature.as_ref().unwrap(), "test_signature_123");
        }
        _ => panic!("Expected Text part"),
    }

    // Test creating thought content with signature
    let thought_content =
        Content::thought_with_signature("This is my thinking process", "thought_signature_456");

    let parts = thought_content.parts.as_ref().unwrap();
    assert_eq!(parts.len(), 1);

    match &parts[0] {
        Part::Text {
            text,
            thought,
            thought_signature,
        } => {
            assert_eq!(text, "This is my thinking process");
            assert_eq!(*thought, Some(true));
            assert_eq!(thought_signature.as_ref().unwrap(), "thought_signature_456");
        }
        _ => panic!("Expected Text part"),
    }

    // Test serialization
    let serialized = serde_json::to_string(&content).unwrap();
    println!("Serialized content with thought signature: {serialized}");
    assert!(serialized.contains("thoughtSignature"));
    assert!(serialized.contains("test_signature_123"));

    // Test serialization of thought content
    let serialized_thought = serde_json::to_string(&thought_content).unwrap();
    println!("Serialized thought content: {serialized_thought}");
    assert!(serialized_thought.contains("thoughtSignature"));
    assert!(serialized_thought.contains("thought_signature_456"));
    assert!(serialized_thought.contains("\"thought\":true"));
}

#[test]
fn test_builder_safety_settings() {
    use crate::{GeminiBuilder, HarmBlockThreshold, HarmCategory, SafetySetting};

    let client = GeminiBuilder::new("_key").build().unwrap();

    let settings = vec![SafetySetting {
        category: HarmCategory::Harassment,
        threshold: HarmBlockThreshold::BlockNone,
    }];

    let builder = client
        .generate_content()
        .with_safety_settings(settings.clone());

    let request = builder.build();

    assert!(request.safety_settings.is_some());
    let req_settings = request.safety_settings.unwrap();
    assert_eq!(req_settings.len(), 1);
    assert_eq!(req_settings[0].category, HarmCategory::Harassment);
}

// ========== Batch response decoding ==========
// The Gemini API follows proto3 JSON conventions and omits empty/output-only
// fields, and it has two layouts for completed results. These tests lock in
// that our batch types tolerate both.

#[test]
fn test_list_batches_response_defaults_operations_when_absent() {
    use crate::batch::model::ListBatchesResponse;

    // An empty list body (`{}`) must decode to an empty operations list, since
    // the API omits the array entirely rather than returning `[]`.
    let resp: ListBatchesResponse = serde_json::from_str("{}").unwrap();
    assert!(resp.operations.is_empty());
    assert!(resp.next_page_token.is_none());
}

#[test]
fn test_batch_metadata_decodes_with_only_state() {
    use crate::batch::model::{BatchMetadata, BatchState};

    // In-progress batch metadata may omit every output-only field except state.
    let meta: BatchMetadata = serde_json::from_str(r#"{"state":"BATCH_STATE_PENDING"}"#).unwrap();
    assert_eq!(meta.state, BatchState::BatchStatePending);
    assert!(meta.create_time.is_none());
    assert_eq!(meta.batch_stats.request_count, 0);
}

#[test]
fn test_batch_operation_response_decodes_legacy_responses_file() {
    use crate::batch::model::BatchOperationResponse;

    // Legacy shape: the file reference sits at the top level.
    let r: BatchOperationResponse =
        serde_json::from_str(r#"{"responsesFile":"files/abc"}"#).unwrap();
    match r {
        BatchOperationResponse::ResponsesFile { responses_file } => {
            assert_eq!(responses_file, "files/abc");
        }
        other => panic!("expected legacy ResponsesFile, got {other:?}"),
    }
}

#[test]
fn test_batch_operation_response_decodes_nested_output() {
    use crate::batch::model::BatchOperationResponse;

    // Newer typed shape: the same data nested under `output`.
    let r: BatchOperationResponse =
        serde_json::from_str(r#"{"output":{"responsesFile":"files/abc"}}"#).unwrap();
    match r {
        BatchOperationResponse::Output { output } => {
            assert_eq!(output.responses_file.as_deref(), Some("files/abc"));
            assert!(output.inlined_responses.is_none());
        }
        other => panic!("expected nested Output, got {other:?}"),
    }
}

#[test]
fn test_inlined_responses_defaults_when_absent() {
    use crate::batch::model::InlinedResponses;

    // Inlined response containers omit the array when there are no responses.
    let r: InlinedResponses = serde_json::from_str("{}").unwrap();
    assert!(r.inlined_responses.is_empty());
}

// ========== Batch lifecycle classification ==========
// `state` is the authoritative lifecycle status; every terminal state must be
// reachable. These exercise the pure classifier directly (no I/O).

fn classify_operation(json: &str) -> crate::batch::handle::BatchOutcome {
    use crate::batch::{handle::BatchStatus, model::BatchOperation};
    let operation: BatchOperation = serde_json::from_str(json).unwrap();
    BatchStatus::classify(&operation)
}

#[test]
fn test_classify_cancelled_state_maps_to_cancelled() {
    use crate::batch::handle::BatchOutcome;

    // A cancelled batch completes with a CANCELLED error payload; `state` must
    // still win, so this resolves to Cancelled rather than a generic failure.
    let outcome = classify_operation(
        r#"{
            "name": "batches/abc",
            "done": true,
            "error": {"code": 1, "message": "Cancelled"},
            "metadata": {"state": "BATCH_STATE_CANCELLED", "batchStats": {"requestCount": "2"}}
        }"#,
    );
    assert!(matches!(outcome, BatchOutcome::Cancelled));
}

#[test]
fn test_classify_expired_state_maps_to_expired() {
    use crate::batch::handle::BatchOutcome;

    let outcome = classify_operation(
        r#"{
            "name": "batches/abc",
            "done": true,
            "metadata": {"state": "BATCH_STATE_EXPIRED", "batchStats": {"requestCount": "2"}}
        }"#,
    );
    assert!(matches!(outcome, BatchOutcome::Expired));
}

#[test]
fn test_classify_failed_state_maps_to_failed() {
    use crate::batch::handle::BatchOutcome;

    let outcome = classify_operation(
        r#"{
            "name": "batches/abc",
            "done": true,
            "error": {"code": 3, "message": "boom"},
            "metadata": {"state": "BATCH_STATE_FAILED", "batchStats": {"requestCount": "2"}}
        }"#,
    );
    assert!(matches!(outcome, BatchOutcome::Failed(_)));
}

#[test]
fn test_classify_pending_state_maps_to_pending() {
    use crate::batch::handle::BatchOutcome;

    let outcome = classify_operation(
        r#"{"name": "batches/abc", "done": false, "metadata": {"state": "BATCH_STATE_PENDING"}}"#,
    );
    assert!(matches!(outcome, BatchOutcome::Pending));
}

#[test]
fn test_classify_running_state_reports_counts() {
    use crate::batch::handle::BatchOutcome;
    use crate::BatchStatus;

    let outcome = classify_operation(
        r#"{
            "name": "batches/abc",
            "done": false,
            "metadata": {
                "state": "BATCH_STATE_RUNNING",
                "batchStats": {
                    "requestCount": "5",
                    "pendingRequestCount": "3",
                    "successfulRequestCount": "1",
                    "failedRequestCount": "1"
                }
            }
        }"#,
    );
    match outcome {
        BatchOutcome::Running(BatchStatus::Running {
            pending_count,
            completed_count,
            failed_count,
            total_count,
        }) => {
            assert_eq!(total_count, 5);
            assert_eq!(pending_count, 3);
            assert_eq!(completed_count, 1);
            assert_eq!(failed_count, 1);
        }
        _ => panic!("expected Running outcome"),
    }
}

#[test]
fn test_classify_succeeded_state_signals_success() {
    use crate::batch::handle::BatchOutcome;

    // Result extraction is deferred (it may require I/O), so success only needs
    // the state to be recognized here.
    let outcome = classify_operation(
        r#"{
            "name": "batches/abc",
            "done": true,
            "metadata": {"state": "BATCH_STATE_SUCCEEDED", "batchStats": {"requestCount": "2"}}
        }"#,
    );
    assert!(matches!(outcome, BatchOutcome::Succeeded));
}

#[test]
fn test_classify_done_with_error_falls_back_to_failed() {
    use crate::batch::handle::BatchOutcome;

    // State not yet terminal, but the LRO is done with an error: must not be
    // reported as Pending forever.
    let outcome = classify_operation(
        r#"{
            "name": "batches/abc",
            "done": true,
            "error": {"code": 13, "message": "internal"},
            "metadata": {"state": "BATCH_STATE_UNSPECIFIED"}
        }"#,
    );
    assert!(matches!(outcome, BatchOutcome::Failed(_)));
}
