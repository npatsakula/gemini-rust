//! The Batch module for managing batch operations.
//!
//! This module provides the [`BatchHandle`] struct, which is a handle to a long-running batch
//! operation on the Gemini API. It allows for checking the status, canceling, and deleting
//! the operation.
//!
//! The status of a batch operation is represented by the [`BatchStatus`] enum, which can be
//! retrieved using the [`BatchHandle::status()`] method. When a batch completes successfully,
//! it transitions to the [`BatchStatus::Succeeded`] state, which contains a vector of
//! [`BatchGenerationResponseItem`].
//!
//! ## Batch Results
//!
//! The [`BatchGenerationResponseItem`] enum represents the outcome of a single request within the batch:
//! - `Success`: Contains the generated `GenerationResponse` and the original request key.
//! - `Error`: Contains an `IndividualRequestError` and the original request key.
//!
//! Results can be delivered in two ways, depending on the size of the batch job:
//! 1.  **Inlined Responses**: For smaller jobs, the results are included directly in the
//!     batch operation's metadata.
//! 2.  **Response File**: For larger jobs (typically >20MB), the results are written to a
//!     file, and the batch metadata will contain a reference to this file. The SDK
//!     handles the downloading and parsing of this file automatically when you call
//!     `status()` on a completed batch.
//!
//! The results are automatically sorted by their original request key (as a number) to ensure
//! a consistent and predictable order.
//!
//! For more information, see the official Google AI documentation:
//! - [Batch Mode Guide](https://ai.google.dev/gemini-api/docs/batch-mode)
//! - [Batch API Reference](https://ai.google.dev/api/batch-mode)
//!
//! # Design Note: Resource Management in Batch Operations
//!
//! The Batch API methods that consume the [`BatchHandle`] struct (`cancel`, `delete`)
//! return `std::result::Result<T, (Self, crate::Error)>` instead of the crate's `Result<T>`.
//! This design follows patterns used in channel libraries (e.g., `std::sync::mpsc::Receiver`)
//! and provides two key benefits:
//!
//! 1. **Resource Safety**: Once a [`BatchHandle`] is consumed by an operation, it cannot be used again,
//!    preventing invalid operations on deleted or canceled batches.
//!
//! 2. **Error Recovery**: If an operation fails due to transient network issues, both the
//!    [`BatchHandle`] and error information are returned, allowing callers to retry the operation.
//!
//! ## Example usage:
//! ```rust,no_run
//! use gemini_rust::{Gemini, Message};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = Gemini::new(std::env::var("GEMINI_API_KEY")?)?;
//!     let request = client.generate_content().with_user_message("Why is the sky blue?").build();
//!     let batch = client.batch_generate_content().with_request(request).execute().await?;
//!
//!     match batch.delete().await {
//!         Ok(()) => println!("Batch deleted successfully!"),
//!         Err((batch, error)) => {
//!             println!("Failed to delete batch: {}", error);
//!             // Can retry: batch.delete().await
//!         }
//!     }
//!     Ok(())
//! }
//! ```

use snafu::{IntoError, OptionExt, ResultExt, Snafu};
use std::{result::Result, sync::Arc};

use super::model::*;
use crate::{
    client::{Error as ClientError, GeminiClient},
    files::handle::FileHandle,
    GenerationResponse,
};

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("batch '{name}' expired before finishing"))]
    BatchExpired {
        /// Batch name.
        name: String,
    },

    #[snafu(display("batch '{name}' failed"))]
    BatchFailed {
        source: OperationError,
        /// Batch name.
        name: String,
    },

    #[snafu(display("client invocation error"))]
    Client { source: Box<ClientError> },

    #[snafu(display("failed to download batch result file '{file_name}'"))]
    FileDownload {
        source: crate::files::Error,
        file_name: String,
    },

    #[snafu(display("failed to decode batch result file content as UTF-8"))]
    FileDecode { source: std::string::FromUtf8Error },

    #[snafu(display("failed to parse line in batch result file"))]
    FileParse {
        source: serde_json::Error,
        line: String,
    },

    /// This error should never occur, as the Google API contract
    /// guarantees that a result will always be provided.
    ///
    /// I put it here anyway to avoid potential panic in case of
    /// Google's dishonesty or GCP internal errors.
    #[snafu(display("batch '{name}' completed but no result provided - API contract violation"))]
    MissingResult {
        /// Batch name.
        name: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatchGenerationResponseItem {
    pub response: Result<GenerationResponse, IndividualRequestError>,
    pub meta: RequestMetadata,
}

/// Represents the overall status of a batch operation.
#[derive(Debug, Clone, PartialEq)]
pub enum BatchStatus {
    /// The operation is waiting to be processed.
    Pending,
    /// The operation is currently being processed.
    Running {
        pending_count: i64,
        completed_count: i64,
        failed_count: i64,
        total_count: i64,
    },
    /// The operation has completed successfully.
    Succeeded {
        results: Vec<BatchGenerationResponseItem>,
    },
    /// The operation was cancelled by the user.
    Cancelled,
    /// The operation has expired.
    Expired,
}

impl BatchStatus {
    /// Downloads and parses a result file identified by its resource name.
    async fn parse_response_file_by_name(
        responses_file: String,
        client: Arc<GeminiClient>,
    ) -> Result<Vec<BatchGenerationResponseItem>, Error> {
        let file = crate::files::model::File {
            name: responses_file,
            ..Default::default()
        };
        Self::parse_response_file(file, client).await
    }

    async fn parse_response_file(
        response_file: crate::files::model::File,
        client: Arc<GeminiClient>,
    ) -> Result<Vec<BatchGenerationResponseItem>, Error> {
        let file = FileHandle::new(client.clone(), response_file);
        let file_content_bytes = file.download().await.context(FileDownloadSnafu {
            file_name: file.name(),
        })?;
        let file_content = String::from_utf8(file_content_bytes).context(FileDecodeSnafu)?;

        let mut results = vec![];
        for line in file_content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let item: BatchResponseFileItem =
                serde_json::from_str(line).context(FileParseSnafu {
                    line: line.to_string(),
                })?;

            results.push(BatchGenerationResponseItem {
                response: item.response.into(),
                meta: RequestMetadata { key: item.key },
            });
        }
        Ok(results)
    }

    async fn process_successful_response(
        response: BatchOperationResponse,
        client: Arc<GeminiClient>,
    ) -> Result<Vec<BatchGenerationResponseItem>, Error> {
        let results = match response {
            BatchOperationResponse::InlinedResponses { inlined_responses } => inlined_responses
                .inlined_responses
                .into_iter()
                .map(|item| BatchGenerationResponseItem {
                    response: item.result.into(),
                    meta: item.metadata,
                })
                .collect(),
            BatchOperationResponse::ResponsesFile { responses_file } => {
                Self::parse_response_file_by_name(responses_file, client).await?
            }
            BatchOperationResponse::Output { output } => {
                if let Some(inlined) = output.inlined_responses {
                    inlined
                        .inlined_responses
                        .into_iter()
                        .map(|item| BatchGenerationResponseItem {
                            response: item.result.into(),
                            meta: item.metadata,
                        })
                        .collect()
                } else if let Some(responses_file) = output.responses_file {
                    Self::parse_response_file_by_name(responses_file, client).await?
                } else {
                    vec![]
                }
            }
        };
        Ok(results)
    }

    async fn from_operation(
        operation: BatchOperation,
        client: Arc<GeminiClient>,
    ) -> Result<Self, Error> {
        match Self::classify(&operation) {
            BatchOutcome::Pending => Ok(BatchStatus::Pending),
            BatchOutcome::Running(status) => Ok(status),
            BatchOutcome::Cancelled => Ok(BatchStatus::Cancelled),
            BatchOutcome::Expired => Ok(BatchStatus::Expired),
            BatchOutcome::Failed(error) => Err(error),
            // Only the success path needs to fetch/parse results (I/O).
            BatchOutcome::Succeeded => Self::succeeded(operation, client).await,
        }
    }

    /// Maps an operation to a lifecycle outcome using `state` as the
    /// authoritative status, as the API specifies. The LRO `done`/`result`
    /// fields are only consulted as a fallback when `state` is not yet
    /// meaningful, so every terminal status remains reachable. This is pure
    /// (no I/O) so the lifecycle transitions can be unit-tested directly.
    pub(crate) fn classify(operation: &BatchOperation) -> BatchOutcome {
        match operation.metadata.state {
            BatchState::BatchStateSucceeded => BatchOutcome::Succeeded,
            BatchState::BatchStateFailed => BatchOutcome::Failed(Self::failure(operation)),
            BatchState::BatchStateCancelled => BatchOutcome::Cancelled,
            BatchState::BatchStateExpired => BatchOutcome::Expired,
            BatchState::BatchStateRunning => BatchOutcome::Running(Self::running(operation)),
            // State isn't terminal. Honor LRO completion as a fallback so a
            // finished operation whose state is under-specified still resolves
            // rather than reporting Pending forever.
            BatchState::BatchStatePending | BatchState::BatchStateUnspecified => {
                match (operation.done, &operation.result) {
                    (true, Some(OperationResult::Error(_))) => {
                        BatchOutcome::Failed(Self::failure(operation))
                    }
                    (true, Some(OperationResult::Response(_))) => BatchOutcome::Succeeded,
                    _ => BatchOutcome::Pending,
                }
            }
        }
    }

    /// Extracts and sorts results from a completed, successful operation.
    async fn succeeded(
        operation: BatchOperation,
        client: Arc<GeminiClient>,
    ) -> Result<BatchStatus, Error> {
        let result = operation.result.context(MissingResultSnafu {
            name: operation.name.clone(),
        })?;
        let response = Result::from(result).context(BatchFailedSnafu {
            name: operation.name,
        })?;
        let mut results = Self::process_successful_response(response, client).await?;
        results.sort_by_key(|r| r.meta.key);
        Ok(BatchStatus::Succeeded { results })
    }

    /// Builds a `BatchFailed` error, preferring the operation's own error
    /// payload and synthesizing one when the API reports a failed state without
    /// any error details.
    fn failure(operation: &BatchOperation) -> Error {
        let source = match &operation.result {
            Some(OperationResult::Error(error)) => OperationError {
                code: error.code,
                message: error.message.clone(),
            },
            _ => OperationError {
                code: -1,
                message: "batch reported a FAILED state without error details".to_string(),
            },
        };
        BatchFailedSnafu {
            name: operation.name.clone(),
        }
        .into_error(source)
    }

    /// Builds the in-progress `Running` status from the batch statistics.
    fn running(operation: &BatchOperation) -> BatchStatus {
        let stats = &operation.metadata.batch_stats;
        let total_count = stats.request_count;
        BatchStatus::Running {
            pending_count: stats.pending_request_count.unwrap_or(total_count),
            completed_count: stats.successful_request_count.unwrap_or(0),
            failed_count: stats.failed_request_count.unwrap_or(0),
            total_count,
        }
    }
}

/// The lifecycle outcome of a batch operation, derived purely from its state.
///
/// `Succeeded` indicates results are available but still need to be fetched and
/// parsed by the caller; every other variant is fully resolved.
pub(crate) enum BatchOutcome {
    /// Waiting to be processed.
    Pending,
    /// Currently processing, carrying the populated [`BatchStatus::Running`].
    Running(BatchStatus),
    /// Completed successfully; results still need to be extracted.
    Succeeded,
    /// Cancelled by the user.
    Cancelled,
    /// Expired before completion.
    Expired,
    /// Failed, carrying the error to surface from `status()`.
    Failed(Error),
}

/// Represents a long-running batch operation, providing methods to manage its lifecycle.
///
/// A `Batch` object is a handle to a batch operation on the Gemini API. It allows you to
/// check the status, cancel the operation, or delete it once it's no longer needed.
pub struct BatchHandle {
    /// The unique resource name of the batch operation, of the form
    /// `batches/{id}` (as returned when the batch is created). This is the
    /// value to pass to [`Gemini::get_batch`](crate::Gemini::get_batch).
    pub name: String,
    client: Arc<GeminiClient>,
}

impl BatchHandle {
    /// Creates a new Batch instance.
    pub(crate) fn new(name: String, client: Arc<GeminiClient>) -> Self {
        Self { name, client }
    }

    /// Returns the unique resource name of the batch operation.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Retrieves the current status of the batch operation by making an API call.
    ///
    /// This method provides a snapshot of the batch's state at a single point in time.
    pub async fn status(&self) -> Result<BatchStatus, Error> {
        let operation: BatchOperation = self
            .client
            .get_batch_operation(&self.name)
            .await
            .map_err(Box::new)
            .context(ClientSnafu)?;

        BatchStatus::from_operation(operation, self.client.clone()).await
    }

    /// Sends a request to the API to cancel the batch operation.
    ///
    /// Cancellation is not guaranteed to be instantaneous. The operation may continue to run for
    /// some time after the cancellation request is made.
    ///
    /// Consumes the batch. If cancellation fails, returns the batch and error information
    /// so it can be retried.
    pub async fn cancel(self) -> Result<(), (Self, ClientError)> {
        match self.client.cancel_batch_operation(&self.name).await {
            Ok(()) => Ok(()),
            Err(e) => Err((self, e)),
        }
    }

    /// Deletes the batch operation resource from the server.
    ///
    /// Note: This method indicates that the client is no longer interested in the operation result.
    /// It does not cancel a running operation. To stop a running batch, use the `cancel` method.
    /// This method should typically be used after the batch has completed.
    ///
    /// Consumes the batch. If deletion fails, returns the batch and error information
    /// so it can be retried.
    pub async fn delete(self) -> Result<(), (Self, ClientError)> {
        match self.client.delete_batch_operation(&self.name).await {
            Ok(()) => Ok(()),
            Err(e) => Err((self, e)),
        }
    }
}
