//! One error type, two envelopes.
//!
//! The OpenAI and Anthropic APIs disagree about where the error object sits
//! and what its `type` field is called, and clients parse those envelopes
//! strictly. Rather than write the failure paths twice, every handler builds
//! the same [`ApiError`] and tags it with the [`Dialect`] its caller speaks.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

/// Which request dialect the caller is speaking, and therefore which error
/// envelope it can parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Dialect {
    OpenAi,
    Anthropic,
}

#[derive(Debug)]
pub(crate) struct ApiError {
    status: StatusCode,
    message: String,
    dialect: Dialect,
}

impl ApiError {
    pub(crate) fn new(status: StatusCode, dialect: Dialect, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            dialect,
        }
    }

    pub(crate) fn bad_request(dialect: Dialect, message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, dialect, message)
    }

    pub(crate) fn internal(dialect: Dialect, message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, dialect, message)
    }

    pub(crate) fn unavailable(dialect: Dialect, message: impl Into<String>) -> Self {
        Self::new(StatusCode::SERVICE_UNAVAILABLE, dialect, message)
    }

    /// What the tests assert against; responses carry it via `payload`.
    #[cfg(test)]
    pub(crate) fn message(&self) -> &str {
        &self.message
    }

    /// The error type string this dialect uses for this status.
    fn kind(&self) -> &'static str {
        match (self.dialect, self.status) {
            (Dialect::Anthropic, StatusCode::UNAUTHORIZED) => "authentication_error",
            (Dialect::Anthropic, StatusCode::SERVICE_UNAVAILABLE) => "overloaded_error",
            (Dialect::Anthropic, status) if status.is_client_error() => "invalid_request_error",
            (Dialect::Anthropic, _) => "api_error",
            (Dialect::OpenAi, status) if status.is_server_error() => "server_error",
            (Dialect::OpenAi, _) => "invalid_request_error",
        }
    }

    /// The JSON body, also used to report a failure that only becomes visible
    /// after the response status has already been sent as part of an event
    /// stream.
    pub(crate) fn payload(&self) -> Value {
        match self.dialect {
            Dialect::OpenAi => json!({
                "error": {
                    "message": self.message,
                    "type": self.kind(),
                    "param": Value::Null,
                    "code": if self.status == StatusCode::UNAUTHORIZED {
                        json!("invalid_api_key")
                    } else {
                        Value::Null
                    },
                }
            }),
            Dialect::Anthropic => json!({
                "type": "error",
                "error": { "type": self.kind(), "message": self.message },
            }),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, axum::Json(self.payload())).into_response()
    }
}

/// Parse a request body into `T`, reporting a parse failure in the caller's
/// own envelope.
///
/// Handlers take the raw bytes rather than `Json<T>` because axum's extractor
/// rejection is plain text, which no client in either dialect knows how to
/// read.
pub(crate) fn parse_body<T: serde::de::DeserializeOwned>(
    body: &[u8],
    dialect: Dialect,
) -> Result<T, ApiError> {
    serde_json::from_slice(body).map_err(|error| {
        ApiError::bad_request(dialect, format!("could not parse request: {error}"))
    })
}
