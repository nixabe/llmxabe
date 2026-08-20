//! API-key authentication.
//!
//! Both header spellings are accepted on every endpoint — OpenAI clients send
//! `Authorization: Bearer <key>`, Anthropic clients send `x-api-key: <key>`,
//! and a server that speaks both dialects should not make the caller care
//! which one it guessed. With no key configured the server is open, which is
//! how it behaved before this existed and how `llama-server` behaves without
//! `--api-key`.

use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::http::header::AUTHORIZATION;
use axum::middleware::Next;
use axum::response::Response;

use super::AppState;
use super::error::{ApiError, Dialect};

/// Compare two keys without a length-dependent early exit.
///
/// The lengths themselves are not secret — a key that is the wrong length is
/// the wrong key — but the position of the first differing byte is.
fn keys_match(presented: &str, expected: &str) -> bool {
    let (presented, expected) = (presented.as_bytes(), expected.as_bytes());
    if presented.len() != expected.len() {
        return false;
    }
    presented
        .iter()
        .zip(expected)
        .fold(0u8, |acc, (a, b)| acc | (a ^ b))
        == 0
}

fn presented_key(request: &Request) -> Option<&str> {
    let headers = request.headers();
    if let Some(key) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        return Some(key.trim());
    }
    headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| {
            let (scheme, key) = value.split_once(' ')?;
            scheme.eq_ignore_ascii_case("bearer").then(|| key.trim())
        })
}

pub(crate) async fn require_api_key(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    let Some(expected) = state.api_key.as_deref() else {
        return Ok(next.run(request).await);
    };
    // The error envelope has to match the endpoint the caller aimed at, and
    // at this point the request has not been routed yet, so the path is all
    // there is to go on.
    let dialect = if request.uri().path().starts_with("/v1/messages") {
        Dialect::Anthropic
    } else {
        Dialect::OpenAi
    };
    match presented_key(&request) {
        Some(key) if keys_match(key, expected) => Ok(next.run(request).await),
        Some(_) => Err(ApiError::new(
            StatusCode::UNAUTHORIZED,
            dialect,
            "the supplied API key is not valid",
        )),
        None => Err(ApiError::new(
            StatusCode::UNAUTHORIZED,
            dialect,
            "this server requires an API key in `Authorization: Bearer <key>` or `x-api-key: <key>`",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_match_only_on_equal_bytes() {
        assert!(keys_match("sk-secret", "sk-secret"));
        assert!(!keys_match("sk-secret", "sk-secreT"));
        assert!(!keys_match("sk-secret", "sk-secret "));
        assert!(!keys_match("", "sk-secret"));
        assert!(keys_match("", ""));
    }

    fn request_with(header: &str, value: &str) -> Request {
        Request::builder()
            .uri("/v1/chat/completions")
            .header(header, value)
            .body(axum::body::Body::empty())
            .expect("test request should build")
    }

    #[test]
    fn either_header_spelling_carries_the_key() {
        assert_eq!(
            presented_key(&request_with("authorization", "Bearer sk-1")),
            Some("sk-1")
        );
        assert_eq!(
            presented_key(&request_with("authorization", "bearer sk-1")),
            Some("sk-1")
        );
        assert_eq!(
            presented_key(&request_with("x-api-key", "sk-1")),
            Some("sk-1")
        );
    }

    #[test]
    fn a_non_bearer_authorization_scheme_carries_no_key() {
        assert_eq!(
            presented_key(&request_with("authorization", "Basic sk-1")),
            None
        );
        assert_eq!(presented_key(&request_with("authorization", "sk-1")), None);
    }
}
