//! Byte sizes on the command line.
//!
//! `--cache-ram 8GiB` is what an operator sizing host memory actually wants
//! to type; `--cache-ram 8589934592` is what the engine needs. This is the
//! translation, and it is strict about the suffix rather than clever, because
//! a budget silently read as a thousandth of what was meant would show up as
//! a mysterious cache miss rate rather than an error.

/// Parse a byte size: a number, optionally followed by a binary unit.
///
/// `4096`, `512K`, `2M`, `8G`, `1T`, and the `KiB`/`MiB`/`GiB`/`TiB` and
/// `KB`/`MB`/`GB` spellings of the same. All units are powers of 1024 —
/// there is no decimal-megabyte reading here, since the quantity being sized
/// is memory.
/// `--cache-ram full` / `-1`: size the arena from the serving configuration
/// rather than a byte budget, so every slot can keep a retention point at
/// every interval across its whole context. llama.cpp spells the same idea
/// `-cram -1` ("no limit"); the value is a sentinel, not a size.
pub const FULL_COVERAGE: u64 = u64::MAX;

pub fn parse_bytes(text: &str) -> Result<u64, String> {
    let text = text.trim();
    if text.is_empty() {
        return Err("expected a byte size, for example `8GiB`".to_owned());
    }
    match text.to_ascii_lowercase().as_str() {
        "full" | "max" | "-1" => return Ok(FULL_COVERAGE),
        _ => {}
    }
    let digits = text
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(text.len());
    let (number, suffix) = text.split_at(digits);
    let number: u64 = number
        .parse()
        .map_err(|_| format!("`{text}` does not start with a number"))?;

    let scale = match suffix.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1u64,
        "k" | "kb" | "kib" => 1 << 10,
        "m" | "mb" | "mib" => 1 << 20,
        "g" | "gb" | "gib" => 1 << 30,
        "t" | "tb" | "tib" => 1 << 40,
        other => {
            return Err(format!(
                "`{other}` is not a size unit; use B, KiB, MiB, GiB or TiB"
            ));
        }
    };
    number
        .checked_mul(scale)
        .ok_or_else(|| format!("`{text}` overflows a 64-bit byte count"))
}

/// Bytes as GiB, for display.
pub fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_number_is_bytes() {
        assert_eq!(parse_bytes("4096"), Ok(4096));
        assert_eq!(parse_bytes("0"), Ok(0));
    }

    #[test]
    fn every_unit_spelling_means_the_same_power_of_two() {
        for text in ["8G", "8g", "8GB", "8gb", "8GiB", "8gib"] {
            assert_eq!(parse_bytes(text), Ok(8 << 30), "{text}");
        }
        assert_eq!(parse_bytes("512K"), Ok(512 << 10));
        assert_eq!(parse_bytes("2M"), Ok(2 << 20));
        assert_eq!(parse_bytes("1T"), Ok(1 << 40));
    }

    #[test]
    fn surrounding_and_internal_space_is_tolerated() {
        assert_eq!(parse_bytes("  8 GiB "), Ok(8 << 30));
    }

    #[test]
    fn an_unknown_unit_is_refused_rather_than_ignored() {
        // Reading `8GG` as 8 bytes would be a thousand-fold error that shows
        // up as a puzzling miss rate rather than a message.
        assert!(parse_bytes("8GG").is_err());
        assert!(parse_bytes("8 gigs").is_err());
        assert!(parse_bytes("lots").is_err());
    }

    #[test]
    fn full_coverage_is_a_sentinel_not_a_size() {
        // Spelled three ways because llama.cpp uses `-1` for the same idea
        // and a reader coming from there will try it.
        for text in ["full", "FULL", "max", "-1"] {
            assert_eq!(parse_bytes(text), Ok(FULL_COVERAGE), "{text}");
        }
        // And it must not collide with a real size.
        assert_ne!(parse_bytes("8GiB"), Ok(FULL_COVERAGE));
        assert!(parse_bytes("").is_err());
    }

    #[test]
    fn an_overflowing_size_is_refused() {
        assert!(parse_bytes("999999999T").is_err());
    }
}
