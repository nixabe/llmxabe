//! Logging setup shared by every `llmxabe` binary.
//!
//! Every binary in this workspace routes its console output through
//! [`tracing`] rather than `println!`, and every one of them accepts the same
//! flag:
//!
//! ```text
//! --log-level info | debug | trace     (default: info)
//! --log-level=debug                    (equals form, also accepted)
//! ```
//!
//! # Why only three levels
//!
//! `tracing`'s level filter is an ordering, not a set: a filter of `info`
//! admits `INFO`, `WARN` and `ERROR`, a filter of `debug` admits those plus
//! `DEBUG`, and so on. So `warn` and `error` need no flag value of their own —
//! they are visible at every setting the flag can express, which is exactly
//! the behaviour wanted. Offering `--log-level error` would only let a user
//! *hide* problems, so the flag does not offer it.
//!
//! # Why tool output is `INFO`, not `println!`
//!
//! The tables these binaries print — `gguf_info`'s tensor inventory,
//! `bench_forward`'s timings, `profile_forward`'s stage breakdown — are the
//! product of the tool, so they must appear by default. `INFO` is the level
//! that means "appears by default", so that is where they go. The consequence
//! is that a caller can raise verbosity to `debug` without losing the table,
//! and can filter on target (`RUST_LOG=xabe_engine=debug`) without losing it
//! either.
//!
//! That only works if the output still *looks* like a table, which is what
//! [`ToolFormat`] is for: at `info` an event is written as its message and
//! nothing else, so the rendering is byte-for-byte what `println!` produced
//! before. `WARN` and `ERROR` keep a level tag, because a warning that looks
//! like ordinary output is a warning nobody reads. At `debug` and `trace` the
//! format switches to level-plus-target, because at that point the reader
//! wants provenance more than alignment.
//!
//! # Interaction with `RUST_LOG`
//!
//! - No `--log-level` on the command line: the filter comes from `RUST_LOG`,
//!   defaulting to `info` when that is unset. Per-target directives work
//!   normally.
//! - `--log-level` given: it wins outright and `RUST_LOG` is ignored. A
//!   warning says so, because silently discarding an environment variable the
//!   user set is worse than the inconsistency it avoids.
//!
//! The explicit flag winning is the deliberate choice: an argument typed on
//! the command line is more recent and more specific than an exported
//! variable, and a user debugging one invocation should not have to unset
//! their shell profile to do it.
//!
//! # Where output goes
//!
//! `INFO`, `DEBUG` and `TRACE` go to stdout; `WARN` and `ERROR` go to stderr.
//! That keeps `gguf-info model.gguf | less` piping a clean table while a
//! warning still reaches a terminal whose stdout has been redirected away.

use std::fmt;
use std::str::FromStr;

use tracing::{Event, Level, Subscriber};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::Writer;
use tracing_subscriber::fmt::writer::MakeWriterExt;
use tracing_subscriber::fmt::{FmtContext, FormatEvent, FormatFields};
use tracing_subscriber::registry::LookupSpan;

/// The verbosity settings `--log-level` accepts.
///
/// Deliberately not a re-export of [`tracing::Level`]: that type has `WARN`
/// and `ERROR` variants, and offering them as flag values would let a caller
/// suppress the diagnostics the tool exists to surface. See the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogLevel {
    /// Tool output and anything worse. The default.
    #[default]
    Info,
    /// Adds per-stage detail: buffer sizes, launch geometry, resolved paths.
    Debug,
    /// Adds per-item detail — per-tensor, per-kernel, per-token.
    Trace,
}

impl LogLevel {
    /// The `tracing` level this admits, inclusive of everything above it.
    pub fn as_tracing(self) -> Level {
        match self {
            Self::Info => Level::INFO,
            Self::Debug => Level::DEBUG,
            Self::Trace => Level::TRACE,
        }
    }

    /// The flag values a user may type, in verbosity order.
    pub const ACCEPTED: [&'static str; 3] = ["info", "debug", "trace"];
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Info => "info",
            Self::Debug => "debug",
            Self::Trace => "trace",
        })
    }
}

impl FromStr for LogLevel {
    type Err = ArgError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "info" => Ok(Self::Info),
            "debug" => Ok(Self::Debug),
            "trace" => Ok(Self::Trace),
            other => Err(ArgError::UnknownLevel(other.to_string())),
        }
    }
}

/// What can go wrong parsing `--log-level`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgError {
    /// The value is not one of [`LogLevel::ACCEPTED`].
    UnknownLevel(String),
    /// `--log-level` was the last argument, with nothing after it.
    MissingValue,
}

impl fmt::Display for ArgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownLevel(got) => write!(
                f,
                "unknown --log-level {got:?}; expected one of {}",
                LogLevel::ACCEPTED.join(", ")
            ),
            Self::MissingValue => write!(
                f,
                "--log-level needs a value: one of {}",
                LogLevel::ACCEPTED.join(", ")
            ),
        }
    }
}

impl std::error::Error for ArgError {}

/// The outcome of scanning a command line for `--log-level`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedArgs {
    /// The requested level, or [`LogLevel::Info`] when the flag was absent.
    pub level: LogLevel,
    /// Whether the flag was actually present. Drives whether `RUST_LOG` is
    /// honoured; see the module docs.
    pub explicit: bool,
    /// Every argument that was not part of the flag, in their original order.
    pub rest: Vec<String>,
}

/// Scan an argument list for `--log-level`, returning it and everything else.
///
/// The caller passes arguments *without* the program name. Both `--log-level
/// debug` and `--log-level=debug` are accepted. A repeated flag takes the last
/// value, matching the usual convention that a later argument overrides an
/// earlier one.
///
/// This is separated from [`init`] so it can be tested without installing a
/// process-global subscriber, which is a once-per-process operation and
/// therefore untestable more than once.
pub fn parse<I, S>(args: I) -> Result<ParsedArgs, ArgError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut level = LogLevel::default();
    let mut explicit = false;
    let mut rest = Vec::new();
    let mut iter = args.into_iter();

    while let Some(arg) = iter.next() {
        let arg = arg.as_ref();
        if let Some(value) = arg.strip_prefix("--log-level=") {
            level = value.parse()?;
            explicit = true;
        } else if arg == "--log-level" {
            let value = iter.next().ok_or(ArgError::MissingValue)?;
            level = value.as_ref().parse()?;
            explicit = true;
        } else {
            rest.push(arg.to_string());
        }
    }

    Ok(ParsedArgs {
        level,
        explicit,
        rest,
    })
}

/// One line of `--help` text, so every binary documents the flag identically.
pub const FLAG_HELP: &str = "  --log-level <info|debug|trace>   console verbosity (default: info); \
                             warnings and errors always show";

/// Install the process-wide subscriber.
///
/// Call once, before any `tracing` macro. Returns `false` if a subscriber was
/// already installed, which is not treated as an error — a test harness or an
/// embedding process may have installed its own, and stamping on it would be
/// worse than deferring to it.
pub fn init(level: LogLevel, explicit: bool) -> bool {
    let verbose = level != LogLevel::Info;
    let rust_log = std::env::var("RUST_LOG").ok();

    // `LogLevel`'s Display is a valid filter directive, which
    // `every_accepted_value_round_trips_through_display` keeps true.
    let filter = match (&rust_log, explicit) {
        (Some(directives), false) => EnvFilter::new(directives.clone()),
        // No RUST_LOG, or an explicit flag that overrides it — module docs.
        _ => EnvFilter::new(level.to_string()),
    };

    // Tool output on stdout, diagnostics on stderr — the ordinary Unix split,
    // so `gguf-info model.gguf | less` still pipes a clean table and a warning
    // still reaches a terminal that has had its stdout redirected away.
    let writer = std::io::stdout
        .with_min_level(Level::INFO)
        .or_else(std::io::stderr.with_max_level(Level::WARN));

    let installed = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .event_format(ToolFormat { verbose })
        .with_writer(writer)
        .try_init()
        .is_ok();

    if installed && explicit && rust_log.is_some() {
        tracing::warn!(
            "RUST_LOG is set but --log-level was given explicitly; using --log-level {level} and \
             ignoring RUST_LOG"
        );
    }
    installed
}

/// Parse `std::env::args()`, install the subscriber, and return the remaining
/// arguments.
///
/// On a bad `--log-level` value this writes the error to stderr and exits with
/// status 2. That is deliberate for a command-line tool: continuing at the
/// default level after the user asked for something else would produce output
/// that silently disagrees with what was requested.
pub fn init_from_args() -> Vec<String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    match parse(&raw) {
        Ok(parsed) => {
            init(parsed.level, parsed.explicit);
            parsed.rest
        }
        Err(e) => {
            // `eprintln!` rather than `error!` on purpose: the failure is that
            // we could not decide what the subscriber should be, so there is
            // no subscriber to log through yet.
            eprintln!("error: {e}");
            eprintln!("{FLAG_HELP}");
            std::process::exit(2);
        }
    }
}

/// Event formatter that keeps tool output looking like tool output.
///
/// At `info` an event is its message and nothing else — no timestamp, no
/// target, no level tag — so a table printed through `tracing::info!` renders
/// exactly as the `println!` it replaced. `WARN` and `ERROR` are tagged even
/// there, because an unlabelled warning is a warning that gets read as data.
///
/// At `debug` and `trace` every event is tagged with level and target, since
/// a reader at those levels is looking for where output came from and no
/// longer needs the columns to line up.
struct ToolFormat {
    verbose: bool,
}

/// Write `rendered` to `out`, one line at a time, with `prefix` on each
/// non-empty line.
///
/// Split out from [`ToolFormat::format_event`] so it can be tested without
/// standing up a subscriber, because the multi-line cases are exactly the ones
/// that are easy to get subtly wrong.
fn prefix_lines(prefix: &str, rendered: &str, out: &mut impl fmt::Write) -> fmt::Result {
    for line in rendered.split('\n') {
        if line.is_empty() {
            writeln!(out)?;
        } else {
            writeln!(out, "{prefix}{line}")?;
        }
    }
    Ok(())
}

impl<S, N> FormatEvent<S, N> for ToolFormat
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        let meta = event.metadata();
        let level = *meta.level();

        let prefix = if self.verbose {
            format!("{:>5} {}: ", level, meta.target())
        } else if level == Level::WARN || level == Level::ERROR {
            format!("{level}: ")
        } else {
            String::new()
        };

        // Render first, then prefix line by line. Several of these messages
        // are multi-line by design — a table header with a blank line before
        // it, a paragraph of explanation after a result — and prefixing only
        // the first line would leave the continuations looking like unrelated
        // output. Blank lines keep no prefix, so a leading `\n` still renders
        // as a blank separator rather than a bare `DEBUG target:`.
        let mut rendered = String::new();
        ctx.field_format()
            .format_fields(Writer::new(&mut rendered), event)?;

        prefix_lines(&prefix, &rendered, &mut writer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_is_info() {
        let parsed = parse::<[&str; 0], &str>([]).expect("empty args parse");
        assert_eq!(parsed.level, LogLevel::Info);
        assert!(
            !parsed.explicit,
            "an absent flag must not count as explicit, or RUST_LOG would never be honoured"
        );
        assert!(parsed.rest.is_empty());
    }

    #[test]
    fn both_flag_forms_are_accepted() {
        let spaced = parse(["--log-level", "debug"]).expect("spaced form");
        let equals = parse(["--log-level=debug"]).expect("equals form");
        assert_eq!(spaced.level, LogLevel::Debug);
        assert_eq!(equals.level, LogLevel::Debug);
        assert!(spaced.explicit && equals.explicit);
    }

    #[test]
    fn other_arguments_survive_in_order() {
        let parsed = parse([
            "model.gguf",
            "--log-level",
            "trace",
            "--tensors",
            "-n",
            "512",
        ])
        .expect("mixed args");
        assert_eq!(parsed.level, LogLevel::Trace);
        assert_eq!(
            parsed.rest,
            vec!["model.gguf", "--tensors", "-n", "512"],
            "the flag must be removed and nothing else disturbed"
        );
    }

    #[test]
    fn a_repeated_flag_takes_the_last_value() {
        let parsed = parse(["--log-level", "trace", "--log-level=info"]).expect("repeated flag");
        assert_eq!(parsed.level, LogLevel::Info);
    }

    #[test]
    fn a_bad_value_is_an_error_not_a_silent_default() {
        // The failure mode this guards against: accepting `warn`, filtering out
        // INFO, and leaving the user staring at a tool that printed nothing.
        assert_eq!(
            parse(["--log-level", "warn"]),
            Err(ArgError::UnknownLevel("warn".into()))
        );
        assert_eq!(
            parse(["--log-level", "verbose"]),
            Err(ArgError::UnknownLevel("verbose".into()))
        );
        assert_eq!(parse(["--log-level"]), Err(ArgError::MissingValue));
    }

    #[test]
    fn levels_are_case_insensitive() {
        assert_eq!("DEBUG".parse::<LogLevel>(), Ok(LogLevel::Debug));
        assert_eq!("Trace".parse::<LogLevel>(), Ok(LogLevel::Trace));
    }

    #[test]
    fn every_accepted_value_round_trips_through_display() {
        for name in LogLevel::ACCEPTED {
            let level: LogLevel = name.parse().expect("ACCEPTED values must parse");
            assert_eq!(
                level.to_string(),
                name,
                "Display and FromStr must agree, or the warning text in init() would name a \
                 value the flag rejects"
            );
        }
    }

    /// The property the whole design rests on: warnings and errors are visible
    /// at every level the flag can express. If this ever fails, the module's
    /// justification for not offering `--log-level warn` is void.
    #[test]
    fn warn_and_error_are_admitted_at_every_accepted_level() {
        for name in LogLevel::ACCEPTED {
            let level: LogLevel = name.parse().expect("ACCEPTED values must parse");
            let admitted = level.as_tracing();
            assert!(
                Level::WARN <= admitted,
                "{name} must admit WARN, or --log-level {name} could hide a warning"
            );
            assert!(
                Level::ERROR <= admitted,
                "{name} must admit ERROR, or --log-level {name} could hide an error"
            );
            assert!(
                Level::INFO <= admitted,
                "{name} must admit INFO, or --log-level {name} would suppress the tool's own \
                 output"
            );
        }
    }

    fn prefixed(prefix: &str, rendered: &str) -> String {
        let mut out = String::new();
        prefix_lines(prefix, rendered, &mut out).expect("String writes cannot fail");
        out
    }

    #[test]
    fn an_unprefixed_line_is_byte_for_byte_what_println_produced() {
        // This is the whole reason tool output can live at INFO: at the
        // default level the rendering must be indistinguishable from the
        // `println!` it replaced, or every table in the workspace reflows.
        assert_eq!(prefixed("", "  q6_K   count=80"), "  q6_K   count=80\n");
    }

    #[test]
    fn every_line_of_a_multi_line_message_is_prefixed() {
        // Prefixing only the first line would leave continuations looking
        // like output from somewhere else entirely.
        assert_eq!(
            prefixed("DEBUG t: ", "header\n  row one\n  row two"),
            "DEBUG t: header\nDEBUG t:   row one\nDEBUG t:   row two\n"
        );
    }

    #[test]
    fn blank_separator_lines_stay_blank() {
        // Roughly twenty call sites open with "\n" to put a blank line before
        // a section heading. Prefixing that empty line would print a bare
        // `DEBUG target:` with nothing after it.
        assert_eq!(
            prefixed("DEBUG t: ", "\ntensor type histogram:"),
            "\nDEBUG t: tensor type histogram:\n"
        );
        assert_eq!(
            prefixed("DEBUG t: ", "arena: 2.291 GiB\n"),
            "DEBUG t: arena: 2.291 GiB\n\n"
        );
    }

    #[test]
    fn the_accepted_list_is_in_verbosity_order() {
        // ACCEPTED is quoted in error messages and in --help; if it were out of
        // order the help text would mislead about which setting is noisier.
        let levels: Vec<Level> = LogLevel::ACCEPTED
            .iter()
            .map(|n| n.parse::<LogLevel>().expect("parses").as_tracing())
            .collect();
        assert!(
            levels.windows(2).all(|w| w[0] < w[1]),
            "ACCEPTED must run quiet to loud, got {levels:?}"
        );
    }
}
