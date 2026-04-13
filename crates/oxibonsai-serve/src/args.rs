//! Command-line argument parsing for oxibonsai-serve.
//!
//! Pure `std::env::args()` parsing — no clap, no structopt.
//!
//! # Supported flags
//!
//! ```text
//! --host <HOST>         Bind host          (default: 0.0.0.0)
//! --port <PORT>         Bind port          (default: 8080)
//! --model <PATH>        Path to GGUF model file
//! --tokenizer <PATH>    Path to tokenizer  (optional)
//! --max-tokens <N>      Default max tokens (default: 256)
//! --temperature <F>     Default temperature (default: 0.7)
//! --seed <N>            RNG seed           (default: 42)
//! --log-level <LEVEL>   Logging level      (default: info)
//! --help                Print help and exit
//! --version             Print version and exit
//! ```

use thiserror::Error;

// ─── Error type ────────────────────────────────────────────────────────────

/// Errors that can occur while parsing command-line arguments.
#[derive(Debug, Error, PartialEq)]
pub enum ParseError {
    /// An unrecognised flag was encountered.
    #[error("unknown option: {0}")]
    UnknownOption(String),

    /// A flag that requires a value was given without one.
    #[error("missing value for option: {0}")]
    MissingValue(String),

    /// A value could not be interpreted as the expected type.
    #[error("invalid value '{value}' for option '{option}': {reason}")]
    InvalidValue {
        option: String,
        value: String,
        reason: String,
    },
}

// ─── ServerArgs ────────────────────────────────────────────────────────────

/// Parsed server configuration derived from argv.
#[derive(Debug, Clone, PartialEq)]
pub struct ServerArgs {
    /// Host address to bind to.
    pub host: String,
    /// TCP port to listen on.
    pub port: u16,
    /// Optional path to a GGUF model file.
    pub model_path: Option<String>,
    /// Optional path to a tokenizer file/directory.
    pub tokenizer_path: Option<String>,
    /// Default maximum tokens to generate per request.
    pub max_tokens: usize,
    /// Default sampling temperature.
    pub temperature: f32,
    /// RNG seed for reproducible generation.
    pub seed: u64,
    /// Tracing log level string (e.g. "info", "debug").
    pub log_level: String,
}

impl Default for ServerArgs {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            model_path: None,
            tokenizer_path: None,
            max_tokens: 256,
            temperature: 0.7,
            seed: 42,
            log_level: "info".to_string(),
        }
    }
}

// ─── Parsing ───────────────────────────────────────────────────────────────

/// Parse a slice of argument strings (typically from `std::env::args().collect()`).
///
/// The first element (program name) is ignored automatically.
///
/// Returns `Ok(None)` if `--help` or `--version` was requested and handled
/// (they print to stderr/stdout and the caller should exit 0).
/// Returns `Ok(Some(args))` for a successful parse.
/// Returns `Err(ParseError)` if the arguments are malformed.
pub fn parse_args_from(argv: &[String]) -> Result<Option<ServerArgs>, ParseError> {
    let mut args = ServerArgs::default();

    // Skip argv[0] (program name) if present.
    let mut iter = argv.iter().peekable();
    // If the first token looks like a program path (no leading '--'), skip it.
    if let Some(first) = iter.peek() {
        if !first.starts_with('-') {
            iter.next();
        }
    }

    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--help" | "-h" => {
                print_help();
                return Ok(None);
            }
            "--version" | "-V" => {
                print_version();
                return Ok(None);
            }
            "--host" => {
                let val = next_value(&mut iter, "--host")?;
                args.host = val.to_string();
            }
            "--port" => {
                let val = next_value(&mut iter, "--port")?;
                args.port = val.parse::<u16>().map_err(|_| ParseError::InvalidValue {
                    option: "--port".to_string(),
                    value: val.to_string(),
                    reason: "must be an integer in 1–65535".to_string(),
                })?;
            }
            "--model" => {
                let val = next_value(&mut iter, "--model")?;
                args.model_path = Some(val.to_string());
            }
            "--tokenizer" => {
                let val = next_value(&mut iter, "--tokenizer")?;
                args.tokenizer_path = Some(val.to_string());
            }
            "--max-tokens" => {
                let val = next_value(&mut iter, "--max-tokens")?;
                args.max_tokens = val.parse::<usize>().map_err(|_| ParseError::InvalidValue {
                    option: "--max-tokens".to_string(),
                    value: val.to_string(),
                    reason: "must be a non-negative integer".to_string(),
                })?;
            }
            "--temperature" => {
                let val = next_value(&mut iter, "--temperature")?;
                args.temperature = val.parse::<f32>().map_err(|_| ParseError::InvalidValue {
                    option: "--temperature".to_string(),
                    value: val.to_string(),
                    reason: "must be a floating-point number".to_string(),
                })?;
            }
            "--seed" => {
                let val = next_value(&mut iter, "--seed")?;
                args.seed = val.parse::<u64>().map_err(|_| ParseError::InvalidValue {
                    option: "--seed".to_string(),
                    value: val.to_string(),
                    reason: "must be a non-negative integer".to_string(),
                })?;
            }
            "--log-level" => {
                let val = next_value(&mut iter, "--log-level")?;
                validate_log_level(val)?;
                args.log_level = val.to_string();
            }
            other => {
                return Err(ParseError::UnknownOption(other.to_string()));
            }
        }
    }

    Ok(Some(args))
}

// ─── Internal helpers ──────────────────────────────────────────────────────

/// Advance the iterator and return the next token, or return a `MissingValue`
/// error if the iterator is exhausted.
fn next_value<'a>(
    iter: &mut std::iter::Peekable<std::slice::Iter<'a, String>>,
    flag: &str,
) -> Result<&'a str, ParseError> {
    match iter.next() {
        Some(v) => Ok(v.as_str()),
        None => Err(ParseError::MissingValue(flag.to_string())),
    }
}

/// Validate that the log-level string is one of the known tracing levels.
fn validate_log_level(level: &str) -> Result<(), ParseError> {
    match level {
        "error" | "warn" | "info" | "debug" | "trace" | "off" => Ok(()),
        other => Err(ParseError::InvalidValue {
            option: "--log-level".to_string(),
            value: other.to_string(),
            reason: "must be one of: error, warn, info, debug, trace, off".to_string(),
        }),
    }
}

// ─── Help / version output ─────────────────────────────────────────────────

/// Print the help text to stderr.
pub fn print_help() {
    eprintln!(
        "\
Usage: oxibonsai-serve [OPTIONS]

Options:
  --host <HOST>         Bind host (default: 0.0.0.0)
  --port <PORT>         Bind port (default: 8080)
  --model <PATH>        Path to GGUF model file
  --tokenizer <PATH>    Path to tokenizer (optional)
  --max-tokens <N>      Default max tokens (default: 256)
  --temperature <F>     Default temperature (default: 0.7)
  --seed <N>            RNG seed (default: 42)
  --log-level <LEVEL>   Logging level: error/warn/info/debug/trace (default: info)
  --help, -h            Show this help
  --version, -V         Show version"
    );
}

/// Print the version string to stdout.
pub fn print_version() {
    println!("oxibonsai-serve {}", crate::banner::VERSION);
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &str) -> String {
        v.to_string()
    }

    fn args(flags: &[&str]) -> Vec<String> {
        // Prepend a fake program name so parse_args_from can skip it.
        std::iter::once("oxibonsai-serve")
            .chain(flags.iter().copied())
            .map(s)
            .collect()
    }

    #[test]
    fn defaults_are_sensible() {
        let defaults = ServerArgs::default();
        assert_eq!(defaults.port, 8080);
        assert_eq!(defaults.max_tokens, 256);
        assert!((defaults.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(defaults.seed, 42);
        assert_eq!(defaults.host, "0.0.0.0");
        assert_eq!(defaults.log_level, "info");
        assert!(defaults.model_path.is_none());
        assert!(defaults.tokenizer_path.is_none());
    }

    #[test]
    fn parse_empty_gives_defaults() {
        let result = parse_args_from(&args(&[])).expect("should parse");
        let parsed = result.expect("should not be help/version");
        assert_eq!(parsed, ServerArgs::default());
    }

    #[test]
    fn parse_host_port() {
        let result = parse_args_from(&args(&["--host", "127.0.0.1", "--port", "3000"]))
            .expect("should parse")
            .expect("should not be help/version");
        assert_eq!(result.host, "127.0.0.1");
        assert_eq!(result.port, 3000);
    }

    #[test]
    fn parse_model_path() {
        let result = parse_args_from(&args(&["--model", "/path/to/model.gguf"]))
            .expect("should parse")
            .expect("should not be help/version");
        assert_eq!(result.model_path, Some("/path/to/model.gguf".to_string()));
    }

    #[test]
    fn parse_temperature() {
        let result = parse_args_from(&args(&["--temperature", "0.5"]))
            .expect("should parse")
            .expect("should not be help/version");
        assert!((result.temperature - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn parse_seed() {
        let result = parse_args_from(&args(&["--seed", "1234"]))
            .expect("should parse")
            .expect("should not be help/version");
        assert_eq!(result.seed, 1234);
    }

    #[test]
    fn parse_log_level() {
        let result = parse_args_from(&args(&["--log-level", "debug"]))
            .expect("should parse")
            .expect("should not be help/version");
        assert_eq!(result.log_level, "debug");
    }

    #[test]
    fn parse_unknown_option_error() {
        let err = parse_args_from(&args(&["--unknown"])).expect_err("should be an error");
        assert!(matches!(err, ParseError::UnknownOption(ref s) if s == "--unknown"));
    }

    #[test]
    fn parse_missing_value_error() {
        // --port with no following value
        let err = parse_args_from(&args(&["--port"])).expect_err("should be an error");
        assert!(matches!(err, ParseError::MissingValue(ref s) if s == "--port"));
    }

    #[test]
    fn parse_invalid_port_error() {
        let err = parse_args_from(&args(&["--port", "abc"])).expect_err("should be an error");
        assert!(
            matches!(err, ParseError::InvalidValue { ref option, .. } if option == "--port"),
            "unexpected error variant: {err:?}"
        );
    }

    #[test]
    fn help_flag_returns_none() {
        let result = parse_args_from(&args(&["--help"])).expect("should not error");
        assert!(result.is_none(), "expected None for --help");
    }

    #[test]
    fn version_flag_returns_none() {
        let result = parse_args_from(&args(&["--version"])).expect("should not error");
        assert!(result.is_none(), "expected None for --version");
    }
}
