//! Workspace automation. Run via `cargo run -p xtask -- <task>`.
//!
//! Tasks:
//! - `gen-prompts` — regenerate the verbatim prompt constants from the Python source.
//! - `ci`          — run `cargo fmt --check`, `cargo clippy -D warnings`, and `cargo test`.

use std::process::{exit, Command};

fn main() {
    let task = std::env::args().nth(1);
    let code = match task.as_deref() {
        Some("gen-prompts") => gen_prompts(),
        Some("ci") => ci(),
        _ => {
            eprintln!("usage: cargo run -p xtask -- <gen-prompts|ci>");
            1
        }
    };
    exit(code);
}

fn run(program: &str, args: &[&str]) -> bool {
    eprintln!("$ {program} {}", args.join(" "));
    Command::new(program)
        .args(args)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn gen_prompts() -> i32 {
    // The generator lives at the repo root relative to the workspace.
    if run("python3", &["mem0-rs/tools/gen_prompts_rs.py"])
        || run("python3", &["tools/gen_prompts_rs.py"])
    {
        0
    } else {
        eprintln!("gen-prompts failed (is python3 available and the path correct?)");
        1
    }
}

fn ci() -> i32 {
    let steps: &[(&str, &[&str])] = &[
        ("cargo", &["fmt", "--all", "--check"]),
        (
            "cargo",
            &["clippy", "--all-targets", "--", "-D", "warnings"],
        ),
        ("cargo", &["test", "--workspace"]),
    ];
    for (program, args) in steps {
        if !run(program, args) {
            return 1;
        }
    }
    0
}
