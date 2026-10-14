//! One driver request → one chat prompt (contracts/apr-code-eval-prompt-v1.yaml,
//! ACE-PRM-001..005).
//!
//! The single-turn agent behind `apr code eval` sends this prompt to `apr serve` and
//! hands the reply back to the driver unchanged; the driver parses it under the run's
//! edit format.
//!
//! - Every file of the working tree is shown verbatim, in path order, in a fence longer
//!   than any backtick run inside it (ACE-PRM-001).
//! - The prompt is built from the request alone, so hidden tests and the gold edit
//!   cannot reach it (ACE-PRM-002).
//! - The system message states the run's edit format with an example that the edit
//!   parser accepts and that applies to the example file (ACE-PRM-003).
//! - Attempt 2 carries the driver's feedback; attempt 1 carries none (ACE-PRM-004).
//! - The same request renders the same bytes; the agentic format needs a tool loop and is
//!   refused (ACE-PRM-005).

use thiserror::Error;

use crate::driver::Request;
use crate::edit::{EditFormat, Files};

/// The two messages sent for one attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prompt {
    /// The system message: the role and the edit format.
    pub system: String,
    /// The user message: the task, the files and, on attempt 2, the feedback.
    pub user: String,
}

/// The request cannot be rendered as a single-turn prompt.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PromptError {
    /// ACE-PRM-005: this format is produced by a tool loop, not by one reply.
    #[error("edit format {0:?} needs a tool loop; the single-turn agent cannot produce it")]
    NeedsToolLoop(EditFormat),
}

/// The file every format example edits.
pub const EXAMPLE_PATH: &str = "src/lib.rs";
/// Its content before the example edit.
pub const EXAMPLE_BEFORE: &str = "pub fn answer() -> u32 {\n    0\n}\n";

const ROLE: &str = "You are fixing a code task. Read the task and the files, then reply with \
the edit that makes the task's tests pass. Reply with the edit only: no explanation, no \
other text outside the edit blocks.";

/// The example reply for `format`, editing [`EXAMPLE_BEFORE`] at [`EXAMPLE_PATH`].
#[must_use]
pub fn example(format: EditFormat) -> Option<String> {
    let p = EXAMPLE_PATH;
    Some(match format {
        EditFormat::Whole => format!("{p}\n```rust\npub fn answer() -> u32 {{\n    42\n}}\n```\n"),
        EditFormat::SearchReplace => {
            format!("{p}\n<<<<<<< SEARCH\n    0\n=======\n    42\n>>>>>>> REPLACE\n")
        }
        EditFormat::Udiff => format!(
            "--- a/{p}\n+++ b/{p}\n@@ -1,3 +1,3 @@\n pub fn answer() -> u32 {{\n-    0\n+    42\n }}\n"
        ),
        EditFormat::Agentic => return None,
    })
}

fn rules(format: EditFormat) -> &'static str {
    match format {
        EditFormat::Whole => {
            "Edit format: whole file. For each file you change, write its path alone on one \
line, then the complete new content in a fenced block."
        }
        EditFormat::SearchReplace => {
            "Edit format: SEARCH/REPLACE. For each change, write the file path alone on one \
line, then a block: <<<<<<< SEARCH, the exact lines to find, =======, the new lines, \
>>>>>>> REPLACE. The search text must occur exactly once in the file. To create a file, \
leave the search part empty."
        }
        EditFormat::Udiff => {
            "Edit format: unified diff. Write --- a/<path> and +++ b/<path> lines, then @@ \
hunks. Context and removed lines must match the file exactly; line numbers are ignored."
        }
        EditFormat::Agentic => "",
    }
}

/// The shortest backtick fence longer than every backtick run in `body`, at least 3.
#[must_use]
pub fn fence_for(body: &str) -> String {
    let longest = body.split(|c| c != '`').map(str::len).max().unwrap_or(0);
    "`".repeat((longest + 1).max(3))
}

fn files_block(files: &Files) -> String {
    let mut out = String::new();
    for (path, body) in files {
        let f = fence_for(body);
        out.push_str(path);
        out.push('\n');
        out.push_str(&f);
        out.push('\n');
        out.push_str(body);
        if !body.ends_with('\n') {
            out.push('\n');
        }
        out.push_str(&f);
        out.push_str("\n\n");
    }
    out
}

/// Render one request (ACE-PRM-001..005).
pub fn render(req: &Request<'_>) -> Result<Prompt, PromptError> {
    let ex = example(req.format).ok_or(PromptError::NeedsToolLoop(req.format))?;
    let system = format!(
        "{ROLE}\n\n{}\n\nExample: if {EXAMPLE_PATH} is\n```\n{EXAMPLE_BEFORE}```\n\nthen \
this reply makes answer() return 42:\n\n{ex}",
        rules(req.format),
    );
    let mut user = format!(
        "# Task {}\n\n{}\n\n# Files\n\n{}",
        req.view.id,
        req.view.prompt.trim_end(),
        files_block(req.files),
    );
    if req.attempt > 1 {
        if let Some(fb) = req.feedback {
            user.push_str("# Your previous edit failed\n\n");
            user.push_str(fb.trim_end());
            user.push_str(
                "\n\nThe files above include your previous edit. Reply with a new edit.\n",
            );
        }
    }
    Ok(Prompt { system, user })
}

#[cfg(test)]
#[path = "prompt_tests.rs"]
mod tests;
