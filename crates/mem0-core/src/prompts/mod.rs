//! Prompts ported from `mem0/mem0/configs/prompts.py`.
//!
//! The large static prompt constants are codegen'd verbatim into
//! [`constants`] by `mem0-rs/tools/gen_prompts_rs.py`. The dynamic builders
//! (`generate_additive_extraction_prompt`, `get_update_memory_messages`) are
//! ported here.

mod constants;
pub use constants::*;

use serde_json::Value;

const PAST_MESSAGE_TRUNCATION_LIMIT: usize = 300;

fn truncate_content(text: &str, limit: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= limit {
        text.to_string()
    } else {
        let truncated: String = chars[..limit].iter().collect();
        format!("{truncated}...")
    }
}

fn format_conversation_history(messages: &[(String, String)]) -> String {
    let mut result = String::new();
    for (role, content) in messages {
        if !role.is_empty() && !content.is_empty() {
            result.push_str(&format!(
                "{role}: {}\n",
                truncate_content(content, PAST_MESSAGE_TRUNCATION_LIMIT)
            ));
        }
    }
    result
}

fn serialize_memories(memories: &[Value]) -> String {
    // Byte-match Python's json.dumps(memories, ensure_ascii=False): ", " between items and
    // ": " after keys. serde_json has no formatter flag for that — to_string is compact,
    // to_string_pretty is indented — so the value is walked and emitted by hand.
    py_json_dumps(&Value::Array(memories.to_vec())).unwrap_or_else(|| "[]".to_string())
}

/// Render a JSON value exactly like Python's `json.dumps(value, ensure_ascii=False)`.
pub fn py_json_dumps(value: &Value) -> Option<String> {
    let mut out = String::new();
    write_py_json(value, &mut out);
    Some(out)
}

fn write_py_json(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => {
            out.push('"');
            for ch in s.chars() {
                match ch {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                    c => out.push(c),
                }
            }
            out.push('"');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (key, val)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py_json(&Value::String(key.clone()), out);
                out.push_str(": ");
                write_py_json(val, out);
            }
            out.push('}');
        }
    }
}

/// Arguments for [`generate_additive_extraction_prompt`].
pub struct AdditivePromptArgs<'a> {
    /// Narrative profile summary (may be empty).
    pub summary: &'a str,
    /// Memories already captured this session.
    pub recently_extracted_memories: &'a [Value],
    /// Existing relevant memories (`[{id,text}, ...]`).
    pub existing_memories: &'a [Value],
    /// The new messages, pre-flattened to a transcript string.
    pub new_messages: &'a str,
    /// Recent `(role, content)` messages preceding the new ones.
    pub last_k_messages: &'a [(String, String)],
    /// Today's date (defaults to current UTC date).
    pub current_date: Option<String>,
    /// When the conversation occurred (defaults to `current_date`).
    pub observation_date: Option<String>,
    /// Optional custom instructions (highest priority).
    pub custom_instructions: Option<&'a str>,
    /// Whether to require same-language extraction.
    pub use_input_language: bool,
}

/// Build the user-side prompt for additive (ADD-only) extraction with linking.
/// Port of `generate_additive_extraction_prompt`.
pub fn generate_additive_extraction_prompt(args: &AdditivePromptArgs) -> String {
    let current_date = args
        .current_date
        .clone()
        .unwrap_or_else(|| chrono::Utc::now().date_naive().to_string());
    let observation_date = args
        .observation_date
        .clone()
        .unwrap_or_else(|| current_date.clone());

    let mut sections: Vec<String> = Vec::new();
    sections.push(format!("## Summary\n{}", args.summary));
    sections.push(format!(
        "## Last k Messages\n{}",
        format_conversation_history(args.last_k_messages)
    ));
    sections.push(format!(
        "## Recently Extracted Memories\n{}",
        serialize_memories(args.recently_extracted_memories)
    ));
    sections.push(format!(
        "## Existing Memories\n{}",
        serialize_memories(args.existing_memories)
    ));
    sections.push(format!("## New Messages\n{}", args.new_messages));
    sections.push(format!("## Observation Date\n{observation_date}"));
    sections.push(format!("## Current Date\n{current_date}"));

    if let Some(ci) = args.custom_instructions {
        if !ci.is_empty() {
            sections.push(format!("## Custom Instructions\n{ci}"));
        }
    }

    if args.use_input_language {
        // Byte-identical to mem0's prompts.py (v2.2.1). The upstream port abbreviated this to
        // four rules; the omitted ones are exactly the CJK / mixed-script cases.
        sections.push(
            r#"## Language Requirement
CRITICAL: Respond in the SAME LANGUAGE and SCRIPT as the input messages.
1. Match the language of the user's messages exactly — if they write in Korean, extract in Korean; Japanese in Japanese; etc.
2. Preserve the exact script/alphabet of the input.
3. Do NOT translate or transliterate into English unless the input is already in English.
4. Maintain all quality standards (contextual richness, temporal grounding, etc.) regardless of language.
5. Technical terms, proper nouns, and brand names should be preserved in their original form as used in the input.
6. If the input mixes languages (e.g., Hinglish), preserve both the mixed language style AND the script.
7. For Japanese: explicitly resolve omitted subjects using conversation context.
8. For CJK languages: maintain appropriate formality level from the source text."#
                .to_string(),
        );
    }

    sections.push("# Output:".to_string());
    sections.join("\n\n")
}

/// Build the update-memory decision prompt. Port of `get_update_memory_messages`.
pub fn get_update_memory_messages(
    retrieved_old_memory: &[Value],
    response_content: &[Value],
    custom_update_memory_prompt: Option<&str>,
) -> String {
    let base = custom_update_memory_prompt.unwrap_or(DEFAULT_UPDATE_MEMORY_PROMPT);

    let current_memory_part = if !retrieved_old_memory.is_empty() {
        format!(
            "\n    Below is the current content of my memory which I have collected till now. \
             You have to update it in the following format only:\n\n    ```\n    {}\n    ```\n\n    ",
            serialize_memories(retrieved_old_memory)
        )
    } else {
        "\n    Current memory is empty.\n\n    ".to_string()
    };

    format!(
        "{base}\n\n    {current_memory_part}\n\n    \
         The new retrieved facts are mentioned in the triple backticks. \
         You have to analyze the new retrieved facts and determine whether these facts \
         should be added, updated, or deleted in the memory.\n\n    ```\n    {}\n    ```\n\n    \
         You must return your response in the following JSON structure only:\n\n    \
         {{\n        \"memory\" : [\n            {{\n                \"id\" : \"<ID of the memory>\",\n \
                        \"text\" : \"<Content of the memory>\",\n                \
         \"event\" : \"<Operation to be performed>\",\n                \
         \"old_memory\" : \"<Old memory content>\"\n            }},\n            ...\n        ]\n    }}\n\n    \
         Do not return anything except the JSON format.\n    ",
        serialize_memories(response_content)
    )
}
