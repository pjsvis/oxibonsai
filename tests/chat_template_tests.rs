//! Chat template tests for Qwen3 format.
//!
//! Tests the Qwen3 chat template rendering and multi-turn conversation handling.

use oxibonsai_tokenizer::{ChatMessage, ChatTemplateKind};

/// Test that Qwen3 template renders system + user turn correctly.
#[test]
fn qwen3_template_system_user() {
    let messages = vec![
        ChatMessage::system("You are a helpful assistant."),
        ChatMessage::user("Hello, how are you?"),
    ];

    let rendered = ChatTemplateKind::Qwen.render(&messages);

    assert!(rendered.contains("<|im_start|>system\n"));
    assert!(rendered.contains("<|im_start|>user\n"));
    assert!(rendered.contains("You are a helpful assistant."));
    assert!(rendered.contains("Hello, how are you?"));
    assert!(rendered.contains("<|im_end|>"));
}

/// Test that Qwen3 template includes generation prompt.
#[test]
fn qwen3_template_with_generation_prompt() {
    let messages = vec![
        ChatMessage::system("You are a helpful assistant."),
        ChatMessage::user("Hi"),
    ];

    let rendered = ChatTemplateKind::Qwen.render_with_generation_prompt(&messages);

    assert!(rendered.contains("<|im_start|>assistant\n"));
    assert!(rendered.ends_with("<|im_start|>assistant\n"));
}

/// Test that multi-turn conversation preserves assistant messages.
#[test]
fn qwen3_template_multi_turn() {
    let messages = vec![
        ChatMessage::system("You are a helpful assistant."),
        ChatMessage::user("What is rust?"),
        ChatMessage::assistant("Rust is a programming language."),
        ChatMessage::user("What is it used for?"),
    ];

    let rendered = ChatTemplateKind::Qwen.render(&messages);

    assert!(rendered.contains("<|im_start|>assistant\nRust is a programming language."));
    assert!(rendered.contains("<|im_start|>user\nWhat is it used for?"));
    // Count occurrences - should have 2 user turns after system
    let user_count = rendered.matches("<|im_start|>user\n").count();
    assert_eq!(user_count, 2);
}

/// Test that ChatMessage convenience constructors work.
#[test]
fn chat_message_constructors() {
    let sys = ChatMessage::system("system prompt");
    assert_eq!(sys.role, "system");
    assert_eq!(sys.content, "system prompt");

    let usr = ChatMessage::user("user message");
    assert_eq!(usr.role, "user");
    assert_eq!(usr.content, "user message");

    let asst = ChatMessage::assistant("assistant response");
    assert_eq!(asst.role, "assistant");
    assert_eq!(asst.content, "assistant response");
}

/// Test that empty history works (single turn).
#[test]
fn qwen3_template_single_turn() {
    let messages = vec![
        ChatMessage::system("You are a helpful assistant."),
        ChatMessage::user("Hello"),
    ];

    let rendered = ChatTemplateKind::Qwen.render(&messages);

    // Should have system and user
    assert!(rendered.contains("<|im_start|>system\n"));
    assert!(rendered.contains("<|im_start|>user\n"));

    // Should NOT have assistant turn (no response yet)
    let assistant_in_system = rendered.match_indices("<|im_start|>assistant\n").count();
    assert_eq!(assistant_in_system, 0);
}

/// Test that system prompt can be updated.
#[test]
fn system_prompt_variations() {
    let messages1 = vec![
        ChatMessage::system("You are a calculator."),
        ChatMessage::user("2+2"),
    ];

    let messages2 = vec![
        ChatMessage::system("You are a poet."),
        ChatMessage::user("2+2"),
    ];

    let rendered1 = ChatTemplateKind::Qwen.render(&messages1);
    let rendered2 = ChatTemplateKind::Qwen.render(&messages2);

    assert!(rendered1.contains("You are a calculator."));
    assert!(rendered2.contains("You are a poet."));
    assert!(!rendered1.contains("You are a poet."));
    assert!(!rendered2.contains("You are a calculator."));
}
