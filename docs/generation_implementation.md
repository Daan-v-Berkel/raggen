RAG Generation System – Implementation Plan

This document outlines the implementation of a generation layer for RAG answer synthesis.

The design prioritizes:
	•	provider-agnostic model usage
	•	support for local and external models
	•	isolation from retrieval logic
	•	minimal coupling with the rest of the system

Generation is optional and must not interfere with retrieval.

⸻

Step 1 — Extend Generation Configuration

Goal

Prepare configuration for selecting generation providers and models.

Requirements

Extend GenerationConfig with fields required for provider-based generation.

Example:

@dataclass
class GenerationConfig:
    enabled: bool = False
    provider: str = ""
    model_id: str = ""

    base_url: str = ""
    api_key_env: str = ""

    temperature: float = 0.0
    max_tokens: int = 800
    timeout_seconds: int = 60

Expected TOML structure

[generation]
enabled = false
provider = ""
model_id = ""
base_url = ""
api_key_env = ""
temperature = 0.0
max_tokens = 800
timeout_seconds = 60

Notes
	•	api_key_env stores the environment variable name, not the key itself.
	•	Secrets must never be stored directly in config files.
	•	All fields must have safe defaults.

⸻

Step 2 — Define Generation Result Model

Goal

Provide a structured return value for generation results.

Requirements

Create generation_models.py.

Example:

@dataclass
class GenerationResult:
    text: str
    model_id: str
    provider: str
    usage: dict | None = None

Notes

Future providers may return:
	•	token usage
	•	provider metadata
	•	cost information

Returning a structured result avoids future refactors.

⸻

Step 3 — Define Provider Interface

Goal

Create a stable interface that all generation providers must implement.

Requirements

Create:

raggen/core/query/providers/base.py

Example:

class GenerationProvider(Protocol):

    def generate(
        self,
        *,
        prompt: str,
        model_id: str,
        settings: dict,
    ) -> GenerationResult:
        ...

Notes

Providers are responsible for:
	•	authentication
	•	API calls
	•	response parsing
	•	retry/timeout handling

Providers must not build prompts or perform retrieval.

⸻

Step 4 — Implement No-Op Provider

Goal

Provide a safe default provider that does nothing.

Requirements

Create:

providers/noop.py

Example:

class NoopProvider:

    def generate(self, *, prompt, model_id, settings):
        raise NotImplementedError(
            "Generation is not implemented for this project."
        )

Notes

Used when:
	•	generation is disabled
	•	provider is unspecified
	•	provider is not implemented

⸻

Step 5 — Implement Provider Loader

Goal

Resolve generation providers from configuration.

Requirements

Create:

raggen/core/query/providers/loader.py

Example:

def load_generation_provider(provider_name: str):

    if provider_name == "":
        return NoopProvider()

    if provider_name == "openai_compatible":
        from .openai_compatible import OpenAICompatibleProvider
        return OpenAICompatibleProvider()

    if provider_name == "ollama":
        from .ollama import OllamaProvider
        return OllamaProvider()

    raise ValueError(f"Unknown generation provider: {provider_name}")

Notes

Provider loading must be deterministic.

Plugin-based loading may be added later.

⸻

Step 6 — Create Prompt Builder

Goal

Separate prompt construction from provider calls.

Requirements

Create:

raggen/core/query/prompt_builder.py

Example:

def build_rag_prompt(query: str, chunks: list[RetrievedChunk]) -> str:

Basic behavior:
	•	include user query
	•	include retrieved chunk text
	•	format clearly for LLM consumption

Example prompt structure:

Answer the question using the context below.

Question:
{query}

Context:
{chunk1}
{chunk2}
...

Notes

Prompt building should remain provider-agnostic.

⸻

Step 7 — Implement Generator Dispatcher

Goal

Provide a single entry point for generation.

Requirements

Create:

raggen/core/query/generator.py

Example interface:

def generate_answer(
    *,
    query: str,
    chunks: list[RetrievedChunk],
    cfg: ProjectConfig,
) -> GenerationResult:

Flow:
	1.	check cfg.generation.enabled
	2.	resolve provider
	3.	build prompt
	4.	call provider.generate()
	5.	return result

Notes

Generator is responsible for:
	•	configuration interpretation
	•	prompt creation
	•	provider dispatch

Generator must not perform retrieval.

⸻

Step 8 — Integrate with Query Service

Goal

Allow query service to optionally generate answers.

Requirements

Modify query/service.py.

After retrieval:

if cfg.generation.enabled:
    result = generate_answer(...)
    answer = result.text
else:
    answer = None

QueryResponse should contain:

answer
used_llm_model

Notes

Retrieval must work independently of generation.

If generation fails, retrieval results should still be returned.

⸻

Step 9 — Environment Variable Handling

Goal

Securely resolve API keys.

Requirements

Providers must resolve keys via:

os.environ[cfg.generation.api_key_env]

If the variable is missing:
	•	raise a clear error
	•	do not silently continue

Example error:

Missing environment variable OPENAI_API_KEY required for generation provider.

Notes

Local providers may not require API keys.

⸻

Step 10 — First Real Provider (Future)

Goal

Add a usable provider implementation.

Recommended first provider:

openai_compatible

Why

Many services support OpenAI-style APIs:
	•	OpenAI
	•	vLLM
	•	LM Studio
	•	Ollama (optionally)
	•	various inference gateways

This allows both:
	•	hosted models
	•	local inference servers

Example config

[generation]
enabled = true
provider = "openai_compatible"
model_id = "gpt-4.1-mini"
api_key_env = "OPENAI_API_KEY"


⸻

Design Principles

Retrieval and generation must remain separate

Retrieval pipeline:

query
→ embedding
→ vector search
→ metadata fetch

Generation pipeline:

query + retrieved chunks
→ prompt builder
→ provider.generate()


⸻

Providers must remain isolated

Provider modules should contain:
	•	HTTP logic
	•	auth
	•	API schemas

They must not import retrieval logic.

⸻

Config must remain stable

The generation config structure must support:
	•	hosted APIs
	•	local servers
	•	SDK-based providers

without redesigning configuration.

⸻

Summary

The generation system introduces:

Component	Purpose
GenerationConfig	user configuration
GenerationResult	structured output
GenerationProvider	provider interface
Provider loader	provider resolution
Prompt builder	RAG prompt construction
Generator dispatcher	provider-agnostic entrypoint

This architecture enables future support for:
	•	OpenAI
	•	Anthropic
	•	Ollama
	•	vLLM
	•	local inference engines
	•	custom providers

without modifying retrieval logic or CLI behavior.
