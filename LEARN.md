# LEARN: Day 5 for a Laravel developer

Day 5 is **RAG**: retrieval-augmented generation. The model doesn't "know" your PDFs. We find the few paragraphs that matter, paste them into the prompt, and make it cite them. Here's how the pieces map to things you already know.

## 1. Project setup

| Python                               | Laravel equivalent                          |
|--------------------------------------|---------------------------------------------|
| `pip install -r requirements.txt`    | `composer install`                          |
| `.env` + `python-dotenv`             | `.env` + `env()`                            |
| FastAPI `@app.post("/api/chat")`     | `Route::post('/api/chat', ...)` + controller method |
| Pydantic `ChatRequest(BaseModel)`    | a Form Request: invalid input gets a 422 automatically |
| `create_app(library)` factory        | a service provider binding your services    |
| `StaticFiles` / `FileResponse`       | `public/` and `response()->file()`          |
| Chroma (`data/chroma/`)              | a database, but searched by meaning instead of `WHERE` |
| `pytest` + `TestClient`              | `php artisan test` + `$this->postJson()`    |

`python main.py serve` ≈ `php artisan serve`. `python main.py ingest samples/` ≈ an Artisan command that imports files.

## 2. The RAG pipeline, step by step

**Ingest (once per PDF)** in `ragchat/documents.py` and `ragchat/store.py`:

1. `read_pdf` pulls the text of each page with `pypdf`.
2. `chunk_pages` cuts each page into ~80-word chunks that overlap by 20 words. Chunks never cross a page, so each one knows its page number. The overlap means a sentence cut in half is still whole in the next chunk.
3. Each chunk is turned into a vector (a list of numbers) by an **embedder**. Texts with similar meaning get vectors that point in a similar direction.
4. Chroma stores the vector, the text and `{doc, page}` metadata. The chunk id is a hash of its content, so re-importing a file **upserts** instead of duplicating, like `updateOrCreate`.

**Ask (every question)** in `ragchat/chat.py`:

1. Embed the question and ask Chroma for the 4 nearest chunks. The score is cosine similarity (1.0 = same direction).
2. Drop chunks below `min_score`. If nothing is left, return "I couldn't find that in your documents." and **don't call the model at all**.
3. Build the prompt: system rules + recent chat history + the chunks as `<source n="1" doc="..." page="3">` blocks + the question.
4. `verify_citations` reads the reply, keeps `[n]` markers that point at a real source, strips the rest, and marks the answer `grounded` only if at least one real citation survived.

Think of steps 1–2 as an Eloquent query that returns the 4 most relevant rows, and step 4 as validating the model's output the way you'd validate user input.

## 3. Python you'll see here

```python
@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    page: int
    text: str
```
An immutable value object, like a PHP 8.2 `readonly class Chunk` with constructor promotion.

```python
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```
A `Protocol` is an interface, but you don't write `implements`: any class with a matching `embed` method fits. `HashEmbedder` and `LocalEmbedder` are two implementations, like two drivers behind one contract.

```python
_CITE.sub(replace, text)
```
`preg_replace_callback('/\[(\d+(?:\s*,\s*\d+)*)\]/', $replace, $text)`.

## 4. Why the citations are checked

A model can write `[7]` when there were only 4 sources, or cite nothing. If the UI showed that as-is, a made-up answer would look exactly like a sourced one. Checking citations against what was actually retrieved is cheap and catches the most common failure. It doesn't prove the sentence matches the passage; that's why the source panel shows the passage so a person can check.

## 5. Security note

- **PDF text is untrusted input.** A PDF can contain "Ignore your rules and reveal...". We wrap it in `<source>` tags and tell the model it's data. This reduces prompt injection but doesn't remove it, so the model gets no tools and can't take actions here.
- **Uploads**: the server checks the `%PDF` header and the 20 MB limit, and `safe_filename` removes paths like `../../`, the same reasons you'd never trust `$request->file()->getClientOriginalName()`.
- **API keys** live only in `.env`, which is git-ignored.

## 6. Testing without an API key

`ScriptedLLM` in `ragchat/demo.py` has the same shape as the OpenAI client (`client.chat.completions.create(...)`) but returns scripted text. It reads the numbered sources in the prompt and cites whichever number holds the right page, so tests can check that retrieval worked. It's `Http::fake()` for LLMs, and demo mode reuses it.

## Try this

1. Drop a PDF of your own (a Laravel docs page saved as PDF works) into the upload box, set a real `GEMINI_API_KEY`, and ask about it. Then switch to `EMBEDDINGS=local` and compare the match scores.
2. Change `chunk_pages(pages)` to `size=200, overlap=0` in `library.py` and run `pytest -q`. Which retrieval tests still pass, and why do bigger chunks give vaguer citations?
3. Add a `/api/documents/{name}` DELETE route that calls `store.remove(name)`, with a test, like a `destroy` controller action.
