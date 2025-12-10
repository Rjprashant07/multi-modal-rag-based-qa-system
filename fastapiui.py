from fastapi.responses import HTMLResponse




@app.get("/qa", response_class=HTMLResponse)
async def qa_frontend():
    """Simple HTML frontend for uploading a PDF and asking questions."""
    return """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <title>Multi-Modal RAG QA</title>
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      max-width: 900px;
      margin: 20px auto;
      padding: 0 16px;
      background: #f5f5f5;
    }
    h1 {
      text-align: center;
    }
    .card {
      background: white;
      padding: 16px 20px;
      margin-bottom: 20px;
      border-radius: 12px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    label {
      font-weight: 600;
      display: block;
      margin-bottom: 4px;
    }
    input[type="text"], textarea {
      width: 100%;
      padding: 8px 10px;
      border-radius: 8px;
      border: 1px solid #ccc;
      box-sizing: border-box;
      margin-bottom: 10px;
      font-family: inherit;
      font-size: 14px;
    }
    input[type="file"] {
      margin: 8px 0 12px;
    }
    button {
      padding: 8px 16px;
      border-radius: 999px;
      border: none;
      cursor: pointer;
      font-weight: 600;
      font-size: 14px;
      margin-top: 4px;
    }
    button.primary {
      background: #2563eb;
      color: white;
    }
    button.secondary {
      background: #e5e7eb;
      color: #111827;
    }
    .row {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .row input[type="text"] {
      flex: 1;
      margin-bottom: 0;
    }
    .pill {
      display: inline-block;
      padding: 2px 8px;
      font-size: 12px;
      border-radius: 999px;
      background: #e5e7eb;
      margin-right: 4px;
    }
    pre {
      white-space: pre-wrap;
      background: #111827;
      color: #e5e7eb;
      padding: 12px;
      border-radius: 8px;
      font-size: 13px;
      max-height: 300px;
      overflow: auto;
    }
    #status {
      margin-top: 6px;
      font-size: 13px;
      color: #4b5563;
    }
    .source {
      background: #f9fafb;
      padding: 8px 10px;
      border-radius: 8px;
      margin-bottom: 6px;
      font-size: 13px;
    }
    .source-header {
      font-weight: 600;
      margin-bottom: 4px;
    }
  </style>
</head>
<body>
  <h1>📄 Multi-Modal RAG QA</h1>

  <!-- Upload / ingest card -->
  <div class="card">
    <h2>1. Upload Document</h2>
    <label for="pdfFile">PDF file</label>
    <input type="file" id="pdfFile" accept="application/pdf" />

    <label for="docIdInput">Document ID (optional)</label>
    <div class="row">
      <input type="text" id="docIdInput" placeholder="Leave empty to auto-generate" />
      <button class="secondary" type="button" onclick="clearDocId()">Clear</button>
    </div>

    <button class="primary" type="button" onclick="uploadPdf()">Ingest PDF</button>
    <div id="status"></div>
    <div id="currentDoc"></div>
  </div>

  <!-- QA card -->
  <div class="card">
    <h2>2. Ask a Question</h2>
    <label for="questionInput">Question</label>
    <textarea id="questionInput" rows="3" placeholder="Example: What is the GDP growth outlook for next year?"></textarea>

    <div class="row">
      <label for="topKInput" style="margin: 0;">Top-k chunks:</label>
      <input type="text" id="topKInput" value="8" style="max-width: 80px;" />
      <button class="primary" type="button" onclick="askQuestion()">Ask</button>
    </div>

    <div id="answerBox" style="margin-top: 12px;"></div>
    <div id="sourcesBox" style="margin-top: 10px;"></div>
  </div>

  <script>
    const baseUrl = window.location.origin;  // same host as FastAPI

    function setStatus(msg) {
      document.getElementById("status").innerText = msg || "";
    }

    function setCurrentDoc(docId) {
      if (!docId) {
        document.getElementById("currentDoc").innerHTML = "";
        return;
      }
      document.getElementById("currentDoc").innerHTML =
        '<span class="pill">Current doc_id:</span> <code>' + docId + '</code>';
    }

    function clearDocId() {
      document.getElementById("docIdInput").value = "";
      setCurrentDoc("");
    }

    async function uploadPdf() {
      const fileInput = document.getElementById("pdfFile");
      const docIdInput = document.getElementById("docIdInput");

      if (fileInput.files.length === 0) {
        alert("Please select a PDF file first.");
        return;
      }

      const file = fileInput.files[0];
      const formData = new FormData();
      formData.append("file", file);
      if (docIdInput.value.trim() !== "") {
        formData.append("doc_id", docIdInput.value.trim());
      }

      setStatus("Uploading and ingesting PDF...");

      try {
        const resp = await fetch(baseUrl + "/ingest", {
          method: "POST",
          body: formData
        });

        if (!resp.ok) {
          const text = await resp.text();
          console.error("Ingest error:", text);
          setStatus("Error during ingest: " + resp.status + " " + resp.statusText);
          return;
        }

        const data = await resp.json();
        console.log("Ingest response:", data);
        setStatus("Ingestion complete. Chunks indexed: " + data.num_chunks);
        docIdInput.value = data.doc_id;
        setCurrentDoc(data.doc_id);
      } catch (err) {
        console.error(err);
        setStatus("Error during ingest: " + err);
      }
    }

    async function askQuestion() {
      const docId = document.getElementById("docIdInput").value.trim();
      const question = document.getElementById("questionInput").value.trim();
      const topKStr = document.getElementById("topKInput").value.trim();

      if (!docId) {
        alert("Please upload a document first (or specify doc_id).");
        return;
      }
      if (!question) {
        alert("Please enter a question.");
        return;
      }

      const topK = parseInt(topKStr || "8", 10);

      setStatus("Asking question...");
      document.getElementById("answerBox").innerHTML = "";
      document.getElementById("sourcesBox").innerHTML = "";

      try {
        const resp = await fetch(baseUrl + "/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            doc_id: docId,
            question: question,
            top_k: topK
          }),
        });

        if (!resp.ok) {
          const text = await resp.text();
          console.error("Query error:", text);
          setStatus("Error during query: " + resp.status + " " + resp.statusText);
          return;
        }

        const data = await resp.json();
        console.log("Query response:", data);
        setStatus("Done.");

        // Show answer
        document.getElementById("answerBox").innerHTML =
          "<h3>Answer</h3><pre>" + (data.answer || "[no answer]") + "</pre>";

        // Show sources
        const sourcesDiv = document.getElementById("sourcesBox");
        sourcesDiv.innerHTML = "<h3>Sources</h3>";
        if (!data.sources || data.sources.length === 0) {
          sourcesDiv.innerHTML += "<p>No sources returned.</p>";
        } else {
          data.sources.forEach((s, idx) => {
            const md = s.metadata || {};
            const src = document.createElement("div");
            src.className = "source";
            src.innerHTML =
              '<div class="source-header">SOURCE ' + (idx + 1) +
              " – page " + (md.page_number ?? "?") +
              ', modality: ' + (md.modality ?? "?") +
              ', section: ' + (md.section || "") +
              "</div>" +
              "<div>" + (s.content ? s.content.substring(0, 400) : "") +
              (s.content && s.content.length > 400 ? "..." : "") +
              "</div>";
            sourcesDiv.appendChild(src);
          });
        }

      } catch (err) {
        console.error(err);
        setStatus("Error during query: " + err);
      }
    }
  </script>
</body>
</html>
    """


