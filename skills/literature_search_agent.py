"""
Literature Search Agent (Gemini API version)
Uses the Gemini API with Google Search grounding to autonomously search for
and evaluate scientific literature on Raman/IR spectroscopy classification
of coffee, when current pipeline performance is below the reference
threshold.

This agent does NOT implement anything automatically. It searches, reasons,
and writes a structured recommendation for human review.

Requires: GEMINI_API_KEY environment variable set, and the google-genai package.
    pip install google-genai pandas
    Windows (PowerShell): $env:GEMINI_API_KEY = "AI..."

Get a free API key at: https://aistudio.google.com/apikey
"""
import os
import sys
import json
import pandas as pd
from google import genai
from google.genai import types

SKILLS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SKILLS_DIR)
RES_DIR = os.path.join(BASE_DIR, "results")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

MODEL = "gemini-2.5-flash"  # free-tier eligible; swap for a newer flash model if you have access
PERFORMANCE_THRESHOLD = 0.80


def build_context_summary():
    """Summarize what's already been tried, for the model's prompt."""
    lines = []

    cls_path = os.path.join(RES_DIR, "classification_results.csv")
    if os.path.exists(cls_path):
        df = pd.read_csv(cls_path).sort_values(by="Composite_Score", ascending=False)
        best = df.iloc[0]
        lines.append(f"Best classifier so far: {best['Model']} "
                     f"(Accuracy {best['Accuracy']:.1%}, F1_Macro {best['F1_Macro']:.1%})")
        lines.append("Full classifier leaderboard:")
        lines.append(df.to_string(index=False))
    else:
        lines.append("No classification results found yet.")

    prep_path = os.path.join(RES_DIR, "preprocessing_optimization_results.csv")
    if os.path.exists(prep_path):
        df = pd.read_csv(prep_path).sort_values(by="Optimization_Score", ascending=False)
        lines.append(f"\nBest preprocessing so far: {df.iloc[0]['Pipeline_Combination']}")

    base_path = os.path.join(RES_DIR, "baseline_comparison.csv")
    if os.path.exists(base_path):
        df = pd.read_csv(base_path).sort_values(by="Composite_Score", ascending=False)
        lines.append(f"Best baseline correction so far: {df.iloc[0]['Method']}")

    feat_path = os.path.join(RES_DIR, "feature_selection_results.csv")
    if os.path.exists(feat_path):
        df = pd.read_csv(feat_path).sort_values(by="Optimization_Score", ascending=False)
        lines.append(f"Best feature count so far: K={int(df.iloc[0]['K'])}")

    return "\n".join(lines)


def build_prompt(context_summary):
    return f"""You are a chemometrics research assistant helping evaluate whether
published literature suggests better methods for a coffee Raman spectroscopy
classification problem.

CURRENT PROBLEM:
130 Raman spectral samples, 6 classes (5 pure coffee origins with 20 samples
each: Arabica, Festtags, Guatemala, Lavazza, Sumatra; 1 Adulteration class
with 30 samples). Each spectrum has 785 wavenumber channels.

WHAT HAS ALREADY BEEN TRIED:
{context_summary}

KNOWN CHARACTERISTICS OF THIS DATA (from prior diagnostic analysis):
- The 5 pure origins are visually very similar to each other in mean
  spectrum -- the discriminating information appears spread across many
  channels rather than concentrated in a few peaks (smaller K values in
  feature selection consistently underperformed using all channels).
- The Adulteration class shows much higher within-class spectral variance
  than the pure-origin classes.
- Current best accuracy (70.0%, Logistic Regression) is BELOW our 80%
  reference threshold.

YOUR TASK:
1. Use Google Search to find scientific literature on Raman or FTIR/NIR
   spectroscopy classification of coffee, specifically covering
   origin/variety discrimination and/or adulteration detection.
2. Identify what preprocessing, feature selection, or classification
   techniques those papers used, and what performance they reported.
3. Evaluate which of those techniques, if any, plausibly address the
   specific characteristics of THIS dataset described above (don't just
   list techniques -- explain why a given technique would or wouldn't help
   given what's already been observed).
4. Recommend 1-3 specific, concrete next steps to try, ranked by how likely
   you think they are to help, with your reasoning.

IMPORTANT: only report papers you actually find via search. Do not invent
or guess at papers or citations.

Respond with ONLY the following JSON structure, no other text before or after:
{{
  "papers_found": [
    {{
      "citation": "Author(s), Year, Journal/Venue, brief title",
      "method_used": "what preprocessing/classification technique they used",
      "reported_performance": "what accuracy/result they reported, if stated",
      "relevance": "why this is or isn't relevant to our specific dataset characteristics"
    }}
  ],
  "recommendations": [
    {{
      "technique": "specific technique name",
      "rationale": "why this addresses our specific problem (spread-out discriminating info / high adulteration variance / etc.)",
      "priority": "high|medium|low",
      "implementation_notes": "brief technical note on how to implement this in our existing scikit-learn-based pipeline"
    }}
  ],
  "summary": "2-3 sentence overall summary of findings and recommended path forward"
}}
"""


def call_gemini_with_search(prompt):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[-] GEMINI_API_KEY environment variable not set.")
        print("    Get a free key at https://aistudio.google.com/apikey")
        print('    Then set it (PowerShell): $env:GEMINI_API_KEY = "AI..."')
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    print(f"[->] Calling Gemini API ({MODEL}) with Google Search grounding "
          f"(this may take 30-90s)...")
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=1.0,  # recommended for grounded responses
        ),
    )
    return response


def extract_sources(response):
    """Pull out the actual search results the model grounded on, for
    transparency -- lets you verify it wasn't inventing sources."""
    sources = []
    try:
        chunks = response.candidates[0].grounding_metadata.grounding_chunks
        if chunks:
            for chunk in chunks:
                if chunk.web:
                    sources.append({"title": chunk.web.title or "", "url": chunk.web.uri or ""})
    except (AttributeError, IndexError, TypeError):
        pass
    return sources


def main():
    print("[+] Literature Search Agent Initiated (Gemini API)...")

    context_summary = build_context_summary()
    print("\n[+] Current pipeline status:")
    print(context_summary)

    prompt = build_prompt(context_summary)
    response = call_gemini_with_search(prompt)

    raw_text = (response.text or "").strip()
    sources = extract_sources(response)

    # Try to parse the model's JSON response
    try:
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        print("[-] WARNING: could not parse model response as JSON. Saving raw text instead.")
        parsed = {"raw_response": raw_text}

    out_path = os.path.join(RES_DIR, "literature_recommendations.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"recommendations": parsed, "sources_searched": sources}, f, indent=2)
    print(f"\n[+] Structured recommendations saved to {out_path}")

    md_lines = ["# Literature Search Findings\n"]
    md_lines.append("*Generated by literature_search_agent.py (Gemini API) -- review before "
                    "implementing any changes.*\n")

    if "papers_found" in parsed:
        md_lines.append("## Papers Found\n")
        for paper in parsed.get("papers_found", []):
            md_lines.append(f"**{paper.get('citation', 'Unknown')}**")
            md_lines.append(f"- Method used: {paper.get('method_used', 'N/A')}")
            md_lines.append(f"- Reported performance: {paper.get('reported_performance', 'N/A')}")
            md_lines.append(f"- Relevance: {paper.get('relevance', 'N/A')}\n")

        md_lines.append("## Recommendations\n")
        for rec in parsed.get("recommendations", []):
            md_lines.append(f"### {rec.get('technique', 'Unknown')} (Priority: {rec.get('priority', 'N/A')})")
            md_lines.append(f"- Rationale: {rec.get('rationale', 'N/A')}")
            md_lines.append(f"- Implementation notes: {rec.get('implementation_notes', 'N/A')}\n")

        md_lines.append("## Summary\n")
        md_lines.append(parsed.get("summary", "N/A") + "\n")
    else:
        md_lines.append("## Raw Response (JSON parsing failed)\n")
        md_lines.append(parsed.get("raw_response", "No response captured."))

    if sources:
        md_lines.append("## Sources Actually Searched\n")
        for s in sources:
            md_lines.append(f"- [{s['title']}]({s['url']})")

    md_path = os.path.join(REPORTS_DIR, "literature_search_findings.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[+] Readable summary saved to {md_path}")

    print("\n" + "="*60)
    print("  LITERATURE SEARCH COMPLETE -- Stopping for user review.")
    print("  Review reports/literature_search_findings.md before")
    print("  implementing any recommended technique.")
    print("="*60)


if __name__ == "__main__":
    main()