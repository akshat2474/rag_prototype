import os
import json
import re
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from rag_handler import MacSafeRAG, OpenRouterClient

load_dotenv()

CSV_FILE_PATH = "filtered_150k.csv"
INDEX_SAVE_PATH = "./my_rag_index"
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting RAG System...")
    rag = MacSafeRAG()
    if os.path.exists(f"{INDEX_SAVE_PATH}/faiss.index"):
        print(f"✅ Found existing index at '{INDEX_SAVE_PATH}'. Loading...")
        rag.load_index(INDEX_SAVE_PATH)
    else:
        print(f"📊 Index not found. Processing '{CSV_FILE_PATH}'...")
        print("⏳ This might take a while but only happens once.")
        rag.process_safely(CSV_FILE_PATH, batch_size=500)
        rag.save_index(INDEX_SAVE_PATH)
        print("🎉 Processing complete! Index has been saved.")
    app_state["rag_client"] = rag
    app_state["openrouter_client"] = OpenRouterClient(api_key=OPENROUTER_KEY)
    print("\n🤖 RAG System ready.")
    yield
    app_state.clear()
    print("✅ RAG System shut down.")

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str
    model: str = "x-ai/grok-4-fast:free"

class QueryResponse(BaseModel):
    answer: str

class GraphPoint(BaseModel):
    x: float
    y: float

class GraphResponse(BaseModel):
    points: List[GraphPoint]

class GraphRequest(BaseModel):
    question: str
    model: str = "x-ai/grok-4-fast:free"


def extract_points_from_text(text: str):
    """Heuristic regex to find pairs of floats (like "123.4, 56.7" or "x:123.4 y:56.7")"""
    pattern = re.compile(r'(\d+\.?\d*)[,\s:]+(\d+\.?\d*)')
    points = []
    for match in pattern.finditer(text):
        try:
            x = float(match.group(1))
            y = float(match.group(2))
            points.append({"x": x, "y": y})
        except:
            continue
    return points


def extract_json_array(text: str):
    """Extract JSON array from text response"""
    # Try to find JSON array pattern
    json_match = re.search(r'\[\s*{.*?}\s*\]', text, re.DOTALL)
    if json_match:
        try:
            json_data = json_match.group(0)
            return json.loads(json_data)
        except json.JSONDecodeError:
            pass
    
    # Try to find simpler array pattern
    array_match = re.search(r'\[[\s\d\.,{}"xy:-]+\]', text, re.DOTALL)
    if array_match:
        try:
            json_data = array_match.group(0)
            return json.loads(json_data)
        except json.JSONDecodeError:
            pass
    
    return None


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    rag_client = app_state.get("rag_client")
    openrouter_client = app_state.get("openrouter_client")

    if not rag_client or not openrouter_client:
        raise HTTPException(status_code=503, detail="RAG system not ready.")

    try:
        print(f"🔍 Searching context for: '{request.question}'")
        rag_result = rag_client.query(request.question, k=5)

        print(f"🧠 Generating answer with model: '{request.model}'...")
        final_answer = openrouter_client.query_with_context(
            question=request.question,
            context=rag_result['context'],
            results=rag_result['results'],
            model=request.model
        )
        print("💬 Answer generated.")
        return QueryResponse(answer=final_answer)

    except Exception as e:
        print(f"🔥 Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/graph", response_model=GraphResponse)
async def generate_graph(request: GraphRequest):
    rag_client = app_state.get("rag_client")
    openrouter_client = app_state.get("openrouter_client")

    if not rag_client or not openrouter_client:
        raise HTTPException(status_code=503, detail="RAG system not ready.")

    try:
        print(f"🔍 Searching context for graph: '{request.question}'")
        rag_result = rag_client.query(request.question, k=5)
        context = rag_result['context']

        system_prompt = (
            "You are a data visualization assistant. Your task is to extract numerical data from the provided context "
            "and return it as a JSON array of coordinate points for plotting. "
            "Each point should be an object with numeric fields 'x' and 'y'. "
            "Analyze the context to identify relevant numerical relationships that can be plotted. "
            "CRITICAL: Respond ONLY with a valid JSON array, no explanations or additional text.\n\n"
            "Example format:\n"
            "[\n"
            "  {\"x\": 0.0, \"y\": 12.3},\n"
            "  {\"x\": 1.0, \"y\": 14.5},\n"
            "  {\"x\": 2.0, \"y\": 13.2}\n"
            "]"
        )

        user_prompt = (
            f"Based on this oceanographic data context, extract numerical data points that answer the question: '{request.question}'\n\n"
            f"Context:\n{context}\n\n"
            "Extract relevant x,y coordinate pairs from the data. For example:\n"
            "- If asking about temperature vs pressure: x=pressure, y=temperature\n"
            "- If asking about depth vs salinity: x=depth, y=salinity\n"
            "- If asking about time series: x=time/date, y=measured_value\n\n"
            "Return ONLY the JSON array of points, nothing else:"
        )

        print(f"🧠 Generating graph data with model: '{request.model}'...")
        
        # Use the specialized graph generation method
        response_text = openrouter_client.generate_graph_data(
            question=request.question,
            context=context,
            model=request.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

        print(f"📊 LLM graph raw response:\n{response_text}")

        # Try to extract JSON array from response
        points_raw = extract_json_array(response_text)
        
        if not points_raw:
            # Fallback: try heuristic numeric extraction
            print("⚠️  JSON extraction failed, trying heuristic extraction...")
            points_raw = extract_points_from_text(response_text)
            
            if not points_raw:
                # Last resort: try to find individual numbers and pair them
                numbers = re.findall(r'\d+\.?\d*', response_text)
                if len(numbers) >= 2:
                    points_raw = []
                    for i in range(0, len(numbers) - 1, 2):
                        try:
                            points_raw.append({
                                "x": float(numbers[i]),
                                "y": float(numbers[i + 1])
                            })
                        except:
                            continue

        if not points_raw:
            raise ValueError("No graph data could be extracted from the response.")

        # Convert to GraphPoint objects
        points = []
        for pt in points_raw:
            try:
                x_val = float(pt.get('x', 0))
                y_val = float(pt.get('y', 0))
                points.append(GraphPoint(x=x_val, y=y_val))
            except (ValueError, TypeError):
                continue

        if not points:
            raise ValueError("No valid coordinate points could be created.")

        print(f"✅ Successfully extracted {len(points)} data points")
        return GraphResponse(points=points)

    except Exception as e:
        print(f"🔥 Graph generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Graph generation error: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    rag_ready = "rag_client" in app_state and app_state["rag_client"] is not None
    openrouter_ready = "openrouter_client" in app_state and app_state["openrouter_client"] is not None
    
    return {
        "status": "healthy" if rag_ready and openrouter_ready else "initializing",
        "rag_ready": rag_ready,
        "openrouter_ready": openrouter_ready
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)