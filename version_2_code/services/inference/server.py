from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

class GenerationRequest(BaseModel):
    text: str
    max_length: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 0.9
    n_alternatives: Optional[int] = None

class GenerationResponse(BaseModel):
    completions: List[dict]
    input_text: str
    metadata: dict

app = FastAPI()

class InferenceServer:
    def __init__(self, generator):
        """
        Initialize inference server
        Args:
            generator: Sequence generator instance
        """
        self.generator = generator
        self.app = app

        @app.post("/generate", response_model=GenerationResponse)
        async def generate(request: GenerationRequest):
            try:
                if request.n_alternatives:
                    completions = self.generator.generate_alternatives(
                        request.text,
                        n_alternatives=request.n_alternatives
                    )
                else:
                    completions = self.generator.generate_continuation(
                        request.text,
                        max_length=request.max_length,
                        temperature=request.temperature,
                        top_p=request.top_p
                    )
                
                return GenerationResponse(
                    completions=completions,
                    input_text=request.text,
                    metadata={
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "max_length": request.max_length
                    }
                )
            
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    def run(self, host="0.0.0.0", port=8000):
        """Run the inference server"""
        import uvicorn
        uvicorn.run(app, host=host, port=port)
