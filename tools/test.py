import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from LLM.llm import SarvamChat  


llm = SarvamChat(api_key="SARVAM_API_KEY")
response = llm.invoke("Tell me about AI in one line")
ans = response.content.strip('" "')
print(ans)
