import pandas as pd
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

# Up-to-date Standalone Imports
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

# 1. Initialize Local LLM (llama3.1 supports Native Tool Calling)
llm = ChatOllama(model="llama3.1", temperature=0.0)

# 2. Define Local Excel-Based NAICS Search Tool
EXCEL_FILE_PATH = "NAICS_Active_Codes_Reference.xlsx"

@tool
def search_excel_naics_table(query: str) -> str:
    """Search the local NAICS Excel file directly by keyword to find matching active codes and descriptions."""
    try:
        df = pd.read_excel(EXCEL_FILE_PATH, sheet_name="NAICS 2022 Codes")
        df["NAICS Code"] = df["NAICS Code"].astype(str)
        
        query_lower = query.lower()
        matches = df[
            (df["Status"] == "Active") & (
                df["NAICS Code"].str.lower().str.contains(query_lower) |
                df["Sector / Industry Title"].str.lower().str.contains(query_lower) |
                df["Description & Scope Summary"].str.lower().str.contains(query_lower)
            )
        ]
        
        if matches.empty:
            return f"No active NAICS codes found in Excel matching '{query}'."
            
        result_df = matches[["NAICS Code", "Sector / Industry Title", "NAICS Level", "Description & Scope Summary"]]
        return result_df.to_string(index=False)
        
    except Exception as e:
        return f"Error reading NAICS Excel file: {str(e)}"

@tool
def search_web_for_company(query: str) -> str:
    """Search the web for company details, business operations, and industry given name and address."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No search results found."
        
        formatted_results = "\n".join([f"Title: {r['title']}\nSnippet: {r['body']}\nURL: {r['href']}" for r in results])
        return formatted_results
    except Exception as e:
        return f"Web search error: {str(e)}"

@tool
def scrape_company_website(url: str) -> str:
    """Scrape text content from the company website URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        
        text_blocks = [p.text.strip() for p in soup.find_all(['p', 'h1', 'h2', 'h3']) if p.text.strip()]
        full_text = " ".join(text_blocks)
        return full_text[:4000] if full_text else "Website returned empty text content."
    except Exception as e:
        return f"Error scraping website: {str(e)}"

tools = [search_web_for_company, scrape_company_website, search_excel_naics_table]

# 3. Create Agent Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an expert compliance agent running completely locally.\n"
     "Your task is to classify a company under the correct active NAICS code using their Name, Address, and Website.\n\n"
     "Steps:\n"
     "1. Use 'search_web_for_company' with the company name, address, and details to understand what they do.\n"
     "2. Use 'scrape_company_website' if a direct URL is given.\n"
     "3. Identify core activities and search the local Excel sheet using 'search_excel_naics_table'.\n"
     "4. Verify the chosen code status is 'Active'.\n"
     "5. Return the NAICS code, exact title, and a clear reason."),
    ("human", "Company Name: {company_name}\nAddress: {address}\nWebsite: {website}"),
    MessagesPlaceholder("agent_scratchpad")
])

# 4. Construct Agent Executor
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. Run Classification Test
if __name__ == "__main__":
    print("\n--- Starting NAICS Agent ---")
    result = agent_executor.invoke({
        "company_name": "Microsoft Corporation",
        "address": "One Microsoft Way, Redmond, WA",
        "website": "https://www.microsoft.com"
    })
    
    print("\n--- Final Agent Classification Output ---")
    print(result["output"])