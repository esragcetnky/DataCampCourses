from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import pandas as pd
import yaml
# ----------------------------------------- Credential File  ---------------------------------------------------------#
CREDENTIALS_PATH = "credentials.yml"
credentials = yaml.safe_load(open(CREDENTIALS_PATH))

@tool
def financial_report(company_name: str, revenue: int, expenses: int) -> str:
    """Generate a financial report for a company that calculates net income."""    
    net_income = revenue - expenses    
    report = f"Financial Report for {company_name}:\n"    
    report += f"Revenue: ${revenue}\n"    
    report += f"Expenses: ${expenses}\n"    
    report += f"Net Income: ${net_income}\n"
    return report


print(financial_report.name)
print(financial_report.description)
print(financial_report.return_direct)
print(financial_report.args)
print("------------------------------------------------------\n")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=credentials["openai_api_key"], temperature=0)
agent = create_react_agent(llm, [financial_report])
messages = agent.invoke({"messages": [("human", "TechStack generated made $10 millionwith $8 million of costs. Generate a financial report.")]})
print(messages['messages'][-1].content)
