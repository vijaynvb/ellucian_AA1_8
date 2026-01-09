from config import bedrock_client, BEDROCK_AGENT_ID, BEDROCK_AGENT_ALIAS_ID, process_event_stream

def invoke_agent(user_message):
    try:
        response = bedrock_client.invoke_agent(
            agentId=BEDROCK_AGENT_ID,
            agentAliasId=BEDROCK_AGENT_ALIAS_ID,
            inputText=user_message,
            sessionId="streamlit-session"
        )
        print("Bedrock Agent Response:", response)
        # agent_response = response['outputText']
        # return the generator so the caller can stream chunks as they arrive
        return process_event_stream(response)
    except Exception as e:  
        return f"Error invoking agent: {str(e)}"
