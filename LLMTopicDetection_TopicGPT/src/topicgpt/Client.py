class Client:
    def __init__(self, api_key: str, azure_endpoint: dict = None) -> None:
        if azure_endpoint:
            from openai import AzureOpenAI
            self.client = AzureOpenAI(api_key=api_key, api_version=azure_endpoint['api_version'], azure_endpoint=azure_endpoint['endpoint'])
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
    
    def __getattr__(self, name):
        """Delegate attribute access to the self.client object."""
        return getattr(self.client, name)
