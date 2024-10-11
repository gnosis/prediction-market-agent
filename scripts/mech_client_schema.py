# ToDo - Build mech-tools enum dynamically

from mech_client.mech_tool_management import get_tool_io_schema
from pydantic import BaseModel, Field, ConfigDict, create_model

from prediction_market_agent.tools.mech.utils import mech_request


class OlasAgent(BaseModel):
    agent_id: int
    tool_ids: list[str]


class OlasTool(BaseModel):
    tool_name: str
    unique_identifier: str


class Property(BaseModel):
    description: str
    type: str


class OutputSchema(BaseModel):
    type: str
    properties: dict[str, Property]
    required: list[str]


class InputConfig(BaseModel):
    type: str
    description: str


class OutputConfig(BaseModel):
    type: str
    description: str
    schema: OutputSchema


class ModelSchema(BaseModel):
    input_schema: InputConfig = Field(..., alias="input")
    output_schema: OutputConfig = Field(..., alias="output")
    model_config = ConfigDict(populate_by_name=True)


OLAS_TYPES_TO_PYTHON_TYPES = {
    "string": str,
    "integer": int,
}


# ToDo - Build langchain tool (or similar) from mech-tools

if __name__ == "__main__":
    # all_tools = get_tools_for_agents(agent_id=None, chain_config='gnosis')

    # tool_container = {i['tool_name']: OlasTool.model_validate(i) for i in all_tools['all_tools_with_identifiers']}
    # agents = []
    # agents = [OlasAgent(agent_id=agent_id, tool_ids=tool_ids) for agent_id, tool_ids in
    #          all_tools['agent_tools_map'].items()]
    # Get tool description #unique_tools = set([tool_id for agent in agents for tool_id in agent.tool_ids])
    # unique_tools = list(set((chain.from_iterable([agent.tool_ids for agent in agents]))))
    # tool_id = tool_container[unique_tools[0]]
    tool_id_identifier = "3-openai-gpt-3.5-turbo"
    io_schema = get_tool_io_schema(tool_id_identifier, "gnosis")
    # print(io_schema)
    model = ModelSchema.model_validate(io_schema)

    # create Pydantic model dynamically
    dict_schema = {}
    for k, v in model.output_schema.schema.properties.items():
        dict_schema[k] = (OLAS_TYPES_TO_PYTHON_TYPES[v.type], ...)

    DynamicModel = create_model("DynamicModel", **dict_schema)

    # ToDo - Call mech and get response
    # ToDo - build response dynamically from output schema

    # mock_response = {
    #     "requestId": 24799777447209263141401480763500015573309571847555969842339861220699351973823,
    #     "result": "Sure! Here's one for you:\n\nWhy couldn't the bicycle stand up by itself?\n\nBecause it was two tired!",
    #     "prompt": "Tell me a joke",
    #     "cost_dict": {},
    #     "metadata": {"model": None, "tool": "openai-gpt-3.5-turbo", "params": {}},
    # }

    response = mech_request("Tell me a joke", "openai-gpt-3.5-turbo", agent_id=3)
    print(response)

    response_parsed = DynamicModel.model_validate(response)

    # function_call = lambda x: mech_request()

    # ToDo - Call function dynamically using attrs
    # ToDo - get agent ID description
    # ToDo - Create function (see dynamic_class = ClassFactory().create_class(class_name, (base,), attributes))
    # class_name = tool_id_identifier.replace("-", " ").title().replace(" ", "")
    # attributes = {
    #     "__name__": class_name,
    #     "__call__": dynamic_function,
    #     "description": summary.summary,
    #     "example_args": example_args,
    # }
    #
    # dc = ClassFactory().create_class("class1", (Function,), {"__call__": lambda x: 1})
