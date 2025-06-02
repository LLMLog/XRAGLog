import openai

openai.api_key = "YOUR_KEY"


def gpt3_5_function_call(prompt):
    retry_time = 5
    while retry_time > 0:
        try:
            completion = openai.ChatCompletion.create(model='gpt-3.5-turbo', temperature=0.5,
                                                      top_p=1,
                                                      n=1,
                                                      stream=False,
                                                      stop=None,
                                                      presence_penalty=0,
                                                      frequency_penalty=0,
                                                      logit_bias={},
                                                      messages=[
                                                          {"role": "user", "content": prompt}
                                                      ],
                                                      functions=[
                                                          {
                                                              "name": "anomaly_classification",
                                                              "description": "classify there exist an anomaly or not",
                                                              "parameters": {
                                                                  "type": "object",
                                                                  # specify that the parameter is an object
                                                                  "properties": {
                                                                      "has_anomaly": {
                                                                          "type": "number",
                                                                          # specify the parameter type as a string
                                                                          "enum": [0, 1],
                                                                          "description": "0: no anomaly, 1: has anomaly",
                                                                      },
                                                                      "thought process": {
                                                                          "type": "string",
                                                                          "description": "the thought process"
                                                                      }
                                                                  },
                                                                  "required": ["has_anomaly"]
                                                                  # specify that the location parameter is required
                                                              }
                                                          }
                                                      ],
                                                      function_call={"name": "anomaly_classification"}
                                                      )
            return completion.choices[0].message.function_call.arguments
        except Exception as e:
            print(e)
            retry_time -= 1
    return ""


def get_embedding(text):
    retry_time = 5
    while retry_time > 0:
        try:
            result = openai.Embedding.create(
                model='text-embedding-ada-002',
                input=text
            )
            return result["data"][0]["embedding"]
        except Exception as e:
            print(e)
            retry_time -= 1
    return []
